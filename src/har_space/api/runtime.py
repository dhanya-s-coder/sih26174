"""Host the existing HAR components for the local API without duplicating inference."""

from __future__ import annotations

import threading
import time
import queue
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src.har_space.alerts.voice import VoiceAlerter
from src.har_space.config.models import ExperimentSpec
from src.har_space.data_models import Event
from src.har_space.engine.tracker import SimpleStepTracker
from src.har_space.events.bus import EventBus
from src.har_space.events.interactions import InteractionExtractor
from src.har_space.gui.overlay import draw_overlay
from src.har_space.gui.runtime_state import RuntimeState
from src.har_space.io.models import Frame
from src.har_space.io.recorder import LocalRecorder
from src.har_space.io.source import FileSource, RTSPSource, WebcamSource
from src.har_space.io.streamer import VideoStreamer
from src.har_space.logging_.writer import StructuredLogWriter
from src.har_space.perception.color_detector import ColorBoxDetector
from src.har_space.perception.hands import HandLandmarker


@dataclass
class _ResetCommand:
    applied: threading.Event = field(default_factory=threading.Event)
    succeeded: bool = False


class HARApplication:
    """Own exactly one instance of each existing HAR pipeline component."""

    def __init__(
        self,
        source_arg: str = "0",
        experiment_path: str = "configs/experiment_demo.yaml",
        mute: bool = False,
        save_annotated: bool = False,
        stream_host: str = "127.0.0.1",
        stream_port: int = 8081,
    ) -> None:
        self.run_dir = Path("runs") / time.strftime("%Y%m%d_%H%M%S")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.spec = ExperimentSpec.load_from_yaml(experiment_path)
        self.bus = EventBus()
        self.log_path = self.run_dir / "steps.jsonl"
        self.log_writer = StructuredLogWriter(str(self.log_path))
        self.runtime = RuntimeState(
            str(self.log_path), len(self.spec.steps), tuple(step.name for step in self.spec.steps)
        )
        for event_type in ("interaction", "alert", "speech", "experiment_reset"):
            self.bus.subscribe(event_type, self.runtime.record_event)

        self.source = self._make_source(source_arg)
        self.detector_error = ""
        try:
            self.detector = ColorBoxDetector()
        except Exception as exc:
            self.detector = None
            self.detector_error = str(exc)
        self.hand_error = ""
        try:
            self.hand_landmarker = HandLandmarker()
        except Exception as exc:
            self.hand_landmarker = None
            self.hand_error = str(exc)

        self.interaction_extractor = InteractionExtractor(self.bus)
        self.tracker = SimpleStepTracker(self.spec, self.bus, self.log_writer)
        self.voice = VoiceAlerter(self.bus, mute=mute)
        self.recorder = LocalRecorder(str(self.run_dir), segment_length_minutes=60) if save_annotated else None
        self.streamer = VideoStreamer("mjpeg", stream_host, stream_port)
        self.stop_event = threading.Event()
        self.reset_event = threading.Event()
        self._commands: queue.Queue[_ResetCommand] = queue.Queue()
        self.pipeline_error = ""
        self.pipeline_thread: Optional[threading.Thread] = None
        self._record_cursor = 0
        self._last_video_timestamp = 0.0
        self._started = False

    @staticmethod
    def _make_source(source_arg: str):
        if source_arg.isdigit():
            return WebcamSource(int(source_arg))
        if source_arg.lower().startswith("rtsp"):
            return RTSPSource(source_arg)
        return FileSource(source_arg, realtime_pacing=True, loop=False)

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self.source.start()
        self.voice.start()
        self.streamer.start()
        if self.recorder:
            self.recorder.start(30.0, (640, 480))
        self.runtime.update_health(self.health_snapshot())
        if self.spec.steps:
            instruction = self.spec.steps[0].instruction
            if instruction and instruction.strip():
                self.bus.publish(Event(
                    event_type="speech", timestamp=time.time(),
                    payload={"text": instruction},
                ))
        self.pipeline_thread = threading.Thread(
            target=self._process_frames, name="har-pipeline", daemon=True
        )
        self.pipeline_thread.start()

    def stop(self) -> None:
        if not self._started:
            return
        self.stop_event.set()
        self.source.stop()
        if self.pipeline_thread and self.pipeline_thread.is_alive():
            self.pipeline_thread.join(timeout=4.0)
        self._fail_pending_commands()
        self.voice.stop()
        self.streamer.stop()
        if self.recorder:
            self.recorder.stop()
        self.log_writer.write_summary(str(self.run_dir / "summary.json"))
        self.runtime.update_health(self.health_snapshot())
        self._started = False

    def request_reset(self) -> None:
        """Queue a reset to be applied by the sole pipeline worker."""
        self.reset_event.set()

    def reset_experiment(self, timeout: float = 5.0) -> bool:
        """Request a tracker reset and wait until the pipeline worker applies it."""
        if not self._started or not self.pipeline_thread or not self.pipeline_thread.is_alive():
            return False
        command = _ResetCommand()
        self._commands.put(command)
        if not command.applied.wait(timeout):
            return False
        return command.succeeded

    def _apply_pending_commands(self) -> None:
        while True:
            try:
                command = self._commands.get_nowait()
            except queue.Empty:
                return
            try:
                self.tracker.reset()
                self.interaction_extractor.reset()
                self.runtime.reset_protocol_view()
                self.bus.publish(Event(
                    event_type="experiment_reset",
                    timestamp=self._last_video_timestamp,
                    payload={"message": "Experiment reset"},
                ))
                if self.spec.steps:
                    instruction = self.spec.steps[0].instruction
                    if instruction and instruction.strip():
                        self.bus.publish(Event(
                            event_type="speech",
                            timestamp=self._last_video_timestamp,
                            payload={"text": instruction, "reset_queue": True},
                        ))
                command.succeeded = True
            except Exception as exc:
                self.pipeline_error = str(exc)
                self.runtime.set_error(f"Experiment reset failed: {exc}")
            finally:
                command.applied.set()

    def _fail_pending_commands(self) -> None:
        while True:
            try:
                command = self._commands.get_nowait()
            except queue.Empty:
                return
            command.succeeded = False
            command.applied.set()

    def health_snapshot(self) -> dict[str, tuple[str, str]]:
        return {
            "camera": (self.source.status, self.source.error_message),
            "hands": (("RUNNING", "") if self.hand_landmarker else ("ERROR", self.hand_error)),
            "detector": (("RUNNING", "") if self.detector else ("ERROR", self.detector_error)),
            "interaction": ("RUNNING", ""),
            "tracker": (("ERROR", self.pipeline_error) if self.pipeline_error else ("RUNNING", "")),
            "tts": (self.voice.status, self.voice.error_message),
            "logging": (("ACTIVE", "") if self.log_path.exists() else ("ERROR", "Log file unavailable")),
            "recording": ((self.recorder.status, self.recorder.error_message)
                          if self.recorder else ("INACTIVE", "")),
            "streaming": (self.streamer.status, self.streamer.error_message),
        }

    def _record_new_steps(self) -> None:
        new_records = self.log_writer.records[self._record_cursor:]
        for record in new_records:
            self.runtime.record_step(record)
        self._record_cursor += len(new_records)

    def _process_frames(self) -> None:
        try:
            while not self.stop_event.is_set():
                self._apply_pending_commands()
                if self.reset_event.is_set():
                    self.reset_event.clear()
                    command = _ResetCommand()
                    self._commands.put(command)
                    self._apply_pending_commands()

                frame = self.source.get_frame(timeout=0.1)
                if frame is None:
                    self.runtime.update_health(self.health_snapshot())
                    if not self.source.running:
                        break
                    continue

                fps, _ = self.source.get_stats()
                self._last_video_timestamp = frame.video_timestamp
                detections = self.detector.process(frame.image) if self.detector else []
                hands = self.hand_landmarker.process(frame.image) if self.hand_landmarker else []
                self.interaction_extractor.process(hands, detections, frame.video_timestamp)
                self._record_new_steps()

                prior = self.runtime.snapshot()
                annotated_image = draw_overlay(
                    frame.image.copy(), detections, hands, self.spec,
                    self.tracker.current_step_idx, self.tracker.completed_steps,
                    prior.latest_alert, fps,
                    show_telemetry=False,
                    interaction_text=prior.current_interaction,
                )
                annotated_frame = Frame(
                    frame_id=frame.frame_id,
                    wall_clock_timestamp=frame.wall_clock_timestamp,
                    video_timestamp=frame.video_timestamp,
                    image=annotated_image,
                    source_name="har-processed",
                )
                self.streamer.push(annotated_frame)
                if self.recorder:
                    self.recorder.push(annotated_frame)

                self.runtime.update_pipeline(
                    frame=annotated_image,
                    frame_id=frame.frame_id,
                    detections=tuple(detections),
                    hands=tuple(hands),
                    step_index=self.tracker.current_step_idx,
                    completed_step_ids=frozenset(self.tracker.completed_steps),
                    active_predicates=tuple(sorted(self.tracker.active_predicates)),
                    fps=fps,
                    health=self.health_snapshot(),
                    recording_path=(self.recorder.current_segment_path or str(self.recorder.output_dir))
                    if self.recorder else "",
                    streaming_endpoint=self.streamer.endpoint
                    if self.streamer.status in {"ACTIVE", "LISTENING"} else "",
                )
        except Exception as exc:
            self.pipeline_error = str(exc)
            self.runtime.set_error(str(exc))
        finally:
            self._record_new_steps()
            self._fail_pending_commands()
            self.runtime.update_health(self.health_snapshot())
            self.runtime.mark_stopped()
