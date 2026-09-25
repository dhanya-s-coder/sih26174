import cv2
import time
import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from src.har_space.config.runtime import RuntimeConfig
from src.har_space.config.models import ExperimentSpec
from src.har_space.io.source import WebcamSource, FileSource, RTSPSource
from src.har_space.io.recorder import LocalRecorder
from src.har_space.io.streamer import VideoStreamer
from src.har_space.io.models import Frame
from src.har_space.events.bus import EventBus
from src.har_space.data_models import Event
from src.har_space.perception.color_detector import ColorBoxDetector
from src.har_space.perception.hands import HandLandmarker
from src.har_space.events.interactions import InteractionExtractor
from src.har_space.engine.tracker import SimpleStepTracker
from src.har_space.alerts.voice import VoiceAlerter
from src.har_space.logging_.writer import StructuredLogWriter
from src.har_space.gui.overlay import draw_overlay
from src.har_space.data_models import Alert

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0")
    parser.add_argument("--experiment", default="configs/experiment_demo.yaml")
    parser.add_argument("--mute", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--save-annotated", action="store_true")
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()

    # Init Run Dir
    run_dir = Path("runs") / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    
    spec = ExperimentSpec.load_from_yaml(args.experiment)
    bus = EventBus()
    
    log_writer = StructuredLogWriter(str(run_dir / "steps.jsonl"))
    
    if args.source.isdigit():
        source = WebcamSource(int(args.source))
    elif args.source.startswith("rtsp"):
        source = RTSPSource(args.source)
    else:
        source = FileSource(args.source, realtime_pacing=True, loop=False)

    streamer = None
    if args.stream:
        streamer = VideoStreamer("mjpeg", "0.0.0.0", 8080)
        
    annotated_recorder = None
    if args.save_annotated:
        annotated_recorder = LocalRecorder(str(run_dir), segment_length_minutes=60)

    # Core Components
    color_det = ColorBoxDetector()
    try:
        hand_lm = HandLandmarker()
    except Exception as e:
        print(e)
        return
        
    extractor = InteractionExtractor(bus)
    tracker = SimpleStepTracker(spec, bus, log_writer)
    voice = VoiceAlerter(bus, mute=args.mute)
    
    # State for UI
    active_alert = None
    object_tracks = {
        "yellow_box": {"inside": False, "removed": False, "removal_frames": 0, "hand_seen": False, "last_center": None, "stable_frames": 0, "placed": False, "hand_contact": False},
        "red_box": {"inside": False, "removed": False, "removal_frames": 0, "hand_seen": False, "last_center": None, "stable_frames": 0, "placed": False, "hand_contact": False},
    }
    def alert_handler(e):
        nonlocal active_alert
        active_alert = Alert(**e.payload)
    bus.subscribe("alert", alert_handler)
    
    # Start
    source.start()
    voice.start()
    if streamer: streamer.start()
    if annotated_recorder: annotated_recorder.start(30.0, (640, 480))
    
    # Initial voice
    bus.publish(type("Event", (), {"event_type": "speech", "payload": {"text": spec.steps[0].instruction}})())

    print(f"Run started in {run_dir}. Press Q to quit, R to reset.")

    try:
        while True:
            frame = source.get_frame(timeout=0.1)
            if not frame:
                if not source.running:
                    break
                continue
                
            fps, _ = source.get_stats()
            
            # Perception
            objects = color_det.process(frame.image)
            hands = hand_lm.process(frame.image)

            # The main box is a visual precondition, not a hand action. Once
            # the white main box is detected, advance the FSM to the first
            # actual manipulation step. Do not require its contents to be
            # visible in the same frame.
            if any(obj.label == "main_box" for obj in objects):
                main_box = next(obj for obj in objects if obj.label == "main_box")
                bus.publish(Event(
                    event_type="interaction",
                    timestamp=frame.video_timestamp,
                    payload={
                        "subject": "object",
                        "verb": "present",
                        "object": "main_box",
                    },
                ))

                # Track colored objects relative to the detected main box.
                # This creates removal and stable-placement FSM events without
                # requiring the box contents to be visible at startup.
                mx1, my1, mx2, my2 = main_box.bbox
                for obj in objects:
                    if obj.label not in object_tracks:
                        continue
                    track = object_tracks[obj.label]
                    cx = (obj.bbox[0] + obj.bbox[2]) / 2.0
                    cy = (obj.bbox[1] + obj.bbox[3]) / 2.0
                    # A hand usually grips the top/edge of a box, so its
                    # bounding-box center will not be inside the hand box.
                    # Treat contact as true when either hand bbox overlaps
                    # the object bbox or a hand landmark lies in the object.
                    def overlaps(a, b):
                        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
                        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
                        return ix2 > ix1 and iy2 > iy1

                    hand_contact = any(
                        overlaps(h.bbox, obj.bbox) or
                        any(obj.bbox[0] <= lm[0] <= obj.bbox[2] and obj.bbox[1] <= lm[1] <= obj.bbox[3]
                            for lm in h.landmarks)
                        for h in hands
                    )
                    track["hand_contact"] = hand_contact
                    inside = mx1 <= cx <= mx2 and my1 <= cy <= my2
                    if inside:
                        track["inside"] = True
                        track["stable_frames"] = 0
                        track["last_center"] = (cx, cy)
                        continue
                    # Contents may be hidden inside the main box at startup,
                    # so they do not need to be detected while inside it.
                    # The first valid observation can be outside the box when
                    # the hand lifts the object into view.
                    if hand_contact:
                        track["hand_seen"] = True

                    if not inside and not track["removed"]:
                        track["removal_frames"] += 1
                    else:
                        track["removal_frames"] = 0

                    # Require the same confirmation window for both colors;
                    # this prevents a box from being marked removed on the
                    # first frame in which it briefly appears outside.
                    if track["removal_frames"] >= 5 and track["hand_seen"] and not track["removed"]:
                        track["removed"] = True
                        bus.publish(Event(
                            event_type="interaction", timestamp=frame.video_timestamp,
                            payload={"subject": "object", "verb": "removed", "object": obj.label, "container": "main_box"}
                        ))
                    # Left/right placement is based on a stable object center.
                    previous = track["last_center"]
                    moved = previous is None or abs(cx - previous[0]) + abs(cy - previous[1]) < 0.02
                    track["stable_frames"] = track["stable_frames"] + 1 if moved and not hand_contact else 0
                    track["last_center"] = (cx, cy)
                    if track["removed"] and track["stable_frames"] >= 10 and not track["placed"]:
                        # Placement is relative to the main box, not the
                        # camera frame. Require the object to be horizontally
                        # outside the corresponding edge and vertically
                        # aligned with the main box.
                        vertical_overlap = obj.bbox[3] >= my1 and obj.bbox[1] <= my2
                        if vertical_overlap and cx < mx1:
                            location = "left"
                        elif vertical_overlap and cx > mx2:
                            location = "right"
                        else:
                            location = None
                        expected = "left" if obj.label == "yellow_box" else "right"
                        if location == expected:
                            track["placed"] = True
                            bus.publish(Event(
                                event_type="interaction", timestamp=frame.video_timestamp,
                                payload={"subject": "object", "verb": "placed", "object": obj.label, "location": location}
                            ))

            # Interactions
            extractor.process(hands, objects, frame.video_timestamp)
            
            # Overlay
            annotated_img = draw_overlay(frame.image.copy(), objects, hands, spec, 
                                         tracker.current_step_idx, tracker.completed_steps,
                                         active_alert, fps)
                                         
            annotated_frame = Frame(frame_id=frame.frame_id, 
                                    wall_clock_timestamp=frame.wall_clock_timestamp,
                                    video_timestamp=frame.video_timestamp,
                                    image=annotated_img, source_name="annotated")
                                    
            if annotated_recorder:
                annotated_recorder.push(annotated_frame)
            if streamer:
                streamer.push(annotated_frame)
                
            if not args.headless:
                cv2.imshow("Live Demo", annotated_img)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    print("Resetting tracker...")
                    tracker.current_step_idx = 0
                    tracker.completed_steps.clear()
                    active_alert = None

    except KeyboardInterrupt:
        pass
    finally:
        print("Shutting down...")
        source.stop()
        voice.stop()
        if streamer: streamer.stop()
        if annotated_recorder: annotated_recorder.stop()
        log_writer.write_summary(str(run_dir / "summary.json"))
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
