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
