import cv2
import time
import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from src.har_space.config.runtime import RuntimeConfig
from src.har_space.config.models import ExperimentSpec
from src.har_space.io.source import WebcamSource, RTSPSource
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
from src.har_space.dashboard.server import global_state, start_dashboard_server
import threading

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="Live webcam index or RTSP URL")
    parser.add_argument("--experiment", default="configs/experiment_demo.yaml")
    parser.add_argument("--mute", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--save-annotated", action="store_true")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--dashboard", action="store_true", help="Launch live web dashboard server")
    args = parser.parse_args()

    # Launch dashboard server if requested
    if args.dashboard:
        t = threading.Thread(target=start_dashboard_server, kwargs={"port": 8080}, daemon=True)
        t.start()

    # Init Run Dir
    run_dir = Path("runs") / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    
    spec = ExperimentSpec.load_from_yaml(args.experiment)
    global_state.configure_steps([
        {
            "id": step.id,
            "name": step.name,
            "instruction": step.instruction,
            "object": step.required_predicates[0].args[0] if step.required_predicates else "System",
            "action": step.name.upper().replace(" ", "_"),
        }
        for step in spec.steps
    ])
    bus = EventBus()
    
    log_writer = StructuredLogWriter(str(run_dir / "steps.jsonl"))
    
    if args.source.isdigit():
        source = WebcamSource(int(args.source))
        global_state.configure_source("live")
    elif args.source.startswith("rtsp"):
        source = RTSPSource(args.source)
        global_state.configure_source("live")
    else:
        parser.error("Only live input is supported. Use a webcam index such as 0 or an RTSP URL.")

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

    def dashboard_log(event_name, details, status="success"):
        global_state.update_live_data(log_entry={
            "time": time.strftime("%H:%M:%S"),
            "event": event_name,
            "status": status,
            "details": details,
        })

    def interaction_handler(event):
        nonlocal last_dashboard_interaction
        payload = event.payload
        action = f"{payload['verb'].upper()}_{payload['object'].upper()}"
        interaction_key = (payload["verb"], payload["object"])
        if interaction_key == last_dashboard_interaction and payload["verb"] == "holds":
            return
        last_dashboard_interaction = interaction_key
        verb_labels = {
            "touches": "Touched",
            "holds": "Held",
            "releases": "Released",
        }
        verb_label = verb_labels.get(payload["verb"], payload["verb"].title())
        global_state.update_live_data(
            action_probs={action: 1.0},
            log_entry={
                "time": time.strftime("%H:%M:%S"),
                "event": "Action recognized",
                "status": "success",
                "details": f"{verb_label} {payload['object'].replace('_', ' ')}",
            },
        )

    def step_result_handler(event):
        payload = event.payload
        status = payload["status"]
        log_status = "success" if status == "completed" else "warning"
        global_state.update_step_status(payload["step_id"], status)
        dashboard_log(
            "Step result",
            f"{payload['step_name']}: {payload['outcome']}",
            log_status,
        )

    def speech_handler(event):
        text = event.payload.get("text", "")
        global_state.update_live_data(voice_text=text)
        dashboard_log("Voice instruction", text, "info")

    def alert_dashboard_handler(event):
        dashboard_log("Protocol alert", event.payload.get("message", ""), "warning")

    bus.subscribe("interaction", interaction_handler)
    bus.subscribe("step_result", step_result_handler)
    bus.subscribe("speech", speech_handler)
    bus.subscribe("alert", alert_dashboard_handler)
    
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

    first_frame_logged = False
    last_dashboard_interaction = None
    last_dashboard_step_idx = tracker.current_step_idx
    last_dashboard_completed = set(tracker.completed_steps)
    try:
        while True:
            frame = source.get_frame(timeout=0.1)
            if not frame:
                if not source.running:
                    print("[AstroHAR] Warning: Camera source stopped or failed to grab frames.")
                    break
                continue

            if not first_frame_logged:
                print("[AstroHAR] Live camera feed active & streaming to http://localhost:8080/")
                first_frame_logged = True
                
            fps, _ = source.get_stats()
            
            # Perception
            objects = color_det.process(frame.image)
            hands = hand_lm.process(frame.image)
            
            # Interactions
            extractor.process(hands, objects, frame.video_timestamp)

            if tracker.current_step_idx != last_dashboard_step_idx:
                dashboard_log(
                    "State transition",
                    f"Step {last_dashboard_step_idx + 1} -> Step {tracker.current_step_idx + 1}",
                )
                last_dashboard_step_idx = tracker.current_step_idx

            if tracker.completed_steps != last_dashboard_completed:
                dashboard_log(
                    "Step result",
                    ", ".join(sorted(tracker.completed_steps)),
                )
                last_dashboard_completed = set(tracker.completed_steps)
            
            # Overlay
            annotated_img = draw_overlay(frame.image.copy(), objects, hands, spec, 
                                         tracker.current_step_idx, tracker.completed_steps,
                                         active_alert, fps)
                                         
            # Update Live Dashboard State
            global_state.update_live_data(
                frame_img=annotated_img,
                current_step_idx=tracker.current_step_idx,
                fps=fps,
                detections={det.label: round(det.confidence, 2) for det in objects},
                completed_steps=list(tracker.completed_steps),
            )

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
