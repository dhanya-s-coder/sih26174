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
    
    def make_source(source_name):
        if source_name == "webcam":
            global_state.configure_source("live")
            return WebcamSource(0)
        if source_name == "recording":
            global_state.configure_source("replay")
            return FileSource("data/raw/experiment.mp4", realtime_pacing=True, loop=False)
        if source_name == "recording2":
            global_state.configure_source("replay")
            return FileSource("data/raw/exp_final.mp4", realtime_pacing=True, loop=False)
        if source_name.isdigit():
            global_state.configure_source("live")
            return WebcamSource(int(source_name))
        if source_name.startswith("rtsp"):
            global_state.configure_source("live")
            return RTSPSource(source_name)
        global_state.configure_source("replay")
        return FileSource(source_name, realtime_pacing=True, loop=False)

    source = make_source(args.source)

    streamer = None
    if args.stream:
        # Keep the dashboard and raw MJPEG stream on separate ports.
        stream_port = 8081 if args.dashboard else 8080
        streamer = VideoStreamer("mjpeg", "0.0.0.0", stream_port)
        
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
        if payload["verb"] == "holds":
            if payload["object"] in dashboard_hold_objects:
                return
            dashboard_hold_objects.add(payload["object"])
        elif payload["verb"] == "releases":
            dashboard_hold_objects.discard(payload["object"])
        if interaction_key == last_dashboard_interaction and payload["verb"] != "releases":
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

    first_frame_logged = False
    last_dashboard_interaction = None
    dashboard_hold_objects = set()
    last_dashboard_step_idx = tracker.current_step_idx
    last_dashboard_completed = set(tracker.completed_steps)
    try:
        while True:
            command = global_state.consume_command() if args.dashboard else None
            if command:
                if command["action"] == "quit":
                    break
                if command["action"] == "source":
                    source.stop()
                    source = make_source(command["source"])
                    source.start()
                    tracker.current_step_idx = 0
                    tracker.completed_steps.clear()
                    tracker.active_predicates.clear()
                    extractor.contact_state.clear()
                    for state in object_tracks.values():
                        state.update({"inside": False, "removed": False, "removal_frames": 0, "hand_seen": False, "last_center": None, "stable_frames": 0, "placed": False, "hand_contact": False})

            frame = source.get_frame(timeout=0.1)
            if not frame:
                if not source.running:
                    global_state.configure_source("idle")
                    time.sleep(0.05)
                continue

            if not first_frame_logged:
                print("[AstroHAR] Live camera feed active & streaming to http://localhost:8080/")
                first_frame_logged = True
                
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
                        # Publish whichever side was actually detected. The
                        # FSM decides whether it is correct or out of order;
                        # filtering wrong placements here would hide them.
                        if location in {"left", "right"}:
                            expected = "left" if obj.label == "yellow_box" else "right"
                            track["placed"] = location == expected
                            bus.publish(Event(
                                event_type="interaction", timestamp=frame.video_timestamp,
                                payload={"subject": "object", "verb": "placed", "object": obj.label, "location": location}
                            ))

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
                
            # In dashboard mode the browser is the display. Do not open a
            # second native OpenCV window unless dashboard mode is disabled.
            if not args.headless and not args.dashboard:
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
