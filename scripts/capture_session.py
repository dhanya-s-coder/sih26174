import cv2
import time
import argparse
import json
import sys
import uuid
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from src.har_space.config.runtime import RuntimeConfig
from src.har_space.io.source import WebcamSource, RTSPSource
from src.har_space.io.recorder import LocalRecorder

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/runtime.yaml")
    parser.add_argument("--scenario", required=True, choices=["correct", "skipped_step", "out_of_order"])
    parser.add_argument("--lighting", default="normal", help="e.g., normal, dim, harsh_shadows")
    parser.add_argument("--angle", default="front", help="Camera angle")
    parser.add_argument("--rotation", type=int, default=0, help="Rig rotation in degrees")
    parser.add_argument("--person", default="subject_1", help="Person identifier")
    parser.add_argument("--notes", default="", help="Extra session notes")
    args = parser.parse_args()

    session_id = f"session_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"
    out_dir = Path("data/raw") / session_id
    out_dir.mkdir(parents=True, exist_ok=True)

    config = RuntimeConfig.load_from_yaml(args.config)
    
    if config.source.type == "webcam":
        source = WebcamSource(int(config.source.path_or_index))
    elif config.source.type == "rtsp":
        source = RTSPSource(config.source.path_or_index)
    else:
        print("Capture session requires a live source (webcam or rtsp).")
        return

    recorder = LocalRecorder(str(out_dir), segment_length_minutes=60)
    
    print(f"Starting capture session: {session_id}")
    print("Hotkeys: [N] Next Step Started | [K] Mark Step Completed | [X] Intentional Error | [Q] Quit")
    
    source.start()
    recorder.start(config.target_fps, config.target_resolution)

    timeline = []
    current_step_start = None

    try:
        while True:
            frame = source.get_frame(timeout=0.1)
            if frame:
                recorder.push(frame)
                cv2.imshow("Capture Session", frame.image)
                key = cv2.waitKey(1) & 0xFF
                
                vt = frame.video_timestamp
                
                if key == ord('q'):
                    break
                elif key == ord('n'):
                    if current_step_start is not None:
                        print("Ending previous step without completion.")
                        timeline.append({"event": "incomplete", "start": current_step_start, "end": vt})
                    current_step_start = vt
                    print(f"[{vt:.2f}] Next step started.")
                elif key == ord('k'):
                    if current_step_start is not None:
                        timeline.append({"event": "completed", "start": current_step_start, "end": vt})
                        print(f"[{vt:.2f}] Step marked completed.")
                        current_step_start = None
                    else:
                        print("No step is currently active!")
                elif key == ord('x'):
                    timeline.append({"event": "intentional_error", "timestamp": vt})
                    print(f"[{vt:.2f}] Intentional error marked.")

    except KeyboardInterrupt:
        pass
    finally:
        source.stop()
        recorder.stop()
        cv2.destroyAllWindows()

        metadata = {
            "session_id": session_id,
            "scenario": args.scenario,
            "lighting": args.lighting,
            "angle": args.angle,
            "rig_rotation_deg": args.rotation,
            "person": args.person,
            "notes": args.notes,
            "timeline": timeline
        }
        with open(out_dir / "session.json", "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Session saved to {out_dir}")

if __name__ == "__main__":
    main()
