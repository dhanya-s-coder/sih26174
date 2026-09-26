"""Offline AstroHAR launcher for Windows and other local environments."""

from pathlib import Path
import argparse
import subprocess
import sys
import time
import webbrowser


ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "models" / "mediapipe" / "hand_landmarker.task"


def main() -> int:
    parser = argparse.ArgumentParser(description="Start the offline AstroHAR dashboard and pipeline.")
    parser.add_argument("--source", default="0", help="Webcam index, video path, or RTSP URL")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--mute", action="store_true")
    args = parser.parse_args()

    if not MODEL.exists():
        print(f"ERROR: Missing local MediaPipe model: {MODEL}")
        print("Place hand_landmarker.task in models/mediapipe before starting offline mode.")
        return 1

    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_demo.py"),
        "--source",
        args.source,
        "--dashboard",
    ]
    if args.mute:
        command.append("--mute")

    print("Starting AstroHAR in offline mode...")
    print(f"Dashboard: http://localhost:{args.port}")
    process = subprocess.Popen(command, cwd=ROOT)

    try:
        time.sleep(1.5)
        webbrowser.open(f"http://localhost:{args.port}")
        process.wait()
    except KeyboardInterrupt:
        print("Stopping AstroHAR...")
        process.terminate()
        process.wait(timeout=5)
    return process.returncode or 0


if __name__ == "__main__":
    raise SystemExit(main())
