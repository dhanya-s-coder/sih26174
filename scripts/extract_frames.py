import cv2
import numpy as np
import argparse
import json
import csv
from pathlib import Path
import os

def dhash(image, hash_size=8):
    """Computes perceptual difference hash (dHash)."""
    resized = cv2.resize(image, (hash_size + 1, hash_size))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    diff = gray[:, 1:] > gray[:, :-1]
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])

def variance_of_laplacian(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True, help="Session ID in data/raw/")
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between frames")
    parser.add_argument("--blur_thresh", type=float, default=100.0, help="Laplacian variance threshold")
    parser.add_argument("--hash_thresh", type=int, default=5, help="Hamming distance for near-duplicates")
    args = parser.parse_args()

    raw_dir = Path("data/raw") / args.session
    if not raw_dir.exists():
        print(f"Session {args.session} not found.")
        return

    frames_dir = Path("data/frames") / args.session
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Load session metadata
    session_json = raw_dir / "session.json"
    boundaries = []
    if session_json.exists():
        with open(session_json, "r") as f:
            meta = json.load(f)
            for evt in meta.get("timeline", []):
                if "start" in evt: boundaries.append(evt["start"])
                if "end" in evt: boundaries.append(evt["end"])
                if "timestamp" in evt: boundaries.append(evt["timestamp"])

    csv_files = sorted(list(raw_dir.glob("*.csv")))
    mp4_files = sorted(list(raw_dir.glob("*.mp4")))

    seen_hashes = []
    extracted_count = 0

    for csv_file, mp4_file in zip(csv_files, mp4_files):
        timestamps = {}
        with open(csv_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                timestamps[int(row["frame_id"])] = float(row["video_timestamp"])

        cap = cv2.VideoCapture(str(mp4_file))
        last_saved_time = -999.0
        frame_id = 0
        
        while True:
            ret, frame = cap.read()
            if not ret: break

            vt = timestamps.get(frame_id, frame_id / 30.0) # Fallback to 30fps if missing
            
            # Decide if we should sample (interval OR near a boundary)
            near_boundary = any(abs(vt - b) < 1.0 for b in boundaries)
            time_to_sample = (vt - last_saved_time) >= args.interval
            
            if time_to_sample or near_boundary:
                blur = variance_of_laplacian(frame)
                if blur >= args.blur_thresh:
                    h = dhash(frame)
                    # Check for duplicates
                    is_dup = False
                    for sh in seen_hashes[-50:]: # Compare against recent hashes
                        # hamming distance between integers
                        dist = bin(h ^ sh).count('1')
                        if dist < args.hash_thresh:
                            is_dup = True
                            break
                            
                    if not is_dup:
                        out_path = frames_dir / f"frame_{vt:.2f}.jpg"
                        cv2.imwrite(str(out_path), frame)
                        seen_hashes.append(h)
                        last_saved_time = vt
                        extracted_count += 1

            frame_id += 1
            
        cap.release()

    print(f"Extracted {extracted_count} frames to {frames_dir}")

if __name__ == "__main__":
    main()
