import cv2
import numpy as np
import os
import argparse

def generate_video(output_path, duration_sec=10, fps=30, width=640, height=480):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    total_frames = duration_sec * fps
    x, y = 50, 50
    dx, dy = 5, 3
    
    for i in range(total_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Moving colored rectangle
        x += dx
        y += dy
        if x < 0 or x + 100 > width: dx = -dx
        if y < 0 or y + 100 > height: dy = -dy
        
        color = (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x+100, y+100), color, -1)
        
        # Frame counter
        cv2.putText(frame, f"Frame: {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Time: {i/fps:.2f}s", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(frame)
        
    out.release()
    print(f"Generated test video: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="test_video.mp4")
    parser.add_argument("--duration", type=int, default=10)
    args = parser.parse_args()
    generate_video(args.output, args.duration)
