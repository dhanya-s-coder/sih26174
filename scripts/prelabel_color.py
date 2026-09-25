import cv2
import numpy as np
import argparse
from pathlib import Path
import yaml

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True)
    parser.add_argument("--classes", default="configs/classes.yaml")
    # Hardcoded HSV ranges for demo, usually loaded from a config
    # Example: yellow, red
    args = parser.parse_args()

    frames_dir = Path("data/frames") / args.session
    labels_dir = Path("data/labels") / args.session
    labels_dir.mkdir(parents=True, exist_ok=True)

    with open(args.classes, "r") as f:
        cls_config = yaml.safe_load(f)["classes"]
        
    class_map = {c["name"]: c["id"] for c in cls_config}
    
    # Red is tricky in HSV (wraps around), yellow is straightforward
    hsv_ranges = {
        "yellow_box": [(20, 100, 100), (30, 255, 255)],
        "red_box": [(0, 100, 100), (10, 255, 255)] # Simple range for bootstrapping
    }

    count = 0
    for img_path in frames_dir.glob("*.jpg"):
        img = cv2.imread(str(img_path))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, w = img.shape[:2]
        
        labels = []
        for cls_name, (lower, upper) in hsv_ranges.items():
            if cls_name not in class_map: continue
            
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                if cv2.contourArea(cnt) > 500: # Filter noise
                    x, y, bw, bh = cv2.boundingRect(cnt)
                    # Convert to YOLO format (center_x, center_y, width, height) normalized
                    cx = (x + bw/2.0) / w
                    cy = (y + bh/2.0) / h
                    nw = bw / w
                    nh = bh / h
                    labels.append(f"{class_map[cls_name]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                    
        txt_path = labels_dir / (img_path.stem + ".txt")
        if not txt_path.exists() or len(labels) > 0:
            with open(txt_path, "w") as f:
                f.write("\n".join(labels) + "\n")
        count += 1
        
    print(f"Pre-labeled {count} images in {labels_dir}")

if __name__ == "__main__":
    main()
