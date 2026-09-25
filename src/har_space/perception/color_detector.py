import cv2
import numpy as np
import yaml
from pathlib import Path
from typing import List, Dict
from src.har_space.data_models import Detection

class ColorBoxDetector:
    def __init__(self, config_path: str = "configs/hsv_ranges.yaml"):
        self.ranges = {}
        if Path(config_path).exists():
            with open(config_path, "r") as f:
                self.ranges = yaml.safe_load(f)
        
        # Simple tracking: persist ID for largest blob of a color
        self.next_id = 0
        self.tracked_objects: Dict[str, int] = {} # {color: id}

    def process(self, frame: np.ndarray) -> List[Detection]:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        detections = []
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        
        for color_name, r in self.ranges.items():
            lower = np.array(r["lower"])
            upper = np.array(r["upper"])
            mask = cv2.inRange(hsv, lower, upper)
            
            # Handle red wrap-around if present
            if "lower2" in r and "upper2" in r:
                mask2 = cv2.inRange(hsv, np.array(r["lower2"]), np.array(r["upper2"]))
                mask = cv2.bitwise_or(mask, mask2)
                
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                largest = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest)
                if area > r.get("min_area", 1000):
                    x, y, w, h = cv2.boundingRect(largest)

                    frame_area = float(frame.shape[0] * frame.shape[1])
                    area_ratio = area / frame_area if frame_area else 0.0
                    if area_ratio < r.get("min_frame_ratio", 0.0) or area_ratio > r.get("max_frame_ratio", 1.0):
                        continue

                    # Reject irregular blobs (for example faces or skin regions)
                    # that happen to fall inside a color range. Experiment
                    # objects are expected to be reasonably rectangular.
                    bbox_area = float(w * h)
                    rectangularity = area / bbox_area if bbox_area else 0.0
                    if rectangularity < 0.55:
                        continue
                    
                    if color_name not in self.tracked_objects:
                        self.tracked_objects[color_name] = self.next_id
                        self.next_id += 1
                        
                    obj_id = self.tracked_objects[color_name]
                    
                    # Convert to normalized
                    fh, fw = frame.shape[:2]
                    det = Detection(
                        label=color_name,
                        confidence=1.0,
                        bbox=[x/fw, y/fh, (x+w)/fw, (y+h)/fh]
                    )
                    # Hack: since Detection doesn't have id, we'll put it in label for demo, e.g., "red_box_0"
                    # But the tracker just needs the class. Let's keep label as class.
                    detections.append(det)
        
        return detections
