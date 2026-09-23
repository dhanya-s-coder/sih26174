import cv2
import numpy as np
import time
from typing import List, Dict
from src.har_space.data_models import HandState, Detection, Alert
from src.har_space.config.models import ExperimentSpec

def draw_overlay(frame: np.ndarray, 
                 detections: List[Detection], 
                 hands: List[HandState], 
                 spec: ExperimentSpec, 
                 current_step_idx: int,
                 completed_steps: set,
                 active_alert: Alert = None,
                 fps: float = 0.0) -> np.ndarray:
                 
    h, w = frame.shape[:2]
    
    # Draw detections
    for det in detections:
        x1 = int(det.bbox[0] * w)
        y1 = int(det.bbox[1] * h)
        x2 = int(det.bbox[2] * w)
        y2 = int(det.bbox[3] * h)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(frame, det.label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Draw hands
    for hand in hands:
        # Bbox
        hx1 = int(hand.bbox[0] * w)
        hy1 = int(hand.bbox[1] * h)
        hx2 = int(hand.bbox[2] * w)
        hy2 = int(hand.bbox[3] * h)
        cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (255, 0, 255), 2)
        cv2.putText(frame, f"{hand.hand_type} ({hand.confidence:.2f})", (hx1, hy1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        
        # Skeleton
        for lm in hand.landmarks:
            cx, cy = int(lm[0] * w), int(lm[1] * h)
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

    # Top Banner
    cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 0), -1)
    curr_step = spec.steps[current_step_idx].name if current_step_idx < len(spec.steps) else "DONE"
    next_step = spec.steps[current_step_idx+1].name if current_step_idx+1 < len(spec.steps) else "None"
    banner_text = f"Current step: {curr_step} | Next: {next_step}"
    cv2.putText(frame, banner_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 120, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Checklist Panel
    panel_y = 60
    for idx, step in enumerate(spec.steps):
        if step.id in completed_steps:
            color = (0, 255, 0)
            status = "[X]"
        elif idx == current_step_idx:
            color = (0, 255, 255)
            status = "[>]"
        else:
            color = (150, 150, 150)
            status = "[ ]"
            
        cv2.putText(frame, f"{status} {step.name}", (10, panel_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        panel_y += 20

    # Alert Banner
    if active_alert and (time.time() - active_alert.timestamp) < 3.0:
        cv2.rectangle(frame, (0, h-50), (w, h), (0, 0, 255), -1)
        cv2.putText(frame, f"ALERT: {active_alert.message}", (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return frame
