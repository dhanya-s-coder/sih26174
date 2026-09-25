import cv2
import numpy as np
import time
import textwrap
from typing import List, Optional
from src.har_space.data_models import HandState, Detection, Alert
from src.har_space.config.models import ExperimentSpec

def draw_overlay(frame: np.ndarray, 
                 detections: List[Detection], 
                 hands: List[HandState], 
                 spec: ExperimentSpec, 
                 current_step_idx: int,
                 completed_steps: set,
                 active_alert: Alert = None,
                 fps: float = 0.0,
                 show_telemetry: bool = True,
                 interaction_text: str = "",
                 current_instruction: str = "",
                 status_messages: Optional[List[str]] = None,
                 step_statuses: Optional[dict[str, str]] = None) -> np.ndarray:
                 
    h, w = frame.shape[:2]
    
    # Draw the actual color-detector boxes without implying a model score.
    for det in detections:
        x1 = int(det.bbox[0] * w)
        y1 = int(det.bbox[1] * h)
        x2 = int(det.bbox[2] * w)
        y2 = int(det.bbox[3] * h)
        color = (0, 0, 255) if det.label == "red_box" else (0, 220, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, det.label.replace("_", " "), (x1, max(18, y1-5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # Draw hands
    for hand in hands:
        # Bbox
        hx1 = int(hand.bbox[0] * w)
        hy1 = int(hand.bbox[1] * h)
        hx2 = int(hand.bbox[2] * w)
        hy2 = int(hand.bbox[3] * h)
        cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (255, 0, 255), 2)
        cv2.putText(frame, hand.hand_type, (hx1, max(18, hy1-5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA)
        
        # Skeleton
        for lm in hand.landmarks:
            cx, cy = int(lm[0] * w), int(lm[1] * h)
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

    if show_telemetry:
        del fps  # Retained in the signature for existing callers; never presented as UI telemetry.
        panel_width = min(max(300, w - 20), 440)
        current_instruction = current_instruction or (
            spec.steps[current_step_idx].instruction
            if current_step_idx < len(spec.steps) else "Experiment complete."
        )
        instruction_lines = textwrap.wrap(
            current_instruction, width=max(24, (panel_width - 28) // 9)
        ) or [""]
        panel_height = min(
            h - 12,
            44 + 24 * len(spec.steps) + 25 + 20 * len(instruction_lines),
        )
        panel = frame.copy()
        cv2.rectangle(panel, (8, 8), (8 + panel_width, 8 + panel_height), (18, 25, 32), -1)
        cv2.addWeighted(panel, 0.82, frame, 0.18, 0, frame)
        cv2.putText(frame, "EXPERIMENT PROGRESS", (20, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA)

        panel_y = 57
        for idx, step in enumerate(spec.steps):
            step_status = (step_statuses or {}).get(
                step.id, "completed" if step.id in completed_steps else ""
            )
            if step_status == "completed":
                color = (0, 255, 0)
                marker = "done"
            elif idx == current_step_idx:
                color = (0, 255, 255)
                marker = "current"
            else:
                color = (150, 150, 150)
                marker = "upcoming"
            marker_x, marker_y = 22, panel_y - 5
            if marker == "done":
                cv2.line(frame, (marker_x, marker_y), (marker_x + 5, marker_y + 5), color, 2)
                cv2.line(frame, (marker_x + 5, marker_y + 5), (marker_x + 13, marker_y - 5), color, 2)
            elif marker == "current":
                cv2.arrowedLine(frame, (marker_x, marker_y), (marker_x + 13, marker_y), color, 2, tipLength=0.45)
            else:
                cv2.circle(frame, (marker_x + 6, marker_y), 6, color, 1)
            cv2.putText(frame, step.name, (44, panel_y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 1, cv2.LINE_AA)
            panel_y += 24

        cv2.putText(frame, "CURRENT INSTRUCTION", (20, panel_y + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (190, 205, 215), 1, cv2.LINE_AA)
        panel_y += 23
        for line in instruction_lines:
            if panel_y > h - 16:
                break
            cv2.putText(frame, line, (20, panel_y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.48, (255, 255, 255), 1, cv2.LINE_AA)
            panel_y += 20

        if active_alert and active_alert.level.lower() in {"warning", "critical", "error"} \
                and (time.time() - active_alert.timestamp) < 5.0:
            cv2.rectangle(frame, (0, h-48), (w, h), (0, 0, 190), -1)
            cv2.putText(frame, f"ALERT: {active_alert.message}"[:100], (12, h-17),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        messages = status_messages or []
        if messages:
            lines = [line for message in messages for line in textwrap.wrap(
                message, width=max(24, (w - 28) // 9)
            )]
            lines = lines[-3:]
            bar_height = 16 + 20 * len(lines)
            top = max(0, h - bar_height - (48 if active_alert else 0))
            cv2.rectangle(frame, (0, top), (w, top + bar_height), (15, 15, 15), -1)
            for index, line in enumerate(lines):
                cv2.putText(frame, line, (12, top + 22 + index * 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (120, 210, 255), 1, cv2.LINE_AA)
    elif interaction_text:
        text = interaction_text.upper()[:60]
        cv2.rectangle(frame, (0, 0), (min(w, 420), 34), (13, 23, 30), -1)
        cv2.putText(frame, text, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 190, 45), 1, cv2.LINE_AA)

    return frame
