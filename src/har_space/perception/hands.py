import sys
import os
import cv2
import numpy as np
from typing import List
from src.har_space.data_models import HandState

# Attempt to import mediapipe. If missing, it will be caught in run_demo or tests.
try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
except ImportError:
    mp = None

class HandLandmarker:
    def __init__(self, model_path: str = "models/mediapipe/hand_landmarker.task"):
        if mp is None:
            raise ImportError("mediapipe is not installed.")
            
        if not os.path.exists(model_path):
            print("ERROR: MediaPipe Hand Landmarker model not found!")
            print(f"Expected at: {model_path}")
            print("Download it from: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
            sys.exit(1)
            
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5)
        self.detector = vision.HandLandmarker.create_from_options(options)
        
    def process(self, frame: np.ndarray, timestamp_ms: int = 0) -> List[HandState]:
        # For video stream we should technically use VIDEO mode and pass timestamp,
        # butIMAGE mode works fine per-frame for our demo and avoids timestamp tracking overhead.
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = self.detector.detect(mp_image)
        
        hands = []
        if result.hand_landmarks:
            for i, (landmarks, handedness) in enumerate(zip(result.hand_landmarks, result.handedness)):
                pts = [[lm.x, lm.y, lm.z] for lm in landmarks]
                
                # compute bbox
                xs = [lm.x for lm in landmarks]
                ys = [lm.y for lm in landmarks]
                bbox = [min(xs), min(ys), max(xs), max(ys)]
                
                hand = HandState(
                    hand_type=handedness[0].category_name.lower(),
                    bbox=bbox,
                    landmarks=pts,
                    confidence=handedness[0].score,
                    state="unknown"
                )
                hands.append(hand)
        return hands
