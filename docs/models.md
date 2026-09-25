# Models Documentation

## Hand Landmarker
We use the **MediaPipe Hand Landmarker Task API** for pose estimation and hand tracking.
To run the live demo, you **must download the model file manually** and place it in the correct directory.

**File Path:** `models/mediapipe/hand_landmarker.task`

**Download URL:** [https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task)

The app is designed to work fully offline once the file is present. It will never attempt to download models at runtime.

## Color Tracking
Currently, objects (`red_box` and `yellow_box`) are detected using purely classical computer vision (HSV segmentation) via `ColorBoxDetector`. This operates on the CPU and requires zero trained weights.

In later milestones, this will be replaced by a YOLO-style object detector using the dataset we collect.
