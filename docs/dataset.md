# Dataset Generation Strategy

## Object Detection vs Pose Estimation
1. **Pose and Hands**: Hands come from MediaPipe (or similar pretrained Pose/Hand models). This means we do NOT need to collect or manually label training data for pose and hand landmarks in our bounding box dataset.
2. **Experiment Objects**: We will train a lightweight object detector (e.g., YOLOv8n or similar) to detect the specific payload items: `main_box_closed`, `main_box_open`, `yellow_box`, and `red_box`. These are the visually distinct states required by the step logic.

By isolating the bespoke experiment objects from the human tracking, we drastically reduce the labeling burden and increase generalization.
