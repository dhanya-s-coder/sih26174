# Data Recording Checklist

## Objective
Record raw video sessions covering diverse environmental factors to ensure the object detector is robust in space-like microgravity conditions.

## Session Breakdown
Aim for **at least 20 short sessions** total for the first training run.
- **Scenarios**: 
  - 10x `correct` execution
  - 5x `skipped_step`
  - 5x `out_of_order`
- **Lighting**: Ensure varying conditions across the 20 sessions (normal, dim, harsh direct light casting shadows, backlit).
- **Backgrounds**: Change the table or background (e.g., clutter, no clutter, different colors).
- **Camera Angles**: Frontal, overhead, 45-degree angle.
- **Rig Rotation (Microgravity Simulation)**: Since astronauts have no fixed "up", rotate the entire payload rig OR physically rotate the camera 90, 180, and 270 degrees in different sessions so the model learns orientation-invariant features.

## Target Volumes
- For the first model iteration, aim for **~200 labeled frames per class** (main_box_closed, main_box_open, yellow_box, red_box). 
- Use the `scripts/dataset_report.py` to track this volume.
