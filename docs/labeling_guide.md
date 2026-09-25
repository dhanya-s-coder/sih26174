# Labeling Guide

We will use **X-AnyLabeling** (a modern, offline-capable, free tool that supports Segment Anything and YOLO) for manual corrections and verifications.

## Workflow
1. Run `scripts/extract_frames.py` to sample frames from a raw session into `data/frames/<session_id>/`.
2. Run `scripts/prelabel_color.py` to bootstrap the bounding boxes for visually distinct objects (yellow box, red box) using color segmentation. This creates initial `.txt` labels in YOLO format in `data/labels/<session_id>/`.
3. Open X-AnyLabeling, and open the `data/frames/<session_id>` directory.
4. Set the labels format to **YOLO** and point it to the `configs/classes.yaml` or a generated `classes.txt`.
5. Review each frame, adjust the bootstrapped boxes, and add boxes for `main_box_closed` and `main_box_open`.
6. Save the labels back into `data/labels/<session_id>/`.
7. Run `scripts/review_labels.py` to generate visual contact sheets of the labels to QA everything quickly without clicking through the GUI again.
