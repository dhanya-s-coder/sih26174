Project: Smart India Hackathon 2026, ISRO PS 26174, "AI Human Activity Recognition for On-board BAS Experiments".

Background: On space missions (BAS, lunar), real-time ground support is impossible due to communication delay and restricted bandwidth. An on-board AI assistant must track an astronaut performing a pre-defined experiment and validate the sequence of steps, processing video locally at the edge. Inputs come from fixed-payload cameras.

Required features:
1. Continuously process local video feeds to track the experiment sequence.
2. At the start and after each step, suggest the next step.
3. Alert (VOICE-based) when a step is skipped or an out-of-sequence step occurs.
4. From live video, generate a timestamped, structured, lightweight text file of conducted steps with outcome/status.
5. Stream the experiment video to a configurable IP AND store it locally.
6. GUI for monitoring all of the above.
7. Deliverable: trained AI model running on an OFFLINE standalone system (no internet or cloud at runtime).
8. Dataset: synthetic/custom, recorded by us (even with a webcam) to replicate the experiment. Needs object detection, pose estimation and hand-object interaction, based on the steps.
9. Optional: astronauts have no fixed up/down in microgravity, so the approach should be orientation-agnostic (track body/hands relative to the payload rack, not the floor). Ideal is orientation-agnostic 3D Human Mesh Recovery.

Sample experiment (source of truth for step definitions):
1. Open the main box.
2. Take out the yellow box and place it on the left side.
3. Take out the red box and place it on the right side.

Engineering principles:
- Python 3.10+, fully offline at runtime, CPU-first (must run on a laptop without a GPU; GPU optional).
- Everything experiment-specific lives in a YAML spec, NOT in code, so a different experiment can be swapped in without code changes.
- Modular pipeline: input -> perception -> event extraction -> step engine -> outputs (voice, log, GUI, stream). Modules talk through typed data models and a simple in-process event bus.
- Every module must be testable with a video file as the "camera" (replay mode), so we can develop without hardware.
- Lightweight, readable code, type hints, docstrings on public functions, no dead code, no over-engineering.
