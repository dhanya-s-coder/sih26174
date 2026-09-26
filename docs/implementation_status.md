# Implementation Status and Operating Rules

## Experiment FSM

The active experiment is configured in `configs/experiment_demo.yaml` and contains five ordered states:

1. Main box present
2. Yellow box removed from the main box
3. Yellow box placed on the left
4. Red box removed from the main box
5. Red box placed on the right

The FSM accepts only the current state's predicate. A future action is reported as out of sequence, and an incorrect left/right placement is reported as a wrong placement.

## Pickup and placement rules

An object removal is confirmed only when the object is visible outside the main-box boundary for several frames and hand contact has been observed. Placement is confirmed only after the object remains stable for several frames. Left and right are calculated relative to the detected main-box edges, not relative to the camera frame.

## Confidence and timeout work

The current perception layer is heuristic and should be treated as a prototype. Before final model integration, confidence values should be propagated from detections and interactions into `StepRecord.confidence`. Step timeouts are configured per experiment step using the optional `timeout` field in the experiment YAML and should be enforced by the step engine during live processing.

## Runtime configuration

Camera source, resolution, recording, and streaming settings belong in `configs/runtime.yaml`. Experiment-specific step definitions belong in `configs/experiment_demo.yaml` or another YAML file passed with `--experiment`; they should not be hardcoded in the perception code.

## Offline operation

Runtime model assets must be stored locally under `models/`. Internet access must not be required while running the dashboard, webcam, or recorded-video replay.

Start the packaged local system from the repository root with:

```powershell
python start_system.py
```

On Windows, `start_system.bat` can be double-clicked. The launcher checks the local hand-landmarker model, starts the dashboard pipeline, and opens `http://localhost:8080`. Source switching is then controlled from the dashboard.

## Verification checklist

- Run the five-step sequence from webcam and recorded video.
- Try red before yellow and confirm an out-of-sequence voice/dashboard alert.
- Place yellow right and red left and confirm a wrong-placement alert.
- Confirm JSONL step records and the end-of-run summary.
- Confirm dashboard source switching and local video display.
