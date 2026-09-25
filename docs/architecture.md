# Architecture Decisions

## Pipeline Design
The pipeline follows a modular data-flow architecture:
**Input -> Perception -> Event Extraction -> Step Engine -> Outputs (Voice, Log, GUI, Stream)**

## Event Bus Usage
**Decision:** Raw video frames are NOT pushed through the in-process event bus.

**Rationale:** The event bus is designed for lightweight messaging (e.g. state transitions, alerts, text logs). Pushing 640x480 RGB arrays at 30fps through a pub/sub mechanism creates high overhead, excessive memory copies, and unnecessary queue management. 
Instead, video frames are passed via dedicated, bounded queues (for recording) and direct references (for MJPEG streaming) directly from the input/perception layers.

### Flow Diagram
```
Camera/File -> FrameSource (Thread 1)
   |--> LocalRecorder (Thread 2)
   |--> Perception Node (Thread 3) -> Extracts Detections/Poses
           |--> EventBus
           |--> Streamer (Thread 4) - Pushes annotated frames
```
