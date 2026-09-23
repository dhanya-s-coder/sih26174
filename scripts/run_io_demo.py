import time
import argparse
import logging
import cv2
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.har_space.config.runtime import RuntimeConfig
from src.har_space.io.source import WebcamSource, FileSource, RTSPSource
from src.har_space.io.recorder import LocalRecorder
from src.har_space.io.streamer import VideoStreamer

logging.basicConfig(level=logging.INFO)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/runtime.yaml")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    config = RuntimeConfig.load_from_yaml(args.config)
    
    # Initialize Source
    if config.source.type == "webcam":
        idx = int(config.source.path_or_index)
        source = WebcamSource(idx)
    elif config.source.type == "file":
        source = FileSource(config.source.path_or_index, realtime_pacing=True, loop=True)
    elif config.source.type == "rtsp":
        source = RTSPSource(config.source.path_or_index)
    else:
        raise ValueError(f"Unknown source type: {config.source.type}")

    # Initialize Recorder
    recorder = None
    if config.recording.enabled:
        recorder = LocalRecorder(
            output_dir=config.recording.directory,
            segment_length_minutes=config.recording.segment_length_minutes
        )

    # Initialize Streamer
    streamer = None
    if config.streaming.enabled:
        streamer = VideoStreamer(
            mode=config.streaming.mode,
            host=config.streaming.host,
            port=config.streaming.port
        )

    # Start components
    source.start()
    if recorder:
        recorder.start(config.target_fps, config.target_resolution)
    if streamer:
        streamer.start()

    logging.info("Press Ctrl+C to stop.")
    try:
        while True:
            frame = source.get_frame(timeout=0.1)
            if frame:
                if recorder:
                    recorder.push(frame)
                if streamer:
                    streamer.push(frame)
                
                if not args.headless:
                    cv2.imshow("Preview", frame.image)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            
            fps, dropped = source.get_stats()
            # print(f"\rFPS: {fps:.1f}, Dropped: {dropped}", end="")
            
    except KeyboardInterrupt:
        logging.info("Shutting down...")
    finally:
        source.stop()
        if recorder:
            recorder.stop()
        if streamer:
            streamer.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
