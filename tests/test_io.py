import os
import cv2
import time
import requests
import threading
from src.har_space.io.source import FileSource
from src.har_space.io.recorder import LocalRecorder
from src.har_space.io.streamer import VideoStreamer
from src.har_space.io.models import Frame
import numpy as np

def test_file_source(tmp_path):
    # Generate a tiny dummy video
    vid_path = str(tmp_path / "test.mp4")
    out = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (100, 100))
    for i in range(10):
        out.write(np.zeros((100, 100, 3), dtype=np.uint8))
    out.release()
    
    source = FileSource(vid_path, realtime_pacing=False, loop=False)
    source.start()
    
    frames_received = 0
    while True:
        frame = source.get_frame(timeout=0.5)
        if frame is None and not source.running:
            break
        if frame:
            frames_received += 1
            assert isinstance(frame.video_timestamp, float)
    
    source.stop()
    assert frames_received == 10

def test_recorder(tmp_path):
    output_dir = tmp_path / "records"
    recorder = LocalRecorder(str(output_dir), segment_length_minutes=0.1)
    recorder.start(fps=30, resolution=(100, 100))
    
    for i in range(5):
        frame = Frame(
            frame_id=i,
            wall_clock_timestamp=time.time() + i,
            video_timestamp=float(i) / 30.0,
            image=np.zeros((100, 100, 3), dtype=np.uint8),
            source_name="test"
        )
        recorder.push(frame)
    
    recorder.stop()
    
    # Check if files exist
    mp4s = list(output_dir.glob("*.mp4"))
    csvs = list(output_dir.glob("*.csv"))
    assert len(mp4s) > 0
    assert len(csvs) > 0

def test_mjpeg_streamer():
    streamer = VideoStreamer("mjpeg", "127.0.0.1", 8081)
    streamer.start()
    
    # Push a test frame
    frame = Frame(
        frame_id=0,
        wall_clock_timestamp=time.time(),
        video_timestamp=0.0,
        image=np.zeros((100, 100, 3), dtype=np.uint8),
        source_name="test"
    )
    streamer.push(frame)
    time.sleep(0.5)
    
    # Fetch from server
    try:
        resp = requests.get("http://127.0.0.1:8081", stream=True, timeout=2)
        assert resp.status_code == 200
        assert b'multipart/x-mixed-replace' in resp.headers['Content-type'].encode()
    finally:
        streamer.stop()
