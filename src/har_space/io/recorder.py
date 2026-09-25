import cv2
import time
import threading
import queue
import csv
import os
import shutil
import logging
from typing import Optional
from .models import Frame

logger = logging.getLogger(__name__)

class LocalRecorder:
    def __init__(self, output_dir: str, segment_length_minutes: float = 5.0, min_disk_space_mb: int = 500):
        self.output_dir = output_dir
        self.segment_length = segment_length_minutes * 60
        self.min_disk_space_mb = min_disk_space_mb
        self.queue = queue.Queue()
        self.running = False
        self.thread: Optional[threading.Thread] = None
        
        self.writer = None
        self.csv_file = None
        self.csv_writer = None
        self.current_segment_start = 0.0
        self.segment_index = 0
        self.fps = 30.0
        self.resolution = (640, 480)
        self.current_segment_path = ""
        self.error_message = ""
        self._recording_started = False
        
        os.makedirs(self.output_dir, exist_ok=True)

    @property
    def status(self) -> str:
        if self.error_message:
            return "ERROR"
        if not self.running:
            return "INACTIVE"
        return "ACTIVE" if self._recording_started else "STARTING"

    def start(self, fps: float, resolution: tuple[int, int]):
        self.error_message = ""
        self._recording_started = False
        self.fps = fps
        self.resolution = resolution
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)

    def push(self, frame: Frame):
        if not self.running:
            return
        try:
            self.queue.put_nowait(frame)
        except queue.Full:
            pass # Unbounded in practice unless we set maxsize, but for safety it's good

    def _check_disk_space(self) -> bool:
        total, used, free = shutil.disk_usage(self.output_dir)
        free_mb = free / (1024 * 1024)
        if free_mb < self.min_disk_space_mb:
            self.error_message = f"Low disk space: {free_mb:.2f} MB available"
            logger.warning("%s. Stopping recording.", self.error_message)
            return False
        return True

    def _rotate_segment(self, timestamp: float):
        self._close_current()
        if not self._check_disk_space():
            self.running = False
            return
            
        time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(timestamp))
        base_name = os.path.join(self.output_dir, f"segment_{time_str}")
        self.current_segment_path = f"{base_name}.mp4"
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(self.current_segment_path, fourcc, self.fps, self.resolution)
        if not self.writer.isOpened():
            self.error_message = f"Could not open video output: {self.current_segment_path}"
            logger.error(self.error_message)
            self._close_current()
            self.running = False
            return
        
        self.csv_file = open(f"{base_name}.csv", 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["frame_id", "wall_clock_timestamp", "video_timestamp"])
        self._recording_started = True
        
        self.current_segment_start = timestamp

    def _close_current(self):
        if self.writer:
            self.writer.release()
            self.writer = None
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None

    def _run(self):
        try:
            while self.running or not self.queue.empty():
                try:
                    frame = self.queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                if self.writer is None or (frame.wall_clock_timestamp - self.current_segment_start) >= self.segment_length:
                    self._rotate_segment(frame.wall_clock_timestamp)
                    if not self.running:
                        break

                if self.writer and self.running:
                    img_resized = cv2.resize(frame.image, self.resolution)
                    self.writer.write(img_resized)
                    if self.csv_writer:
                        self.csv_writer.writerow([frame.frame_id, frame.wall_clock_timestamp, frame.video_timestamp])
        except Exception as exc:
            self.error_message = str(exc)
            logger.exception("Recording failed")
            self.running = False
        finally:
            self._close_current()
