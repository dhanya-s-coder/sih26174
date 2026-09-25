import time
import threading
import queue
import cv2
import logging
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Any
from .models import Frame

logger = logging.getLogger(__name__)

class FrameSource(ABC):
    def __init__(self, name: str, queue_size: int = 5):
        self.name = name
        self.frame_queue = queue.Queue(maxsize=queue_size)
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.frames_read = 0
        self.frames_dropped = 0
        self.start_time = 0.0

    def start(self):
        self.running = True
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)

    def get_frame(self, timeout: float = 0.1) -> Optional[Frame]:
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _push_frame(self, frame: Frame):
        try:
            self.frame_queue.put_nowait(frame)
        except queue.Full:
            try:
                self.frame_queue.get_nowait()
                self.frame_queue.put_nowait(frame)
                self.frames_dropped += 1
            except (queue.Empty, queue.Full):
                pass

    @abstractmethod
    def _run(self):
        pass

    def get_stats(self) -> Tuple[float, int]:
        elapsed = time.time() - self.start_time
        fps = self.frames_read / elapsed if elapsed > 0 else 0.0
        return fps, self.frames_dropped


class OpenCVSource(FrameSource):
    """Base class for sources using cv2.VideoCapture."""
    def __init__(self, name: str, source_id: Any, auto_reconnect: bool = False, realtime_pacing: bool = False, loop: bool = False):
        super().__init__(name)
        self.source_id = source_id
        self.auto_reconnect = auto_reconnect
        self.realtime_pacing = realtime_pacing
        self.loop = loop
        self.cap = None

    def _open(self):
        import os
        if isinstance(self.source_id, int):
            # Try default backend first
            self.cap = cv2.VideoCapture(self.source_id)
            if not self.cap or not self.cap.isOpened():
                if os.name == 'nt':
                    self.cap = cv2.VideoCapture(self.source_id, cv2.CAP_DSHOW)
                if (not self.cap or not self.cap.isOpened()) and os.name == 'nt':
                    self.cap = cv2.VideoCapture(self.source_id, cv2.CAP_MSMF)
            
            # If camera index 0 failed, try scanning index 1 or 2
            if (not self.cap or not self.cap.isOpened()) and self.source_id == 0:
                logger.warning("Camera index 0 failed to open. Trying camera index 1...")
                for idx in [1, 2]:
                    self.cap = cv2.VideoCapture(idx)
                    if self.cap and self.cap.isOpened():
                        logger.info(f"Opened alternative camera index {idx}")
                        break
        else:
            self.cap = cv2.VideoCapture(self.source_id)

        opened = self.cap.isOpened() if self.cap else False
        if opened:
            logger.info(f"Successfully opened source {self.name} ({self.source_id})")
        else:
            logger.error(f"Could not open camera/video source {self.name} ({self.source_id})")
        return opened

    def _run(self):
        while self.running:
            if not self.cap or not self.cap.isOpened():
                if not self._open():
                    if not self.auto_reconnect:
                        logger.error(f"Failed to open source {self.name}. Exiting.")
                        break
                    time.sleep(2.0)
                    continue

            ret, img = self.cap.read()
            if not ret:
                if self.loop:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                if self.auto_reconnect:
                    self.cap.release()
                    time.sleep(2.0)
                    continue
                break

            video_ts = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            now = time.time()
            
            if self.realtime_pacing and self.frames_read > 0:
                expected_wall_time = self.start_time + video_ts
                sleep_time = expected_wall_time - now
                if sleep_time > 0:
                    time.sleep(sleep_time)
                now = time.time()

            frame = Frame(
                frame_id=self.frames_read,
                wall_clock_timestamp=now,
                video_timestamp=video_ts,
                image=img,
                source_name=self.name
            )
            self._push_frame(frame)
            self.frames_read += 1

        self.running = False
        if self.cap:
            self.cap.release()


class WebcamSource(OpenCVSource):
    def __init__(self, index: int = 0):
        super().__init__("webcam", index, auto_reconnect=True)

class RTSPSource(OpenCVSource):
    def __init__(self, url: str):
        super().__init__("rtsp", url, auto_reconnect=True)

class FileSource(OpenCVSource):
    def __init__(self, filepath: str, realtime_pacing: bool = True, loop: bool = False):
        super().__init__("file", filepath, auto_reconnect=False, realtime_pacing=realtime_pacing, loop=loop)
