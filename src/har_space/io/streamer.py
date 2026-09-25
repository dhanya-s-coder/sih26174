import cv2
import threading
import subprocess
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional
from .models import Frame

logger = logging.getLogger(__name__)

class MJPEGHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=--jpgboundary')
            self.end_headers()
            
            streamer = self.server.streamer
            while streamer.running:
                frame = streamer.latest_frame
                if frame is not None:
                    ret, jpeg = cv2.imencode('.jpg', frame.image)
                    if ret:
                        try:
                            self.wfile.write(b'--jpgboundary\r\n')
                            self.send_header('Content-type', 'image/jpeg')
                            self.send_header('Content-length', str(len(jpeg)))
                            self.end_headers()
                            self.wfile.write(jpeg.tobytes())
                            self.wfile.write(b'\r\n')
                        except Exception as e:
                            break
                streamer.new_frame_event.wait(timeout=0.1)
                streamer.new_frame_event.clear()
        else:
            self.send_response(404)
            self.end_headers()
            
    def log_message(self, format, *args):
        # Suppress standard logging for each request
        pass

class VideoStreamer:
    def __init__(self, mode: str, host: str, port: int):
        self.mode = mode
        self.host = host
        self.port = port
        self.running = False
        self.latest_frame: Optional[Frame] = None
        self.new_frame_event = threading.Event()
        self.server = None
        self.server_thread = None

        if self.mode == "ffmpeg":
            import shutil
            if not shutil.which("ffmpeg"):
                logger.warning("ffmpeg not found on PATH. Falling back to mjpeg.")
                self.mode = "mjpeg"
                
    def start(self):
        self.running = True
        if self.mode == "mjpeg":
            self.server = ThreadingHTTPServer((self.host, self.port), MJPEGHandler)
            self.server.daemon_threads = True
            self.server.streamer = self
            self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
            self.server_thread.start()
            logger.info(f"MJPEG server started at http://{self.host}:{self.port}/")
        elif self.mode == "ffmpeg":
            # Simple UDP MPEGTS push via stdin
            self.ffmpeg_cmd = [
                'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
                '-pix_fmt', 'bgr24', '-s', '640x480', '-r', '30',
                '-i', '-', '-c:v', 'libx264', '-preset', 'ultrafast', '-tune', 'zerolatency',
                '-f', 'mpegts', f'udp://{self.host}:{self.port}'
            ]
            self.ffmpeg_proc = subprocess.Popen(self.ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
            logger.info(f"FFmpeg UDP stream started to {self.host}:{self.port}")

    def stop(self):
        self.running = False
        if self.mode == "mjpeg" and self.server:
            self.server.shutdown()
            self.server.server_close()
            if self.server_thread:
                self.server_thread.join()
        elif self.mode == "ffmpeg" and hasattr(self, 'ffmpeg_proc'):
            self.ffmpeg_proc.terminate()
            self.ffmpeg_proc.wait()

    def push(self, frame: Frame):
        if not self.running:
            return
        
        if self.mode == "mjpeg":
            self.latest_frame = frame
            self.new_frame_event.set()
        elif self.mode == "ffmpeg" and self.ffmpeg_proc.poll() is None:
            try:
                # Assuming frame is 640x480
                img = cv2.resize(frame.image, (640, 480))
                self.ffmpeg_proc.stdin.write(img.tobytes())
            except Exception:
                pass
