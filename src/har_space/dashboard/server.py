import os
import sys
import json
import time
import cv2
import numpy as np
import threading
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
from pathlib import Path

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

class AstroHARState:
    """Thread-safe global state store for AstroHAR Dashboard."""
    def __init__(self):
        self.lock = threading.Lock()
        self.start_time = time.time()
        self.fps = 29.8
        self.inference_ms = 78.0
        self.current_step_idx = 0
        self.completed_steps = []
        self.step_statuses = {}
        self.source_mode = "live"
        
        self.steps = [
            {"id": "step_001", "name": "Main Box Present", "instruction": "The main box is ready. Take out the yellow box.", "object": "Main Box", "action": "OBJECT_PRESENT_MAIN_BOX"},
            {"id": "step_002", "name": "Take Out Yellow Box", "instruction": "Take the yellow box out of the main box.", "object": "Yellow Box", "action": "OBJECT_REMOVED_YELLOW_BOX"},
            {"id": "step_003", "name": "Place Yellow Box Left", "instruction": "Place the yellow box on the left.", "object": "Yellow Box", "action": "OBJECT_PLACED_YELLOW_LEFT"},
            {"id": "step_004", "name": "Take Out Red Box", "instruction": "Take the red box out of the main box.", "object": "Red Box", "action": "OBJECT_REMOVED_RED_BOX"},
            {"id": "step_005", "name": "Place Red Box Right", "instruction": "Place the red box on the right.", "object": "Red Box", "action": "OBJECT_PLACED_RED_RIGHT"}
        ]
        
        self.action_probs = {}
        
        self.detections = {}
        
        self.voice_text = "Waiting for an instruction."
        self.latest_frame = None
        self.logs = []
        self.pending_command = None

    def get_dict(self):
        with self.lock:
            elapsed = time.time() - self.start_time
            mins = int(elapsed // 60)
            secs = int(elapsed % 60)
            return {
                "system_online": True,
                "source_mode": self.source_mode,
                "fps": round(self.fps, 1),
                "inference_ms": self.inference_ms,
                "current_step_idx": self.current_step_idx,
                "completed_steps": self.completed_steps,
                "step_statuses": self.step_statuses,
                "steps": self.steps,
                "action_probs": self.action_probs,
                "detections": self.detections,
                "voice_text": self.voice_text,
                "elapsed": f"{mins} min {secs} sec",
                "logs": self.logs
            }

    def configure_steps(self, steps):
        with self.lock:
            self.start_time = time.time()
            self.steps = steps
            self.current_step_idx = 0
            self.completed_steps = []
            self.step_statuses = {}
            self.action_probs = {}
            self.detections = {}
            self.voice_text = "Waiting for an instruction."
            self.latest_frame = None
            self.logs = []

    def configure_source(self, source_mode):
        with self.lock:
            self.source_mode = source_mode

    def request_source(self, source):
        with self.lock:
            self.pending_command = {"action": "source", "source": source}

    def request_quit(self):
        with self.lock:
            self.pending_command = {"action": "quit"}

    def consume_command(self):
        with self.lock:
            command = self.pending_command
            self.pending_command = None
            return command

    def update_live_data(self, frame_img=None, current_step_idx=None, fps=None,
                         detections=None, completed_steps=None,
                         action_probs=None, log_entry=None, voice_text=None):
        with self.lock:
            if frame_img is not None:
                self.latest_frame = frame_img.copy()
            if current_step_idx is not None:
                self.current_step_idx = current_step_idx
            if fps is not None:
                self.fps = fps
            if detections is not None:
                self.detections = detections
            if completed_steps is not None:
                self.completed_steps = completed_steps
            if action_probs is not None:
                self.action_probs = action_probs
            if log_entry is not None:
                self.logs.append(log_entry)
                self.logs = self.logs[-50:]
            if voice_text is not None:
                self.voice_text = voice_text

    def update_step_status(self, step_id, status):
        with self.lock:
            self.step_statuses[step_id] = status

    def get_latest_frame(self):
        with self.lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
            return None

global_state = AstroHARState()

class DashboardHTTPHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            self.serve_file(STATIC_DIR / "index.html", "text/html")
        elif path == "/styles.css":
            self.serve_file(STATIC_DIR / "styles.css", "text/css")
        elif path == "/dashboard.js":
            self.serve_file(STATIC_DIR / "dashboard.js", "application/javascript")
        elif path == "/api/state":
            self.send_json(global_state.get_dict())
        elif path == "/stream":
            self.serve_mjpeg_stream()
        else:
            self.send_error(404, "File Not Found")

    def do_POST(self):
        if urlparse(self.path).path != "/api/control":
            self.send_error(404, "Not Found")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            action = payload.get("action")
            if action == "webcam":
                global_state.request_source("webcam")
            elif action == "recording":
                global_state.request_source("recording")
            elif action == "recording2":
                global_state.request_source("recording2")
            elif action == "quit":
                global_state.request_quit()
            else:
                self.send_error(400, "Unknown action")
                return
            self.send_json({"ok": True, "action": action})
        except Exception as exc:
            self.send_error(400, str(exc))

    def serve_file(self, filepath: Path, content_type: str):
        if not filepath.exists():
            self.send_error(404, "File Not Found")
            return
        
        with open(filepath, "rb") as f:
            content = f.read()
        
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def send_json(self, data: dict):
        body = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def serve_mjpeg_stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache, private")
        self.send_header("Pragma", "no-cache")
        self.end_headers()

        width, height = 640, 360
        frame_cnt = 0
        
        try:
            while True:
                img = global_state.get_latest_frame()
                if img is None:
                    # Render futuristic synthetic camera feed
                    img = np.zeros((height, width, 3), dtype=np.uint8)
                    img[:] = (15, 10, 5) # Dark space blue background
                    
                    # Moving HUD target
                    cx = int(width * 0.5 + 50 * np.sin(frame_cnt * 0.05))
                    cy = int(height * 0.5 + 30 * np.cos(frame_cnt * 0.05))
                    
                    # Bounding boxes
                    cv2.rectangle(img, (cx-40, cy-40), (cx+40, cy+40), (254, 242, 0), 2)
                    cv2.putText(img, "Red Box [0.92]", (cx-40, cy-45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (254, 242, 0), 1)
                    
                    cv2.rectangle(img, (80, 60), (560, 300), (255, 185, 16), 1)
                    cv2.putText(img, "Main Payload Box [0.95]", (85, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 185, 16), 1)

                    # Frame ticker & timestamp
                    cv2.putText(img, f"AstroHAR STANDBY - CAMERA CONNECTING... {frame_cnt}", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 242, 254), 1)
                    frame_cnt += 1

                ret, jpeg = cv2.imencode(".jpg", img)
                if not ret:
                    time.sleep(0.03)
                    continue

                frame_bytes = (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(jpeg)).encode("utf-8") + b"\r\n\r\n" +
                    jpeg.tobytes() + b"\r\n"
                )
                self.wfile.write(frame_bytes)
                self.wfile.flush()
                time.sleep(0.03)
        except Exception:
            pass

    def log_message(self, format, *args):
        pass # Suppress verbose server log output

def start_dashboard_server(host: str = "0.0.0.0", port: int = 8080):
    server = ThreadingHTTPServer((host, port), DashboardHTTPHandler)
    server.daemon_threads = True
    print("\n=======================================================")
    print(" [AstroHAR] Mission Control Dashboard Server Online")
    print(f" URL: http://localhost:{port}/")
    print(" Status: Safe • Smart • Autonomous")
    print("=======================================================\n")
    server.serve_forever()

if __name__ == "__main__":
    port = 8080
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    start_dashboard_server(port=port)
