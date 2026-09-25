import threading
import queue
import logging
from src.har_space.events.bus import EventBus
from src.har_space.data_models import Event

logger = logging.getLogger(__name__)

class VoiceAlerter:
    def __init__(self, event_bus: EventBus, mute: bool = False):
        self.bus = event_bus
        self.mute = mute
        self.queue = queue.Queue()
        self.running = False
        self.thread = None
        self._state_lock = threading.Lock()
        self._status = "MUTED" if mute else "STOPPED"
        self._latest_message = ""
        self._error_message = ""
        
        self.bus.subscribe("speech", self.handle_speech_event)

    @property
    def status(self):
        with self._state_lock:
            return self._status

    @property
    def latest_message(self):
        with self._state_lock:
            return self._latest_message

    @property
    def error_message(self):
        with self._state_lock:
            return self._error_message

    def start(self):
        if self.mute:
            with self._state_lock:
                self._status = "MUTED"
            return
        self.running = True
        with self._state_lock:
            self._status = "STARTING"
            self._error_message = ""
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        self.queue.put(None) # wakeup
        if self.thread:
            self.thread.join(timeout=2.0)
        with self._state_lock:
            if self._status != "ERROR":
                self._status = "STOPPED"

    def handle_speech_event(self, event: Event):
        if self.mute:
            return
        
        text = event.payload.get("text", "")
        if not text:
            return
        with self._state_lock:
            self._latest_message = text
        if event.payload.get("reset_queue", False):
            # A protocol reset invalidates pending guidance from the prior run.
            with self.queue.mutex:
                self.queue.queue.clear()
        # Preserve FIFO speech for normal events: an alert must not discard a
        # queued instruction announcing a newly active experiment step.
        self.queue.put(text)

    def _run(self):
        try:
            import pyttsx3
            engine = pyttsx3.init('sapi5')
        except Exception as e:
            logger.warning(f"Failed to initialize pyttsx3: {e}. Voice alerts disabled.")
            with self._state_lock:
                self._status = "ERROR"
                self._error_message = str(e)
            return

        with self._state_lock:
            self._status = "READY"

        while self.running:
            text = self.queue.get()
            if text is None:
                break
                
            try:
                with self._state_lock:
                    self._status = "PLAYING"
                engine.say(text)
                engine.runAndWait()
                with self._state_lock:
                    if self.running:
                        self._status = "READY"
            except Exception as e:
                logger.warning(f"TTS error: {e}. Reinitializing...")
                with self._state_lock:
                    self._status = "ERROR"
                    self._error_message = str(e)
                try:
                    engine = pyttsx3.init('sapi5')
                    with self._state_lock:
                        self._error_message = ""
                        if self.running:
                            self._status = "READY"
                except Exception as reinit_error:
                    with self._state_lock:
                        self._error_message = str(reinit_error)
