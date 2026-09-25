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
        
        self.bus.subscribe("speech", self.handle_speech_event)

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        self.queue.put(None) # wakeup
        if self.thread:
            self.thread.join(timeout=2.0)

    def handle_speech_event(self, event: Event):
        if self.mute:
            return
        
        text = event.payload.get("text", "")
        priority = event.payload.get("priority", False)
        
        if priority:
            # Clear pending standard speech
            with self.queue.mutex:
                self.queue.queue.clear()
        
        self.queue.put(text)

    def _run(self):
        try:
            import pyttsx3
        except Exception as e:
            logger.warning(f"Failed to initialize pyttsx3: {e}. Voice alerts disabled.")
            return

        while self.running:
            text = self.queue.get()
            if text is None:
                break

            if not text:
                continue

            engine = None
            try:
                # Recreate the SAPI engine for each utterance. On Windows, a
                # long-lived pyttsx3 engine can stop consuming later messages.
                engine = pyttsx3.init('sapi5')
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                logger.warning(f"TTS error: {e}. Retrying instruction...")
                try:
                    if engine is not None:
                        engine.stop()
                    engine = pyttsx3.init('sapi5')
                    engine.say(text)
                    engine.runAndWait()
                except Exception as retry_error:
                    logger.warning(f"TTS retry failed: {retry_error}")
            finally:
                if engine is not None:
                    try:
                        engine.stop()
                    except Exception:
                        pass
