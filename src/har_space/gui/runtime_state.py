"""Thread-safe, latest-value runtime handoff for the OpenCV dashboard."""

from collections import deque
from dataclasses import dataclass
from datetime import datetime
import threading
import time
from typing import Any, Deque, Dict, FrozenSet, Optional, Tuple

import numpy as np

from src.har_space.data_models import Alert, Detection, Event, HandState


@dataclass(frozen=True)
class ActivityItem:
    received_at: float
    category: str
    description: str
    level: str = "info"


@dataclass(frozen=True)
class RuntimeSnapshot:
    """A consistent view of runtime values at the time it was requested."""

    frame: Optional[np.ndarray]
    detections: Tuple[Detection, ...]
    hands: Tuple[HandState, ...]
    step_index: int
    step_count: int
    completed_step_ids: FrozenSet[str]
    step_statuses: Dict[str, str]
    active_predicates: Tuple[str, ...]
    current_interaction: str
    latest_interaction: str
    latest_speech: str
    latest_alert: Optional[Alert]
    alert_expected: str
    alert_detected: str
    recent_activity: Tuple[ActivityItem, ...]
    fps: Optional[float]
    health: Dict[str, Tuple[str, str]]
    logging_path: str
    recording_path: str
    streaming_endpoint: str
    experiment_status: str
    error_message: str
    frame_id: Optional[int]
    updated_at: float
    duration_seconds: Optional[float]
    error_count: int


class RuntimeState:
    """Stores the latest pipeline output without copying full-resolution frames."""

    def __init__(self, logging_path: str, step_count: int, step_names: Tuple[str, ...] = ()) -> None:
        self._lock = threading.RLock()
        self._frame: Optional[np.ndarray] = None
        self._detections: Tuple[Detection, ...] = ()
        self._hands: Tuple[HandState, ...] = ()
        self._step_index = 0
        self._step_count = step_count
        self._completed_step_ids: FrozenSet[str] = frozenset()
        self._step_statuses: Dict[str, str] = {}
        self._active_predicates: Tuple[str, ...] = ()
        self._active_interactions: Dict[str, str] = {}
        self._current_interaction = ""
        self._latest_interaction = ""
        self._latest_speech = ""
        self._latest_alert: Optional[Alert] = None
        self._alert_expected = ""
        self._alert_detected = ""
        self._step_names = step_names
        self._started_at = time.time()
        self._completed_at: Optional[float] = None
        self._error_count = 0
        self._activity: Deque[ActivityItem] = deque(maxlen=100)
        self._fps: Optional[float] = None
        self._health: Dict[str, Tuple[str, str]] = {}
        self._logging_path = logging_path
        self._recording_path = ""
        self._streaming_endpoint = ""
        self._experiment_status = "STARTING"
        self._error_message = ""
        self._frame_id: Optional[int] = None
        self._updated_at = time.time()

    def record_event(self, event: Event) -> None:
        """Capture an actual EventBus event for display and recent activity."""
        now = time.time()
        event_type = event.event_type.lower()
        payload = event.payload
        level = "info"
        if event_type == "interaction":
            raw_verb = str(payload.get("verb", "interaction"))
            raw_object = str(payload.get("object", "N/A"))
            verb = raw_verb.upper()
            obj = raw_object.replace("_", " ").upper()
            description = f"{verb} {obj}"
            category = "INTERACTION"
        elif event_type == "speech":
            description = str(payload.get("text", "")) or "Speech event"
            category = "VOICE"
            level = "warning" if payload.get("priority") else "info"
        elif event_type == "alert":
            alert = Alert(**payload)
            description = alert.message
            category = "ALERT"
            level = alert.level.lower()
        elif event_type == "experiment_reset":
            description = str(payload.get("message", "Experiment reset"))
            category = "EXPERIMENT"
        else:
            description = str(payload)[:120]
            category = event_type.upper()
        with self._lock:
            if event_type == "interaction":
                self._latest_interaction = description
                if raw_verb == "releases":
                    self._active_interactions.pop(raw_object, None)
                else:
                    self._active_interactions[raw_object] = f"{verb} {obj}"
                self._current_interaction = next(reversed(self._active_interactions.values()), "")
            elif event_type == "speech":
                self._latest_speech = description
            elif event_type == "alert":
                self._latest_alert = alert
                current_index = min(self._step_index, len(self._step_names) - 1)
                self._alert_expected = self._step_names[current_index] if self._step_names else "Unavailable"
                self._alert_detected = self._latest_interaction or "Unavailable"
            repeated_interaction = (
                event_type == "interaction"
                and self._activity
                and self._activity[0].category == category
                and self._activity[0].description == description
            )
            if not repeated_interaction:
                self._activity.appendleft(ActivityItem(now, category, description, level))
            self._updated_at = now

    def record_step(self, record: Any) -> None:
        """Add a real step-log record to the activity feed."""
        status = str(getattr(record.status, "value", record.status)).upper()
        now = time.time()
        with self._lock:
            self._step_statuses[record.step_id] = status.lower()
            if status in {"SKIPPED", "OUT_OF_SEQUENCE", "FAILED"}:
                self._error_count += 1
            self._activity.appendleft(
                ActivityItem(now, "STEP", f"{record.step_name} - {status}",
                             "warning" if status in {"SKIPPED", "OUT_OF_SEQUENCE", "FAILED"} else "success")
            )
            self._updated_at = now

    def update_pipeline(
        self,
        *,
        frame: np.ndarray,
        frame_id: int,
        detections: Tuple[Detection, ...],
        hands: Tuple[HandState, ...],
        step_index: int,
        completed_step_ids: FrozenSet[str],
        active_predicates: Tuple[str, ...],
        fps: Optional[float],
        health: Dict[str, Tuple[str, str]],
        recording_path: str = "",
        streaming_endpoint: str = "",
    ) -> None:
        """Atomically publish data already computed by the existing pipeline."""
        with self._lock:
            self._frame = frame
            self._frame_id = frame_id
            self._detections = detections
            self._hands = hands
            self._step_index = step_index
            self._completed_step_ids = completed_step_ids
            self._active_predicates = active_predicates
            self._fps = fps
            self._health = dict(health)
            self._recording_path = recording_path
            self._streaming_endpoint = streaming_endpoint
            self._experiment_status = "COMPLETE" if step_index >= self._step_count else "RUNNING"
            if self._experiment_status == "COMPLETE" and self._completed_at is None:
                self._completed_at = time.time()
            self._updated_at = time.time()

    def update_health(self, health: Dict[str, Tuple[str, str]]) -> None:
        with self._lock:
            self._health = dict(health)
            self._updated_at = time.time()

    def reset_protocol_view(self) -> None:
        """Clear displayed per-step outcomes when the existing tracker is reset."""
        with self._lock:
            self._step_index = 0
            self._completed_step_ids = frozenset()
            self._active_predicates = ()
            self._step_statuses.clear()
            self._active_interactions.clear()
            self._current_interaction = ""
            self._latest_interaction = ""
            self._latest_speech = ""
            self._latest_alert = None
            self._alert_expected = ""
            self._alert_detected = ""
            self._error_count = 0
            self._activity.clear()
            self._completed_at = None
            self._started_at = time.time()
            self._experiment_status = "RUNNING"
            self._updated_at = time.time()

    def set_error(self, message: str) -> None:
        now = time.time()
        with self._lock:
            self._error_message = message
            self._experiment_status = "ERROR"
            self._activity.appendleft(ActivityItem(now, "SYSTEM", message, "error"))
            self._updated_at = now

    def mark_stopped(self) -> None:
        with self._lock:
            if self._experiment_status not in {"COMPLETE", "ERROR"}:
                self._experiment_status = "STOPPED"
            self._updated_at = time.time()

    def snapshot(self) -> RuntimeSnapshot:
        with self._lock:
            return RuntimeSnapshot(
                frame=self._frame,
                detections=self._detections,
                hands=self._hands,
                step_index=self._step_index,
                step_count=self._step_count,
                completed_step_ids=self._completed_step_ids,
                step_statuses=dict(self._step_statuses),
                active_predicates=self._active_predicates,
                current_interaction=self._current_interaction,
                latest_interaction=self._latest_interaction,
                latest_speech=self._latest_speech,
                latest_alert=self._latest_alert,
                alert_expected=self._alert_expected,
                alert_detected=self._alert_detected,
                recent_activity=tuple(self._activity),
                fps=self._fps,
                health=dict(self._health),
                logging_path=self._logging_path,
                recording_path=self._recording_path,
                streaming_endpoint=self._streaming_endpoint,
                experiment_status=self._experiment_status,
                error_message=self._error_message,
                frame_id=self._frame_id,
                updated_at=self._updated_at,
                duration_seconds=max(0.0, (self._completed_at or time.time()) - self._started_at),
                error_count=self._error_count,
            )


def format_activity_time(timestamp: float) -> str:
    """Format an event's local receive time for the activity table."""
    return datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")
