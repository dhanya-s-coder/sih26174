import time
from typing import List, Dict, Optional
from src.har_space.config.models import ExperimentSpec
from src.har_space.data_models import Event, StepStatus, StepRecord, Alert
from src.har_space.events.bus import EventBus
from src.har_space.logging_.writer import StructuredLogWriter

class SimpleStepTracker:
    def __init__(self, spec: ExperimentSpec, event_bus: EventBus, log_writer: StructuredLogWriter, alert_cooldown: float = 4.0):
        self.spec = spec
        self.bus = event_bus
        self.log_writer = log_writer
        self.alert_cooldown = alert_cooldown
        
        self.steps = spec.steps
        self.current_step_idx = 0
        self.last_alert_time = 0.0
        self.completed_steps = set()
        
        # To enforce release between steps
        self.active_predicates = set()
        
        self.bus.subscribe("interaction", self.handle_interaction)

    def handle_interaction(self, event: Event):
        if self.current_step_idx >= len(self.steps):
            return
            
        payload = event.payload
        verb = payload["verb"]
        obj = payload["object"]
        
        # Interaction events are mapped to declarative predicates. Hand events
        # become hand_touches/hand_holds, while object-state events can satisfy
        # object_present/object_removed/object_placed predicates from YAML.
        subject = payload.get("subject", "hand")
        pred = f"{subject}_{verb}({obj})"
        if subject == "object" and payload.get("location"):
            pred = f"object_{verb}({obj},{payload['location']})"
        elif subject == "object" and payload.get("container"):
            pred = f"object_{verb}({obj},{payload['container']})"
        
        if subject == "hand" and verb == "releases":
            # clear hold/touch states for this object
            self.active_predicates.discard(f"hand_touches({obj})")
            self.active_predicates.discard(f"hand_holds({obj})")
            return

        # Detect actions that belong to a later state before accepting them.
        # Example: removing the red box while the FSM is still waiting for
        # yellow, or placing yellow on the wrong side.
        current_step = self.steps[self.current_step_idx]
        current_requirements = {
            f"{req.type}({','.join(req.args)})"
            for req in current_step.required_predicates
        }
        future_requirements = {
            f"{req.type}({','.join(req.args)})"
            for step in self.steps[self.current_step_idx + 1:]
            for req in step.required_predicates
        }
        wrong_location = (
            subject == "object"
            and verb == "placed"
            and payload.get("location")
            and pred not in current_requirements
        )
        if wrong_location:
            expected_location = next(
                (req.args[1] for req in current_step.required_predicates
                 if req.type == "object_placed" and len(req.args) > 1),
                "the required location",
            )
            self._fire_alert(
                f"Wrong placement: {obj.replace('_', ' ').title()} must be placed on the {expected_location}.",
                "warning",
                event.timestamp,
            )
            return

        if pred in future_requirements:
            self._fire_alert(
                f"Out of sequence: {verb.replace('_', ' ').title()} {obj.replace('_', ' ')}.",
                "warning",
                event.timestamp,
            )
            return

        if pred in self.active_predicates:
            return
            
        self.active_predicates.add(pred)
        
        self._evaluate_steps(event.timestamp)

    def _evaluate_steps(self, video_ts: float):
        # Evaluate only the current FSM state. Repeated detections for a
        # completed state must not trigger false out-of-sequence alerts.
        if self.current_step_idx >= len(self.steps):
            return
        step = self.steps[self.current_step_idx]
        satisfied = all(
            f"{req.type}({','.join(req.args)})" in self.active_predicates
            for req in step.required_predicates
        )
        if satisfied:
            self._complete_step(self.current_step_idx, video_ts)

    def _complete_step(self, idx: int, video_ts: float):
        step = self.steps[idx]
        self.completed_steps.add(step.id)
        self.current_step_idx = idx + 1
        
        self._log_record(step, StepStatus.completed, video_ts, step.expected_outcome)
        
        # Announce next step or completion
        if self.current_step_idx < len(self.steps):
            next_step = self.steps[self.current_step_idx]
            self.bus.publish(Event(event_type="speech", timestamp=video_ts, payload={"text": next_step.instruction}))
        else:
            self.bus.publish(Event(event_type="speech", timestamp=video_ts, payload={"text": "Experiment complete."}))
            self._fire_alert("Experiment complete.", "info", video_ts)

    def _skip_step(self, idx: int, video_ts: float):
        step = self.steps[idx]
        self.completed_steps.add(step.id)
        self._log_record(step, StepStatus.skipped, video_ts, "Skipped")

    def _fire_alert(self, message: str, level: str, video_ts: float):
        now = time.time()
        if now - self.last_alert_time >= self.alert_cooldown:
            alert = Alert(message=message, level=level, timestamp=now)
            self.bus.publish(Event(event_type="alert", timestamp=video_ts, payload=alert.model_dump()))
            self.bus.publish(Event(event_type="speech", timestamp=video_ts, payload={"text": message, "priority": True}))
            self.last_alert_time = now

    def _log_record(self, step, status: StepStatus, video_ts: float, outcome: str):
        record = StepRecord(
            iso_timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            video_timestamp=video_ts,
            step_id=step.id,
            step_name=step.name,
            status=status,
            outcome_text=outcome,
            confidence=1.0
        )
        self.log_writer.write_record(record)
        self.bus.publish(Event(
            event_type="step_result",
            timestamp=video_ts,
            payload={
                "step_id": step.id,
                "step_name": step.name,
                "status": status.value,
                "outcome": outcome,
            },
        ))
