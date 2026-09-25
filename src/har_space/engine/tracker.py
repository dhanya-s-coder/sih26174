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
        
        pred = f"hand_{verb}({obj})"
        
        if verb == "releases":
            # clear hold/touch states for this object
            self.active_predicates.discard(f"hand_touches({obj})")
            self.active_predicates.discard(f"hand_holds({obj})")
            return
            
        if pred in self.active_predicates:
            return
            
        self.active_predicates.add(pred)

        repeated_step = next(
            (
                step for step in self.steps
                if step.id in self.completed_steps
                and any(f"{req.type}({req.args[0]})" == pred for req in step.required_predicates)
            ),
            None,
        )
        if repeated_step is not None:
            self._fire_alert("Out of sequence! Step already completed.", "warning", event.timestamp)
            self._log_record(repeated_step, StepStatus.out_of_sequence, event.timestamp, "Repeated")
            return
        
        self._evaluate_steps(event.timestamp)

    def reset(self) -> None:
        """Reset only protocol state; the owning runtime calls this on its pipeline thread."""
        self.current_step_idx = 0
        self.last_alert_time = 0.0
        self.completed_steps.clear()
        self.active_predicates.clear()

    def _evaluate_steps(self, video_ts: float):
        # Check all steps to see if the current active predicates satisfy them
        for idx, step in enumerate(self.steps):
            satisfied = True
            for req in step.required_predicates:
                req_str = f"{req.type}({req.args[0]})"
                if req_str not in self.active_predicates:
                    satisfied = False
                    break
                    
            if satisfied:
                if idx == self.current_step_idx:
                    # Correct step
                    self._complete_step(idx, video_ts)
                elif idx > self.current_step_idx:
                    # Skipped steps
                    skipped_steps = self.steps[self.current_step_idx:idx]
                    for skipped_idx in range(self.current_step_idx, idx):
                        self._skip_step(skipped_idx, video_ts)
                    self._complete_step(idx, video_ts)
                    missed = ", ".join(step.name for step in skipped_steps)
                    self._fire_alert(f"Step skipped! You missed: {missed}", "warning", video_ts)
                # Completed prior-step predicates may remain active until hand release.
                # They are not new actions and must not produce false violations.

    def _complete_step(self, idx: int, video_ts: float):
        step = self.steps[idx]
        self.completed_steps.add(step.id)
        self.current_step_idx = idx + 1
        
        self._log_record(step, StepStatus.completed, video_ts, step.expected_outcome)
        
        # Announce next step or completion
        if self.current_step_idx < len(self.steps):
            next_step = self.steps[self.current_step_idx]
            if next_step.instruction and next_step.instruction.strip():
                self.bus.publish(Event(event_type="speech", timestamp=video_ts, payload={"text": next_step.instruction}))
        else:
            self.bus.publish(Event(event_type="speech", timestamp=video_ts, payload={"text": "Experiment complete."}))
            completion_alert = Alert(message="Experiment complete.", level="info", timestamp=time.time())
            self.bus.publish(Event(event_type="alert", timestamp=video_ts, payload=completion_alert.model_dump()))

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
