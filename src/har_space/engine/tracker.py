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
        
        self._evaluate_steps(event.timestamp)

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
                    for skipped_idx in range(self.current_step_idx, idx):
                        self._skip_step(skipped_idx, video_ts)
                    self._complete_step(idx, video_ts)
                    self._fire_alert(f"Step skipped! You missed: {self.steps[self.current_step_idx-1].name}", "warning", video_ts)
                elif idx < self.current_step_idx and step.id not in self.completed_steps:
                    # Should not happen typically if we mark them skipped, but just in case
                    pass
                elif step.id in self.completed_steps:
                    # Out of sequence (repeating a done step)
                    self._fire_alert("Out of sequence! Step already completed.", "warning", video_ts)
                    self._log_record(step, StepStatus.out_of_sequence, video_ts, "Repeated")

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
