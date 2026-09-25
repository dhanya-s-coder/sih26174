import time
import pytest
from src.har_space.config.models import ExperimentSpec, Predicate, ExperimentStep
from src.har_space.events.bus import EventBus
from src.har_space.engine.tracker import SimpleStepTracker
from src.har_space.logging_.writer import StructuredLogWriter
from src.har_space.data_models import Event, HandState, Detection
from src.har_space.events.interactions import InteractionExtractor
from src.har_space.perception.color_detector import ColorBoxDetector
import numpy as np

@pytest.fixture
def spec():
    return ExperimentSpec(
        name="Test",
        version="1.0",
        steps=[
            ExperimentStep(id="1", name="Step 1", instruction="Do 1", position=1, 
                           required_predicates=[Predicate(type="hand_touches", args=["red_box"])], expected_outcome="Done 1"),
            ExperimentStep(id="2", name="Step 2", instruction="Do 2", position=2, 
                           required_predicates=[Predicate(type="hand_touches", args=["yellow_box"])], expected_outcome="Done 2"),
        ]
    )

@pytest.fixture
def tracker_setup(tmp_path, spec):
    bus = EventBus()
    writer = StructuredLogWriter(str(tmp_path / "log.jsonl"))
    tracker = SimpleStepTracker(spec, bus, writer, alert_cooldown=1.0)
    return bus, tracker, writer

def test_correct_sequence(tracker_setup):
    bus, tracker, writer = tracker_setup
    
    # Touch red box -> Step 1
    bus.publish(Event(event_type="interaction", timestamp=0.1, payload={"subject": "hand", "verb": "touches", "object": "red_box"}))
    assert tracker.current_step_idx == 1
    
    # Touch yellow box -> Step 2
    bus.publish(Event(event_type="interaction", timestamp=0.2, payload={"subject": "hand", "verb": "touches", "object": "yellow_box"}))
    assert tracker.current_step_idx == 2

def test_skipped_step(tracker_setup):
    bus, tracker, writer = tracker_setup
    
    # Touch yellow box directly -> Skips Step 1
    bus.publish(Event(event_type="interaction", timestamp=0.1, payload={"subject": "hand", "verb": "touches", "object": "yellow_box"}))
    assert tracker.current_step_idx == 2
    
    # Check logs
    assert len(writer.records) == 2
    assert writer.records[0].status == "skipped"
    assert writer.records[1].status == "completed"

def test_repeated_step_and_cooldown(tracker_setup):
    bus, tracker, writer = tracker_setup
    
    alerts_fired = []
    bus.subscribe("alert", lambda e: alerts_fired.append(e))
    
    # Complete Step 1
    bus.publish(Event(event_type="interaction", timestamp=0.1, payload={"subject": "hand", "verb": "touches", "object": "red_box"}))
    bus.publish(Event(event_type="interaction", timestamp=0.2, payload={"subject": "hand", "verb": "releases", "object": "red_box"}))
    
    # Repeat Step 1
    bus.publish(Event(event_type="interaction", timestamp=0.3, payload={"subject": "hand", "verb": "touches", "object": "red_box"}))
    
    # Alert should fire
    assert len(alerts_fired) == 1
    
    # Repeat immediately (should hit cooldown)
    bus.publish(Event(event_type="interaction", timestamp=0.4, payload={"subject": "hand", "verb": "touches", "object": "red_box"}))
    assert len(alerts_fired) == 1
    
    # Wait for cooldown and repeat
    tracker.last_alert_time -= 5.0  # mock time passing (cooldown is 4.0 in real, 1.0 in test)
    bus.publish(Event(event_type="interaction", timestamp=0.45, payload={"subject": "hand", "verb": "releases", "object": "red_box"}))
    bus.publish(Event(event_type="interaction", timestamp=0.5, payload={"subject": "hand", "verb": "touches", "object": "red_box"}))
    assert len(alerts_fired) == 2

def test_release_required(tracker_setup):
    bus, tracker, writer = tracker_setup
    
    # Touch red box (completes step 1)
    bus.publish(Event(event_type="interaction", timestamp=0.1, payload={"subject": "hand", "verb": "touches", "object": "red_box"}))
    assert tracker.current_step_idx == 1
    
    # Without releasing, touching red box again does nothing new since the predicate is already active
    # (Tracker sets it in active_predicates and evaluates on each event)
    bus.publish(Event(event_type="interaction", timestamp=0.2, payload={"subject": "hand", "verb": "touches", "object": "red_box"}))
    
    # Release it
    bus.publish(Event(event_type="interaction", timestamp=0.3, payload={"subject": "hand", "verb": "releases", "object": "red_box"}))
    
    # Touch again -> Repeated step
    bus.publish(Event(event_type="interaction", timestamp=0.4, payload={"subject": "hand", "verb": "touches", "object": "red_box"}))
    assert len(writer.records) == 2
    assert writer.records[1].status == "out_of_sequence"

def test_color_box_detector():
    det = ColorBoxDetector()
    det.ranges = {
        "red_box": {"lower": [0, 100, 100], "upper": [10, 255, 255]},
        "yellow_box": {"lower": [20, 100, 100], "upper": [30, 255, 255]}
    }
    
    # Create an image with a red and yellow square
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Red is hue 0
    img[10:50, 10:50] = [0, 0, 255] # BGR red (hue 0) -> 40x40 = 1600 area
    # Yellow is hue ~30
    img[50:90, 50:90] = [0, 255, 255] # BGR yellow (hue 30) -> 40x40 = 1600 area
    
    detections = det.process(img)
    assert len(detections) == 2
    labels = set([d.label for d in detections])
    assert "red_box" in labels
    assert "yellow_box" in labels

def test_interaction_hysteresis():
    bus = EventBus()
    events = []
    bus.subscribe("interaction", lambda e: events.append(e))
    
    ex = InteractionExtractor(bus, padding=0.0, min_frames=2, release_frames=2)
    
    # Object at 0.0 to 0.5
    obj = Detection(label="red_box", confidence=1.0, bbox=[0.0, 0.0, 0.5, 0.5])
    
    # Hand outside
    h_out = HandState(hand_type="right", bbox=[0.6, 0.6, 0.9, 0.9], landmarks=[[0.7, 0.7, 0.0]], confidence=1.0)
    # Hand inside
    h_in = HandState(hand_type="right", bbox=[0.2, 0.2, 0.4, 0.4], landmarks=[[0.3, 0.3, 0.0]], confidence=1.0)
    
    # Frame 1: inside (not enough for min_frames)
    ex.process([h_in], [obj], 0.1)
    assert len(events) == 0
    
    # Frame 2: inside (triggers touches)
    ex.process([h_in], [obj], 0.2)
    assert len(events) == 1
    assert events[-1].payload["verb"] == "touches"
    
    # Frame 3: outside (not enough for release)
    ex.process([h_out], [obj], 0.3)
    assert len(events) == 1
    
    # Frame 4: outside (triggers release)
    ex.process([h_out], [obj], 0.4)
    assert len(events) == 2
    assert events[-1].payload["verb"] == "releases"
