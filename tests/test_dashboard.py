import queue
import threading
import time
from types import SimpleNamespace

import numpy as np

from fastapi.testclient import TestClient

from src.har_space.api.server import create_app
from src.har_space.api.runtime import HARApplication
from src.har_space.config.models import ExperimentSpec, ExperimentStep, Predicate
from src.har_space.data_models import Event, StepRecord, StepStatus
from src.har_space.engine.tracker import SimpleStepTracker
from src.har_space.events.bus import EventBus
from src.har_space.events.interactions import InteractionExtractor
from src.har_space.gui.runtime_state import RuntimeState
from src.har_space.logging_.writer import StructuredLogWriter


def make_spec():
    return ExperimentSpec(
        name="Configured test protocol",
        version="1.0",
        steps=[
            ExperimentStep(
                id="configured_step_a",
                name="Touch configured red box",
                instruction="Touch the configured red box",
                position=1,
                required_predicates=[Predicate(type="hand_touches", args=["red_box"])],
                expected_outcome="Red box touched",
            ),
            ExperimentStep(
                id="configured_step_b",
                name="Touch configured yellow box",
                instruction="Touch the configured yellow box",
                position=2,
                required_predicates=[Predicate(type="hand_touches", args=["yellow_box"])],
                expected_outcome="Yellow box touched",
            ),
        ],
    )


def make_test_runtime(tmp_path):
    spec = make_spec()
    log_path = tmp_path / "steps.jsonl"
    state = RuntimeState(str(log_path), len(spec.steps), tuple(step.name for step in spec.steps))
    state.update_pipeline(
        frame=None,
        frame_id=12,
        detections=(),
        hands=(),
        step_index=1,
        completed_step_ids=frozenset({"configured_step_a"}),
        active_predicates=("hand_touches(yellow_box)",),
        fps=8.5,
        health={
            "camera": ("ONLINE", ""), "detector": ("RUNNING", ""),
            "hands": ("RUNNING", ""), "tracker": ("RUNNING", ""),
            "tts": ("READY", ""),
        },
    )
    record = StepRecord(
        iso_timestamp="2026-09-25T12:00:00Z", video_timestamp=1.0,
        step_id="configured_step_a", step_name=spec.steps[0].name,
        status=StepStatus.completed, outcome_text="Red box touched", confidence=1.0,
    )
    state.record_step(record)
    state.record_event(Event(event_type="interaction", timestamp=1.1,
                             payload={"subject": "hand", "verb": "touches", "object": "yellow_box"}))
    state.record_event(Event(event_type="speech", timestamp=1.2,
                             payload={"text": "Touch the yellow box"}))
    runtime = SimpleNamespace(runtime=state, spec=spec, streamer=SimpleNamespace(port=8081),
                              log_path=log_path, log_writer=SimpleNamespace(records=[record]))
    return runtime, state


def test_rest_endpoints_reflect_existing_protocol_runtime(tmp_path):
    runtime, _ = make_test_runtime(tmp_path)
    client = TestClient(create_app(runtime))

    assert client.get("/api/health").json()["status"] == "ok"
    status = client.get("/api/status").json()
    assert status["health"]["camera"] == "ONLINE"
    assert client.get("/api/experiment").json()["name"] == "Configured test protocol"

    progress = client.get("/api/progress").json()
    assert progress["current_step"]["name"] == "Touch configured yellow box"
    assert [step["status"] for step in progress["steps"]] == ["completed", "current"]

    state = client.get("/api/state").json()
    assert state["action"] == "TOUCHES YELLOW BOX"
    assert state["assistant"]["message"] == "Touch the yellow box"
    assert state["camera"]["stream_url"] == "http://127.0.0.1:8081/"
    logs = client.get("/api/logs").json()["items"]
    assert any(item["event"] == "Touch configured red box - COMPLETED" for item in logs)
    assert any(item["event"] == "TOUCHES YELLOW BOX" for item in logs)
    step_records = client.get("/api/logs").json()["step_records"]
    assert len(step_records) == 1
    assert step_records[0]["event"] == "Touch configured red box — COMPLETED: Red box touched"


def test_websocket_sends_runtime_update_envelope(tmp_path):
    runtime, state = make_test_runtime(tmp_path)
    client = TestClient(create_app(runtime))

    with client.websocket_connect("/ws") as websocket:
        message = websocket.receive_json()
        state.record_event(Event(event_type="speech", timestamp=2.0,
                                 payload={"text": "The next configured step."}))
        updated = websocket.receive_json()

    assert message["type"] == "runtime_update"
    assert message["data"]["experiment"]["name"] == "Configured test protocol"
    assert message["data"]["progress"]["steps"][0]["status"] == "completed"
    assert updated["data"]["assistant"]["message"] == "The next configured step."


def test_existing_tracker_violation_reaches_api_state(tmp_path):
    spec = make_spec()
    bus = EventBus()
    runtime_state = RuntimeState("steps.jsonl", len(spec.steps), tuple(step.name for step in spec.steps))
    for event_type in ("interaction", "alert", "speech"):
        bus.subscribe(event_type, runtime_state.record_event)
    tracker = SimpleStepTracker(
        spec, bus, StructuredLogWriter(str(tmp_path / "steps.jsonl")), alert_cooldown=0
    )

    bus.publish(Event(event_type="interaction", timestamp=1.0,
                      payload={"subject": "hand", "verb": "touches", "object": "red_box"}))
    runtime_state.update_pipeline(
        frame=None, frame_id=1, detections=(), hands=(), step_index=tracker.current_step_idx,
        completed_step_ids=frozenset(tracker.completed_steps),
        active_predicates=tuple(sorted(tracker.active_predicates)), fps=None, health={},
    )
    bus.publish(Event(event_type="interaction", timestamp=1.1,
                      payload={"subject": "hand", "verb": "releases", "object": "red_box"}))
    bus.publish(Event(event_type="interaction", timestamp=1.2,
                      payload={"subject": "hand", "verb": "touches", "object": "red_box"}))

    runtime = SimpleNamespace(runtime=runtime_state, spec=spec,
                              streamer=SimpleNamespace(port=8081), log_path=tmp_path / "steps.jsonl")
    alert = TestClient(create_app(runtime)).get("/api/state").json()["alert"]
    assert alert["level"] == "warning"
    assert alert["expected"] == "Touch configured yellow box"
    assert alert["detected"] == "TOUCHES RED BOX"
    assert "Out of sequence" in alert["message"]


def test_reset_endpoint_resets_tracker_and_broadcasts_without_restarting_camera(tmp_path):
    spec = make_spec()
    bus = EventBus()
    writer = StructuredLogWriter(str(tmp_path / "steps.jsonl"))
    state = RuntimeState(str(tmp_path / "steps.jsonl"), len(spec.steps), tuple(step.name for step in spec.steps))
    for event_type in ("interaction", "alert", "speech", "experiment_reset"):
        bus.subscribe(event_type, state.record_event)
    tracker = SimpleStepTracker(spec, bus, writer, alert_cooldown=0)
    extractor = InteractionExtractor(bus)
    source = SimpleNamespace(restart_count=0)

    initial_touch = Event(event_type="interaction", timestamp=1.0,
                          payload={"subject": "hand", "verb": "touches", "object": "red_box"})
    bus.publish(initial_touch)
    state.record_step(writer.records[0])
    frame = np.zeros((30, 40, 3), dtype=np.uint8)
    state.update_pipeline(
        frame=frame, frame_id=44, detections=(), hands=(), step_index=tracker.current_step_idx,
        completed_step_ids=frozenset(tracker.completed_steps),
        active_predicates=tuple(sorted(tracker.active_predicates)), fps=10.0,
        health={"camera": ("ONLINE", ""), "detector": ("RUNNING", ""),
                "hands": ("RUNNING", ""), "tracker": ("RUNNING", ""), "tts": ("READY", "")},
    )
    state.record_event(Event(event_type="interaction", timestamp=1.1,
                             payload={"subject": "hand", "verb": "touches", "object": "yellow_box"}))
    state.record_event(Event(event_type="alert", timestamp=1.2,
                             payload={"message": "Old warning", "level": "warning", "timestamp": time.time()}))

    app = HARApplication.__new__(HARApplication)
    app.spec = spec
    app.bus = bus
    app.runtime = state
    app.tracker = tracker
    app.interaction_extractor = extractor
    app._commands = queue.Queue()
    app._started = True
    app._last_video_timestamp = 12.5
    app.pipeline_error = ""
    app.log_path = writer.filepath
    app.log_writer = writer
    app.streamer = SimpleNamespace(port=8081)
    app.source = source
    stop_worker = threading.Event()

    def consume_pipeline_commands():
        while not stop_worker.is_set():
            app._apply_pending_commands()
            time.sleep(0.002)

    app.pipeline_thread = threading.Thread(target=consume_pipeline_commands, daemon=True)
    app.pipeline_thread.start()
    client = TestClient(create_app(app))
    try:
        with client.websocket_connect("/ws") as websocket:
            first = websocket.receive_json()["data"]
            assert first["progress"]["current_index"] == 1

            response = client.post("/api/experiment/reset")
            assert response.status_code == 200
            reset = response.json()
            assert reset["progress"]["current_index"] == 0
            assert [step["status"] for step in reset["progress"]["steps"]] == ["current", "pending"]
            assert reset["action"] == ""
            assert reset["alert"] is None
            assert reset["assistant"]["message"] == spec.steps[0].instruction
            assert reset["camera"]["frame_id"] == 44
            assert reset["health"]["camera"] == "ONLINE"
            assert any(item["event"] == "Experiment reset" for item in reset["activity"])

            pushed = websocket.receive_json()["data"]
            assert pushed["progress"]["current_index"] == 0
            assert pushed["assistant"]["message"] == spec.steps[0].instruction
    finally:
        stop_worker.set()
        app.pipeline_thread.join(timeout=1)

    assert source.restart_count == 0
    assert tracker.current_step_idx == 0
    assert tracker.completed_steps == set()
    assert tracker.active_predicates == set()
