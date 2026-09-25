import os
import json
from src.har_space.data_models import StepRecord, StepStatus
from src.har_space.logging_.writer import StructuredLogWriter

def test_log_writer(tmp_path):
    log_file = tmp_path / "steps.jsonl"
    summary_file = tmp_path / "summary.json"
    
    writer = StructuredLogWriter(str(log_file))
    
    record = StepRecord(
        iso_timestamp="2026-09-23T12:00:00Z",
        video_timestamp=10.5,
        step_id="step_1",
        step_name="First Step",
        status=StepStatus.completed,
        outcome_text="done",
        confidence=0.95
    )
    
    writer.write_record(record)
    assert os.path.exists(log_file)
    
    # Read back jsonl
    with open(log_file, "r") as f:
        lines = f.readlines()
        assert len(lines) == 1
        loaded_record = json.loads(lines[0])
        assert loaded_record["step_id"] == "step_1"
        assert loaded_record["status"] == "completed"

    writer.write_summary(str(summary_file))
    assert os.path.exists(summary_file)
    
    with open(summary_file, "r") as f:
        summary = json.load(f)
        assert summary["total_steps_recorded"] == 1
        assert summary["completed_steps"] == 1
        assert summary["final_status"] == "success"
