import json
import os
from typing import List
from src.har_space.data_models import StepRecord, StepStatus

class StructuredLogWriter:
    """Writes timestamped, structured JSON Lines of conducted steps."""
    def __init__(self, filepath: str):
        self.filepath = filepath
        os.makedirs(os.path.dirname(self.filepath) or '.', exist_ok=True)
        self.records: List[StepRecord] = []
        # Clear existing file or start fresh
        with open(self.filepath, 'w') as f:
            pass
        
    def write_record(self, record: StepRecord) -> None:
        """Appends a single StepRecord to the log file."""
        self.records.append(record)
        with open(self.filepath, 'a') as f:
            f.write(record.model_dump_json() + '\n')
            
    def write_summary(self, summary_filepath: str) -> None:
        """Generates a compact end-of-run summary file."""
        completed = sum(1 for r in self.records if r.status == StepStatus.completed)
        total = len(self.records)
        summary = {
            "total_steps_recorded": total,
            "completed_steps": completed,
            "final_status": "success" if completed == total and total > 0 else "incomplete"
        }
        
        os.makedirs(os.path.dirname(summary_filepath) or '.', exist_ok=True)
        with open(summary_filepath, 'w') as f:
            json.dump(summary, f, indent=2)
