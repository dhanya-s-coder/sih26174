from enum import Enum
from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field

# Shared Data Models

class Detection(BaseModel):
    label: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]

class Pose(BaseModel):
    keypoints: List[List[float]]  # [[x, y, z, conf], ...]

class HandState(BaseModel):
    hand_type: str  # "left" or "right"
    bbox: List[float] # Extents [x1, y1, x2, y2]
    landmarks: List[List[float]] # [[x, y, z], ...]
    confidence: float
    state: str = "unknown"

class Interaction(BaseModel):
    subject: str  # e.g., "hand"
    verb: str  # e.g., "touches", "holds"
    object: str  # e.g., "red_box"

class Event(BaseModel):
    event_type: str
    timestamp: float
    payload: Dict[str, Any]

class StepStatus(str, Enum):
    pending = "pending"
    in_progress = "in_progress"
    completed = "completed"
    skipped = "skipped"
    out_of_sequence = "out_of_sequence"
    failed = "failed"

class StepRecord(BaseModel):
    iso_timestamp: str
    video_timestamp: float
    step_id: str
    step_name: str
    status: StepStatus
    outcome_text: str
    confidence: float

class Alert(BaseModel):
    message: str
    level: str  # e.g., "info", "warning", "critical"
    timestamp: float
