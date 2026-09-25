from typing import Tuple
from pydantic import BaseModel

class SourceConfig(BaseModel):
    type: str  # "webcam", "file", "rtsp"
    path_or_index: str

class RecordingConfig(BaseModel):
    enabled: bool
    directory: str
    segment_length_minutes: float

class StreamingConfig(BaseModel):
    enabled: bool
    mode: str  # "mjpeg" or "ffmpeg"
    host: str
    port: int
    protocol: str  # e.g., "tcp", "udp"

class RuntimeConfig(BaseModel):
    source: SourceConfig
    target_resolution: Tuple[int, int]
    target_fps: float
    recording: RecordingConfig
    streaming: StreamingConfig

    @classmethod
    def load_from_yaml(cls, filepath: str) -> "RuntimeConfig":
        import yaml
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)
