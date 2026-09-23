import numpy as np
from pydantic import BaseModel, ConfigDict

class Frame(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    frame_id: int
    wall_clock_timestamp: float
    video_timestamp: float
    image: np.ndarray
    source_name: str
