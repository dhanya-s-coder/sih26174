from typing import List, Optional
from pydantic import BaseModel
import yaml

class Predicate(BaseModel):
    type: str
    args: List[str]

class ExperimentStep(BaseModel):
    id: str
    name: str
    instruction: str
    position: int
    required_predicates: List[Predicate]
    min_duration: Optional[float] = None
    timeout: Optional[float] = None
    expected_outcome: str

class ExperimentSpec(BaseModel):
    name: str
    version: str
    steps: List[ExperimentStep]

    @classmethod
    def load_from_yaml(cls, filepath: str) -> "ExperimentSpec":
        """Loads and validates an experiment spec from a YAML file."""
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)
