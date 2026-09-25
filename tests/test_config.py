import pytest
from pydantic import ValidationError
import os
import yaml
from src.har_space.config.models import ExperimentSpec

def test_load_valid_spec(tmp_path):
    # Create a valid spec file
    valid_data = {
        "name": "Test Experiment",
        "version": "1.0",
        "steps": [
            {
                "id": "step_1",
                "name": "First Step",
                "instruction": "Do it",
                "position": 1,
                "required_predicates": [{"type": "touches", "args": ["box"]}],
                "expected_outcome": "done"
            }
        ]
    }
    file_path = tmp_path / "valid.yaml"
    with open(file_path, "w") as f:
        yaml.dump(valid_data, f)
        
    spec = ExperimentSpec.load_from_yaml(str(file_path))
    assert spec.name == "Test Experiment"
    assert len(spec.steps) == 1
    assert spec.steps[0].id == "step_1"

def test_load_invalid_spec(tmp_path):
    # Missing required field "name"
    invalid_data = {
        "version": "1.0",
        "steps": []
    }
    file_path = tmp_path / "invalid.yaml"
    with open(file_path, "w") as f:
        yaml.dump(invalid_data, f)
        
    with pytest.raises(ValidationError):
        ExperimentSpec.load_from_yaml(str(file_path))

def test_load_real_sample_yaml():
    spec_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'experiment_sample.yaml')
    spec = ExperimentSpec.load_from_yaml(spec_path)
    assert len(spec.steps) == 3
    assert spec.steps[0].id == "step_001"
    assert spec.steps[1].id == "step_002"
    assert spec.steps[2].id == "step_003"
