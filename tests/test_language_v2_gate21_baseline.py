from nmind.language.gate2_dataset import build_gate2_dataset
from nmind.language.gate21_baseline import zero_step_gate2_baseline

def test_zero_step_gate2_baseline_is_neural_only():
    result=zero_step_gate2_baseline(build_gate2_dataset()["sealed"][:4])
    assert result["optimizer_steps"] == 0 and result["response_source"].startswith("neural_only")
