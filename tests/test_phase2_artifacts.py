import pytest
from nmind.evaluation.phase2_artifacts import validate_arm_artifact
def test_missing_artifacts_rejected():
    with pytest.raises(ValueError): validate_arm_artifact({'timing_only':False})
def test_timing_artifact_is_explicit():
    validate_arm_artifact({'checkpoint':None,'sample_logits':None,'metadata':{},'timing_only':True})
