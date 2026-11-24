import pytest
from nmind.evaluation.phase2_training_protocol import epoch_order,staircase_plan,validate_param_match
def test_epoch_is_shuffled_without_replacement():
    x=epoch_order(100,7); assert len(x)==100 and len(set(x))==100
def test_parameter_hard_gate():
    assert validate_param_match(100,100)
    with pytest.raises(ValueError): validate_param_match(106,100)
def test_staircase_records_seen_samples_and_epochs():
    p=staircase_plan(); assert p[-1]['seen_samples']==16000 and p[-1]['cumulative_steps']==1672
