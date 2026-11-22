from nmind.language.gate2_diagnose import data_coverage, state_equivalence

def test_gate21_exposes_l1_truncation_and_missing_answer_mask():
    result=data_coverage()
    assert result["trainer_rows_used"] == 256
    assert result["l1_training_rows_used"] == 0
    assert result["answer_boundary_supervision"] == "none"

def test_gate21_state_chunk_protocol_is_zero_step():
    result=state_equivalence()
    assert result["optimizer_steps"] == 0
    assert result["max_abs_logit_diff_last_token"] < 1e-4
