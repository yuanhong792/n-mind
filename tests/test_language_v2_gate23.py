from pathlib import Path
from nmind.language.gate22_preflight import answer_mask_audit
from nmind.language.gate2_dataset import build_gate2_dataset

def test_gate23_mask_has_no_prompt_targets():
    item=build_gate2_dataset()["train"][257]
    audit=answer_mask_audit(item)
    assert audit["valid"] and audit["masked_answer_targets"] >= 2

def test_gate23_diagnostics_are_separate_from_gate22():
    assert Path("experiments/language_v2/gate23").exists()
    assert not Path("experiments/language_v2/gate23/runs").exists()
