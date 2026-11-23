import pytest
from nmind.language.gate24_dataset import build_gate24, manifest, stratified_label_permutation, context_blind
from nmind.language.gate24_security import audit_controls
from nmind.language.gate24_training_gate import TrainingAuthRequired, require_gate24_authorization

def test_gate24_semantic_split_and_manifest():
    d=build_gate24(); m=manifest(d)
    assert m["semantic_group_disjoint"] and m["sealed_pairs"] >= 40

def test_gate24_counterfactual_oracle_answers_change():
    d=build_gate24(); cf=[x for x in d["sealed"] if x.kind=="counterfactual"]
    assert len(cf)==80 and all(cf[i].answer != cf[i+1].answer for i in range(0,len(cf),2))

def test_gate24_label_control_is_real_derangement():
    d=build_gate24(); _,meta=stratified_label_permutation(d["train"],seed=7)
    assert meta["fixed_points"]==0 and meta["changed_fraction"]==1.0

def test_gate24_context_blind_has_no_location_relation():
    d=build_gate24(); blind=context_blind(d["train"])
    assert all("一号位置" not in x.prompt and "二号位置" not in x.prompt and "三号位置" not in x.prompt and "四号位置" not in x.prompt for x in blind)

def test_gate24_training_requires_next_authorization():
    with pytest.raises(TrainingAuthRequired): require_gate24_authorization(0)

def test_gate24_control_audit_passes():
    assert audit_controls()["status"] == "PASS"
