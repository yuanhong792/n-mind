from nmind.language.gate22_preflight import run_gate22_preflight

def test_gate22_preflight_confirms_conditions():
    r=run_gate22_preflight()
    assert r["status"] == "PASS"
    assert r["sampling"]["l0_rows"] == 256 and r["sampling"]["l1_rows"] == 256
    assert r["answer_mask"]["all_valid"] is True
    assert r["leakage"]["train_dev_sealed_disjoint"] is True
    assert r["negative_control"]["labels_permuted"] is True
