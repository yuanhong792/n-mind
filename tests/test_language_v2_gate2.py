import torch
from nmind.language.gate2_dataset import build_gate2_dataset, manifest
from nmind.language.gate2_preflight import run_preflight
from nmind.language.gate2_trainer import train_arm, balanced_rows

def test_gate2_dataset_counts_and_sealed_pairs():
    data=build_gate2_dataset(); mf=manifest(data)
    assert mf["counts"]["train"] == 768 and mf["counts"]["dev"] == 144 and mf["counts"]["sealed"] == 240
    assert mf["sealed_pair_count"] == 80
    assert not ({x.item_id for x in data["train"]} & {x.item_id for x in data["sealed"]})

def test_gate2_preflight_passes_offline():
    result=run_preflight(".")
    assert result["status"] == "PASS" and result["network_enabled"] is False and result["optimizer_steps"] == 0

def test_lr0_arm_preserves_parameters(tmp_path):
    result=train_arm(build_gate2_dataset()["train"][:4],tmp_path/"lr0",steps=2,seed=9,lr=0.0)
    assert result["optimizer_steps_executed"] == 2 and result["parameters_changed"] is False

def test_training_rows_cover_l1_and_answer_mask(tmp_path):
    rows=balanced_rows(build_gate2_dataset()["train"])
    assert {x.level for x in rows[:4]} == {"L0", "L1"}
    result=train_arm(build_gate2_dataset()["train"],tmp_path/"probe",steps=2,seed=3,lr=0.0)
    assert result["loss_mask"] == "answer_span_plus_eos" and set(result["levels_used"]) == {"L0", "L1"}
