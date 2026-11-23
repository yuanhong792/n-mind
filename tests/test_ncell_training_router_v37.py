"""Resource-aware training routing: GitHub CPU by default, opt-in Windows CUDA."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from nmind.evaluation.ncell_ablation import build_dataset, build_tokenizer, run_experiment

# Route planning file is a script in experiments/, so load it without making
# experiments an importable package and without any side effects.
import importlib.util

PROJECT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT / "experiments" / "v37_ncell_architecture" / "route_training.py"
spec = importlib.util.spec_from_file_location("v37_route_training", MODULE_PATH)
assert spec is not None and spec.loader is not None
router = importlib.util.module_from_spec(spec)
spec.loader.exec_module(router)


ARMS = ["legacy", "router_ste", "attention", "router_attention", "reset", "all"]


def test_small_training_automatically_stays_on_free_cpu():
    x = router.decide_route(steps=16, arms=ARMS, estimated_ram_gb=2.0)
    assert x["route"] == "GITHUB_FREE_CPU"
    assert x["device"] == "cpu"
    assert x["total_steps_max"] == 96


def test_large_training_generates_local_manual_plan_without_execution(tmp_path):
    json_path, ps_path = router.write_plan(
        tmp_path, steps=256, arms=ARMS, estimated_ram_gb=4.0)
    manifest = json.loads(json_path.read_text(encoding="utf-8"))
    commands = ps_path.read_text(encoding="utf-8")
    assert manifest["route"] == "LOCAL_MANUAL"
    assert manifest["training_performed"] is False
    assert manifest["automatic_local_execution"] is False
    assert manifest["stable_checkpoint_modified"] is False
    assert "CLOUD_PER_ARM_STEP_POLICY" in manifest["reasons"]
    assert "--steps 256" in commands
    assert "--device cuda" in commands
    assert "--max-steps-per-arm 2048" in commands
    assert "torch.cuda.is_available()" in commands


def test_gpu_and_estimated_ram_can_force_local():
    assert router.decide_route(steps=16, arms=ARMS,
                                estimated_ram_gb=12)["route"] == "LOCAL_MANUAL"
    assert router.decide_route(steps=16, arms=["legacy"],
                                estimated_ram_gb=1, requires_gpu=True)["route"] == "LOCAL_MANUAL"
    assert router.decide_route(steps=16, arms=["legacy"],
                                estimated_ram_gb=1, force_local=True)["route"] == "LOCAL_MANUAL"


def test_cloud_route_plan_is_inert_and_no_gpu_command_executed(tmp_path):
    j, ps = router.write_plan(tmp_path, steps=8, arms=["legacy", "reset"])
    assert json.loads(j.read_text(encoding="utf-8"))["route"] == "GITHUB_FREE_CPU"
    script = ps.read_text(encoding="utf-8")
    assert script.startswith("# GITHUB_FREE_CPU")
    assert all(line.startswith("#") for line in script.splitlines())


@pytest.mark.parametrize("kw", [
    dict(steps=-1, arms=["legacy"], estimated_ram_gb=1),
    dict(steps=2049, arms=["legacy"], estimated_ram_gb=1),
    dict(steps=1, arms=[], estimated_ram_gb=1),
    dict(steps=1, arms=["legacy", "legacy"], estimated_ram_gb=1),
    dict(steps=1, arms=["legacy"], estimated_ram_gb=0),
])
def test_invalid_job_refused(kw):
    with pytest.raises(ValueError):
        router.decide_route(**kw)


def test_cloud_cpu_neural_experiment_rejects_unapproved_larger_steps():
    with pytest.raises(ValueError):
        run_experiment(steps=129, arms=("legacy",), eval_samples=2)
    with pytest.raises(ValueError):
        run_experiment(steps=1, arms=("legacy",), eval_samples=2, device="bogus")


def test_cuda_request_does_not_silently_fall_back(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        run_experiment(steps=1, arms=("legacy",), eval_samples=2, device="cuda")


def test_small_cpu_train_still_works_with_explicit_device():
    torch.set_num_threads(1)
    r = run_experiment(steps=1, batch_size=2, eval_samples=2,
                       arms=("legacy",), device="cpu",
                       wall_seconds_per_arm=40)
    assert r["config"]["device"] == "cpu"
    assert r["arms"]["legacy"]["training"]["optimizer_steps_executed"] == 1
