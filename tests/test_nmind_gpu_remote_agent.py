"""Guardrails for the N-MIND pull-only Windows GPU agent."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "scripts" / "nmind_gpu_remote_agent.py"
spec = importlib.util.spec_from_file_location("gpu_agent", path)
agent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agent)


def job(**changes):
    base = {
        "schema": "nmind-remote-gpu/1", "enabled": True, "action": "train",
        "request_id": "gpu-test-01", "source_revision": "a" * 40,
        "steps": 256, "arms": ["legacy", "router_ste"],
        "seed": 20261123, "batch_size": 4, "eval_samples": 40,
        "max_seconds_per_arm": 900, "min_vram_gb": 8,
        "requires_gpu": True}
    base.update(changes)
    return base


def test_disabled_remote_job_is_inert():
    assert agent.validate_request(None) is None
    assert agent.validate_request(job(enabled=False)) is None


def test_only_fixed_cuda_training_command_is_exposed(tmp_path):
    command = agent.training_command(agent.validate_request(job()), ROOT, tmp_path)
    assert "--device" in command and command[command.index("--device") + 1] == "cuda"
    assert "run_ablation.py" in command[1]
    assert "--save-candidates" in command
    assert "--final-sealed" not in command


@pytest.mark.parametrize("changes", [
    {"request_id": "../escape"},
    {"source_revision": "master"},
    {"steps": 2049}, {"steps": True},
    {"arms": ["legacy", "legacy"]}, {"arms": ["unregistered"]},
    {"seed": -1}, {"batch_size": 0},
    {"requires_gpu": False}, {"min_vram_gb": float("nan")},
    {"max_seconds_per_arm": 3601},
    {"command": "powershell malicious code"},
])
def test_rejects_untrusted_or_unbudgeted_job(changes):
    with pytest.raises(ValueError):
        agent.validate_request(job(**changes))


def test_cancel_only_validates_target_id():
    cancel = {"schema": "nmind-remote-gpu/1", "enabled": True,
              "action": "cancel", "request_id": "cancel-001",
              "target_request_id": "gpu-test-01"}
    assert agent.validate_request(cancel)["action"] == "cancel"
    with pytest.raises(ValueError):
        agent.validate_request({**cancel, "target_request_id": "../etc"})


def test_checkout_must_match_pinned_revision(monkeypatch, tmp_path):
    class Result:
        stdout = "b" * 40
        returncode = 0
    monkeypatch.setattr(agent.subprocess, "run", lambda *a, **k: Result())
    with pytest.raises(RuntimeError, match="differs"):
        agent.check_checkout(tmp_path, "a" * 40)


def test_agent_may_not_train_on_linux(monkeypatch):
    monkeypatch.setattr(agent.platform, "system", lambda: "Linux")
    with pytest.raises(RuntimeError, match="Windows"):
        agent.cuda_preflight(8)
