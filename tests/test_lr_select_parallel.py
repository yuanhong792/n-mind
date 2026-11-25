"""阶段 2.4 步骤 3.3：全局上限触发的真实模拟测试（替换 20261124 的 5 行存根）。

要求（本次指令步骤 3.3）：模拟全局上限触发，断言已完成任务的结果文件仍然存在且完整。
实现：用 lr_select_parallel_v2.run_selection 的同一执行器，工作进程换成假 worker
（快任务写完结果文件即退出；慢任务睡 30s 等待被 terminate），cap_s=2s 触发上限。
同时覆盖选择规则：3 个 LR 齐且 complete 才可选；缺任一 → INCONCLUSIVE_BUDGET；
并列时 dev_loss 更低者胜。
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V2 = ROOT / "experiments" / "phase2_baselines" / "lr_select_parallel_v2.py"
_spec = importlib.util.spec_from_file_location("lr_select_v2", V2)
lr_v2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lr_v2)

FAKE_WORKER = """\
import argparse, json, os, time
from pathlib import Path
ap = argparse.ArgumentParser()
ap.add_argument("--worker", action="store_true")
ap.add_argument("--model"); ap.add_argument("--lr", type=float)
ap.add_argument("--seed", type=int); ap.add_argument("--out")
a = ap.parse_args()
out = Path(a.out)
time.sleep({"fast": 0.05, "slow": 30.0}[a.model])   # 先干活：慢任务在完成前就被终止
rec = {"model": a.model, "lr": a.lr, "seed": a.seed, "steps": 1000,
       "dev_rank_accuracy": 0.2, "dev_loss": 1.0 + a.lr, "wall_s": 0.05,
       "complete": True, "env": {"fake": True, "pid": os.getpid()}}
tmp = out / f".{a.model}_lr{a.lr}.json.tmp"
tmp.write_text(json.dumps(rec), encoding="utf8")
tmp.replace(out / f"{a.model}_lr{a.lr}.json")
"""


def _load():
    return lr_v2


def test_global_cap_keeps_completed_task_files_and_marks_timeout(tmp_path):
    mod = _load()
    worker = tmp_path / "fake_worker.py"
    worker.write_text(FAKE_WORKER, encoding="utf8")
    tasks = [("fast", 1e-4, 1), ("fast", 2e-4, 1), ("fast", 3e-4, 1),
              ("slow", 1e-4, 1), ("slow", 2e-4, 1), ("slow", 3e-4, 1)]
    summary = mod.run_selection(worker_script=str(worker), tasks=tasks,
                                 out_dir=tmp_path / "out", cap_s=2.0, poll_s=0.2)
    # 1) 上限确实触发，慢任务被终止
    assert summary["cap_triggered"] is True
    assert len(summary["killed_tasks"]) == 3
    killed = {tuple(k) for k in summary["killed_tasks"]}
    assert ("slow", 1e-4, 1) in killed
    # 2) 已完成任务的 JSON 存在且完整（含全部必需键）
    for lr in (1e-4, 2e-4, 3e-4):
        f = tmp_path / "out" / f"fast_lr{lr}.json"
        assert f.exists(), f"已完成任务结果文件缺失: {f}"
        d = json.loads(f.read_text(encoding="utf8"))
        for k in mod.REQUIRED_KEYS:
            assert k in d, f"结果文件缺键 {k}"
        assert d["complete"] is True and d["steps"] == 1000
    # 3) 被终止任务没有留下冒充完整的文件
    for lr in (1e-4, 2e-4, 3e-4):
        assert not (tmp_path / "out" / f"slow_lr{lr}.json").exists()
    # 4) 选择规则：fast 3/3 → SELECTED；slow 0/3 → INCONCLUSIVE_BUDGET
    assert summary["selection"]["fast"]["status"] == "SELECTED"
    assert summary["selection"]["slow"]["status"] == "INCONCLUSIVE_BUDGET"
    # 5) summary.json 已落盘（上限触发也必须写）
    assert json.loads((tmp_path / "out" / "summary.json").read_text(encoding="utf8"))[
        "cap_triggered"] is True


def test_selection_rule_needs_all_three_lr_files(tmp_path):
    mod = _load()
    out = tmp_path / "out"
    out.mkdir()
    base = {"steps": 1000, "complete": True, "seed": 1, "env": {}}
    # gru 只有 2/3 个 LR 文件 → INCONCLUSIVE_BUDGET
    for lr, acc in ((1e-4, 0.1), (2e-4, 0.3)):
        (out / f"gru_lr{lr}.json").write_text(json.dumps(
            {**base, "model": "gru", "lr": lr, "dev_rank_accuracy": acc,
             "dev_loss": 1.0, "wall_s": 1.0}), encoding="utf8")
    # transformer 3/3 且 rank 并列 → 选 dev_loss 更低者
    for lr, loss in ((1e-4, 0.9), (2e-4, 0.5), (3e-4, 0.7)):
        (out / f"transformer_lr{lr}.json").write_text(json.dumps(
            {**base, "model": "transformer", "lr": lr, "dev_rank_accuracy": 0.25,
             "dev_loss": loss, "wall_s": 1.0}), encoding="utf8")
    # delta 3/3 但其中一个 complete=False → INCONCLUSIVE_BUDGET
    for lr, comp in ((1e-4, True), (2e-4, False), (3e-4, True)):
        (out / f"delta_lr{lr}.json").write_text(json.dumps(
            {**base, "model": "delta", "lr": lr, "dev_rank_accuracy": 0.2,
             "dev_loss": 1.0, "wall_s": 1.0, "complete": comp}), encoding="utf8")
    sel = mod.select_from_files(out, models=("gru", "transformer", "delta"),
                                 lrs=(1e-4, 2e-4, 3e-4))
    assert sel["gru"]["status"] == "INCONCLUSIVE_BUDGET"
    assert sel["delta"]["status"] == "INCONCLUSIVE_BUDGET"
    assert sel["transformer"]["status"] == "SELECTED"
    assert sel["transformer"]["lr"] == 2e-4          # 并列取 dev_loss 更低
