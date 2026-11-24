"""M41 共享预算与唯一训练入口强制审计（零梯度/零 optimizer.step）。

检查：
- 唯一 optimizer.step 调用点、步数与墙钟预算、reserve-before-step；
- 未确认 reserve 跨重启仍保守计入；
- 正控制 256 / 主训练 3840 / 总 4096 的硬上限；
- 当前未授权状态下 optimizer.step = 0。

输出 M41_BUDGET_AUDIT.json；不修改真实账本。
"""

from __future__ import annotations

import ast
import json
import tempfile
from pathlib import Path

from budget_manager import BudgetManager, STAGE_MAIN, STAGE_POSITIVE

HERE = Path(__file__).resolve().parent


def optimizer_step_sites():
    sites = []
    for p in sorted(HERE.rglob("*.py")):
        if "__pycache__" in p.parts:
            continue
        tree = ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "step"):
                try:
                    recv = ast.unparse(node.func.value)
                except Exception:
                    recv = "?"
                sites.append({"file": str(p.relative_to(HERE)).replace("\\", "/"),
                              "line": node.lineno, "receiver": recv})
    return sites


def audit():
    checks = {}

    live = BudgetManager()
    live_summary = live.summary()
    training = json.loads((HERE / "M4_TRAINING_RESULTS.json").read_text(encoding="utf-8"))
    plan = json.loads((HERE / "M4_PREREGISTERED_PLAN.json").read_text(encoding="utf-8"))
    zero = (live_summary["steps_used"] == 0
            and live_summary["pending_unconfirmed"] == 0
            and int(training.get("optimizer_steps_executed", -1)) == 0
            and "AWAITING_AUTHORIZATION" in plan.get("status", "")
            and plan["authorization"]["status"] == "NOT_YET_GRANTED")
    checks["UNAUTHORIZED_ZERO_STEP"] = {
        "pass": zero,
        "observed": {"ledger_steps": live_summary["steps_used"],
                     "pending": live_summary["pending_unconfirmed"],
                     "training_result_steps": training.get("optimizer_steps_executed"),
                     "plan_status": plan.get("status")},
        "required": {"ledger_steps": 0, "training_result_steps": 0,
                     "plan_status_contains": "AWAITING_AUTHORIZATION"},
    }

    sites = optimizer_step_sites()
    sole = (len(sites) == 1 and sites[0]["file"] == "train_m4.py")
    checks["SOLE_TRAINING_ENTRY"] = {
        "pass": sole, "observed": sites,
        "required": "唯一 optimizer.step 位于 train_m4.py",
    }

    with tempfile.TemporaryDirectory() as td:
        ledger = Path(td) / "budget.jsonl"
        bm = BudgetManager(ledger=ledger)
        rid = bm.reserve_step("probe", "R1_gru_reference", 1, 0, "positive",
                              {"B": 16, "T": 17}, 12345)
        events = [json.loads(l) for l in ledger.read_text(encoding="utf-8").splitlines()]
        reserve_first = events[0]["type"] == "reserve" and events[0]["rid"] == rid
        bm.confirm_step(rid, loss=0.5, macs=12345)
        events = [json.loads(l) for l in ledger.read_text(encoding="utf-8").splitlines()]
        confirmed = events[1]["type"] == "step" and events[1]["rid"] == rid
        s1 = bm.summary()
        reserve_before = reserve_first and confirmed and s1["steps_used"] == 1 \
            and s1["pending_unconfirmed"] == 0
        checks["RESERVE_BEFORE_CONFIRM"] = {
            "pass": reserve_before,
            "observed": {"event_types": [e["type"] for e in events[:2]],
                         "steps_used": s1["steps_used"],
                         "pending": s1["pending_unconfirmed"]},
            "required": {"event_types": ["reserve", "step"], "steps_used": 1, "pending": 0},
        }

        rid2 = bm.reserve_step("probe2", "R2", 1, 1, "positive", {"B": 16, "T": 17}, 99)
        bm2 = BudgetManager(ledger=ledger)
        s2 = bm2.summary()
        conservative = s2["steps_used"] == 2 and s2["pending_unconfirmed"] == 1 \
            and rid2 in bm2.pending
        checks["UNCONFIRMED_CONSERVATIVE"] = {
            "pass": conservative,
            "observed": {"steps_used_after_restart": s2["steps_used"],
                         "pending_after_restart": s2["pending_unconfirmed"],
                         "rid": rid2},
            "required": {"steps_used_after_restart": 2, "pending_after_restart": 1},
        }

        ledger2 = Path(td) / "caps.jsonl"
        bm3 = BudgetManager(ledger=ledger2)
        for i in range(STAGE_POSITIVE):
            bm3.reserve_step("cap_pos", "R1", 1, i, "positive")
        pos_reject = False
        try:
            bm3.reserve_step("cap_pos", "R1", 1, STAGE_POSITIVE, "positive")
        except RuntimeError as exc:
            pos_reject = "positive_stage_budget" in str(exc)
        bm3.reserve_step("cap_main", "R1", 1, 0, "main")
        for i in range(1, STAGE_MAIN):
            bm3.reserve_step("cap_main", "R1", 1, i, "main")
        main_reject = False
        try:
            bm3.reserve_step("cap_main", "R1", 1, STAGE_MAIN, "main")
        except RuntimeError as exc:
            main_reject = "step_budget" in str(exc) or "main_stage_budget" in str(exc)
        s3 = bm3.summary()
        caps = (pos_reject and main_reject and s3["steps_used"] == 4096
                and s3["stages"].get("positive") == 256
                and s3["stages"].get("main") == 3840)
        checks["SHARED_BUDGET_CAPS"] = {
            "pass": caps,
            "observed": {"steps_used": s3["steps_used"], "stages": s3["stages"],
                         "positive_rejected": pos_reject, "main_rejected": main_reject},
            "required": {"positive": 256, "main": 3840, "total": 4096,
                         "max_wall_seconds": 3600},
        }

    status = "PASS" if all(v["pass"] for v in checks.values()) else "FAILED"
    out = {"status": status, "checks": checks,
           "conclusion": ("共享预算、唯一入口与零步状态通过审计"
                          if status == "PASS" else "预算/入口审计失败，禁止授权训练"),
           "note": "审计使用临时账本，不修改真实 M4_BUDGET_LEDGER.jsonl。"}
    p = HERE / "M41F_BUDGET_AUDIT.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[m41-budget] status={status} 写出 {p}")
    return out


if __name__ == "__main__":
    audit()
