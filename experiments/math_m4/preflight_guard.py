"""M41F 唯一训练入口的强制预检守卫（零梯度）。

任何真实 optimizer.step 前必须通过：
- 最新最终预检报告存在且状态为 PASS；
- 当前代码哈希、数据哈希、预登记方案哈希与预检记录完全一致（拒绝过期预检）；
- 预登记预算仍为 4096 step / 3600s；
- 共享预算台账未超限；
- 正控制阶段（main）必须有两个参考模型完整且通过的门禁记录。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

CODE_FILES = [
    "train_m4.py",
    "preflight_guard.py",
    "evaluate_m4.py",
    "compute_accounting_m4.py",
    "budget_manager.py",
    "positive_control.py",
    "dataset_m4.py",
    "reference_interpreter.py",
    "tokenizer_m4.py",
    "models/gru_reference.py",
    "models/transformer_reference.py",
    "models/low_rank_interaction.py",
    "models/selective_interaction.py",
    "models/base.py",
]
DATA_FILES = [
    "m4_train.jsonl", "m4_val.jsonl", "m4_sealed.jsonl",
    "m4_counterfactual_val.jsonl", "m4_counterfactual_sealed.jsonl",
    "m4_distance_val.jsonl", "m4_distance_sealed.jsonl",
    "m4_generalization.jsonl",
]
PLAN = "M4_PREREGISTERED_PLAN.json"
REPORT = "M41F_PREFLIGHT_REPORT.json"
POSITIVE = "M4_POSITIVE_CONTROL.json"
REQUIRED_POSITIVE = {
    "positive_R1_gru_reference_s1",
    "positive_R2_tiny_transformer_s1",
}


def _manifest(paths):
    items, h = {}, hashlib.sha256()
    for rel in paths:
        p = HERE / rel
        if not p.exists():
            raise FileNotFoundError(p)
        blob = p.read_bytes()
        sha = hashlib.sha256(blob).hexdigest()
        items[rel] = {"sha256": sha, "bytes": len(blob)}
        h.update(rel.encode("utf-8") + b"\0" + sha.encode("ascii") + b"\n")
    return {"combined_sha256": h.hexdigest(), "files": items}


def code_manifest():
    return _manifest(CODE_FILES)


def data_manifest():
    return _manifest([f"data/{f}" for f in DATA_FILES])


def plan_manifest():
    return _manifest([PLAN])


def _load(path):
    return json.loads((HERE / path).read_text(encoding="utf-8"))


def verify_plan():
    plan = _load(PLAN)
    budget = plan.get("budget", {})
    checks = {
        "status_awaiting": "AWAITING_AUTHORIZATION" in plan.get("status", ""),
        "auth_not_granted": plan.get("authorization", {}).get("status") == "NOT_YET_GRANTED",
        "max_steps_4096": budget.get("max_optimizer_steps") == 4096,
        "max_wall_3600": budget.get("max_wall_seconds") == 3600,
        "positive_256": budget.get("breakdown", {}).get(
            "positive_control", {}).get("total") == 256,
        "main_3840": budget.get("breakdown", {}).get("main", {}).get("total") == 3840,
    }
    return {"pass": all(checks.values()), "checks": checks,
            "plan": plan.get("status"), "authorization": plan.get("authorization")}


def verify_budget():
    from budget_manager import BudgetManager
    s = BudgetManager().summary()
    checks = {
        "steps_within_limit": s["steps_used"] <= s["max_steps"],
        "wall_within_limit": s["wall_used"] <= s["max_wall"],
        "not_over_total": s["steps_used"] <= 4096,
    }
    return {"pass": all(checks.values()), "checks": checks, "summary": s}


def load_preflight():
    p = HERE / REPORT
    if not p.exists():
        return {"pass": False, "reason": f"缺少最新预检报告 {REPORT}", "report": None}
    report = _load(REPORT)
    return {"pass": True, "report": report}


def verify_preflight(report_override=None):
    if report_override is None:
        loaded = load_preflight()
        if not loaded["pass"]:
            return loaded
        report = loaded["report"]
    else:
        report = report_override
    cur_code = code_manifest()
    cur_data = data_manifest()
    cur_plan = plan_manifest()
    rec = report.get("manifests", {})
    checks = {
        "status_pass": report.get("status") == "PREFLIGHT_PASS_AWAITING_AUTHORIZATION",
        "code_current": rec.get("code", {}).get("combined_sha256") == cur_code["combined_sha256"],
        "data_current": rec.get("data", {}).get("combined_sha256") == cur_data["combined_sha256"],
        "plan_current": rec.get("plan", {}).get("combined_sha256") == cur_plan["combined_sha256"],
        "gates_all_pass": all(report.get("gates", {}).values())
        and len(report.get("gates", {})) == 10,
    }
    return {"pass": all(checks.values()), "checks": checks,
            "current_code_sha256": cur_code["combined_sha256"],
            "current_data_sha256": cur_data["combined_sha256"],
            "recorded_code_sha256": rec.get("code", {}).get("combined_sha256"),
            "recorded_data_sha256": rec.get("data", {}).get("combined_sha256"),
            "report_status": report.get("status"),
            "reason": None if all(checks.values()) else "预检已过期或未通过"}


def verify_positive_control(require_pass, obj_override=None):
    if obj_override is None:
        p = HERE / POSITIVE
        if not p.exists():
            return {"pass": False, "reason": "缺少正控制报告"}
        obj = _load(POSITIVE)
    else:
        obj = obj_override
    gates = obj.get("gates", {})
    missing = sorted(REQUIRED_POSITIVE - set(gates))
    bad = sorted(k for k, v in gates.items() if not v.get("gate", {}).get("pass", False))
    complete = obj.get("status") == "COMPLETE" and not missing
    passed = complete and not bad
    checks = {
        "both_reference_models_present": not missing,
        "all_gates_pass": not bad,
        "report_complete": obj.get("status") == "COMPLETE",
    }
    return {"pass": passed if require_pass else not missing,
            "checks": checks, "missing": missing, "failed": bad,
            "status": obj.get("status"),
            "reason": None if (passed or not require_pass) else
            "正控制未完整通过，主训练门禁拒绝"}


def verify_training_entry(require_positive_pass=False):
    code = code_manifest()
    data = data_manifest()
    plan = plan_manifest()
    out = {
        "preflight": verify_preflight(),
        "plan": verify_plan(),
        "budget": verify_budget(),
        "positive_control": verify_positive_control(require_positive_pass),
        "manifests": {"code": code, "data": data, "plan": plan},
        "require_positive_pass": require_positive_pass,
    }
    out["pass"] = all(out[k]["pass"] for k in
                      ("preflight", "plan", "budget", "positive_control"))
    return out


def assert_training_entry(require_positive_pass=False):
    res = verify_training_entry(require_positive_pass)
    if not res["pass"]:
        failed = [k for k in ("preflight", "plan", "budget", "positive_control")
                  if not res[k]["pass"]]
        raise RuntimeError(f"训练入口预检拒绝：{failed}；"
                           f"preflight_reason={res['preflight'].get('reason')}；"
                           f"positive_reason={res['positive_control'].get('reason')}")
    return res
