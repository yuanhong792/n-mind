"""M41F 训练入口守卫与状态分类审计（零梯度）。"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import preflight_guard as guard
from train_m4 import classify_status

HERE = Path(__file__).resolve().parent


def main():
    report = json.loads((HERE / "M41F_PREFLIGHT_REPORT.json").read_text(encoding="utf-8"))
    positive = json.loads((HERE / "M4_POSITIVE_CONTROL.json").read_text(encoding="utf-8"))
    budget = {"steps_used": 0, "max_steps": 4096}
    results_done = {"models": {f"main_x{i}": {"status": "DONE"} for i in range(8)}}
    results_part = {"models": {f"main_x{i}": {"status": "DONE"} for i in range(3)}}

    stale_code = copy.deepcopy(report)
    stale_code["manifests"]["code"]["combined_sha256"] = "0" * 64
    stale_data = copy.deepcopy(report)
    stale_data["manifests"]["data"]["combined_sha256"] = "1" * 64

    missing_positive = copy.deepcopy(positive)
    missing_positive["gates"].pop("positive_R2_tiny_transformer_s1", None)
    partial_positive = copy.deepcopy(positive)
    partial_positive["status"] = "PARTIALLY_COMPLETED"

    checks = {
        "CURRENT_PREFLIGHT_VALID": guard.verify_preflight()["pass"],
        "STALE_CODE_REJECTED": not guard.verify_preflight(stale_code)["pass"],
        "STALE_DATA_REJECTED": not guard.verify_preflight(stale_data)["pass"],
        "MISSING_REFERENCE_MODEL_REJECTED": not guard.verify_positive_control(
            True, missing_positive)["pass"],
        "PARTIAL_POSITIVE_REPORT_REJECTED": not guard.verify_positive_control(
            True, partial_positive)["pass"],
        "MAIN_GATE_REJECTED_WITHOUT_POSITIVE": not guard.verify_training_entry(True)["pass"],
        "STATUS_COMPLETED": classify_status(
            results_done, budget, True, "main", True) == "COMPLETED",
        "STATUS_PARTIALLY_COMPLETED": classify_status(
            results_part, budget, True, "main", True) == "PARTIALLY_COMPLETED",
        "STATUS_POSITIVE_FAILED": classify_status(
            {}, budget, True, "all", False) == "POSITIVE_CONTROL_FAILED",
        "STATUS_OVER_BUDGET": classify_status(
            results_part, {"steps_used": 4096, "max_steps": 4096}, True, "main", True)
        == "OVER_BUDGET",
        "STATUS_ABORTED_EXCEPTION": classify_status(
            {}, budget, True, "main", True, exception=RuntimeError("x")) == "ABORTED_EXCEPTION",
    }
    out = {
        "status": "PASS" if all(checks.values()) else "FAILED",
        "checks": checks,
        "detail": {
            "current_preflight": guard.verify_preflight(),
            "current_entry_without_positive": guard.verify_training_entry(True),
            "positive_report_status": positive.get("status"),
        },
        "note": "当前未授权/未训练，主训练门禁应拒绝；授权后正控制通过才会放行。",
    }
    p = HERE / "M41F_GUARD_AUDIT.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[m41f-guard] status={out['status']} 写出 {p}")
    return out


if __name__ == "__main__":
    main()
