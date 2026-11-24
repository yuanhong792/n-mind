"""M41 训练前强制准入报告生成器（零梯度）。

汇总已实际生成的审计文件，逐项计算十项准入条件；任何失败都不会硬编码为 true。
输出 M41_PREFLIGHT_REPORT.json。
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from budget_manager import BudgetManager
from positive_control import alignment_report, control_rows, gate_spec
from tokenizer_m4 import M4Tokenizer

HERE = Path(__file__).resolve().parent


def load(name):
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def numeric(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def deep_summary(obj, max_failure_examples=10):
    diffs, failures, n_false = [], [], 0

    def walk(x):
        nonlocal n_false
        if isinstance(x, dict):
            for k, v in x.items():
                if k == "max_abs_diff" and numeric(v):
                    diffs.append(float(v))
                if k == "failures" and isinstance(v, list):
                    failures.extend(v[:max_failure_examples])
                if k == "pass" and v is False:
                    n_false += 1
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(obj)
    return {"max_abs_diff": round(max(diffs), 10) if diffs else None,
            "n_failed_assertions": n_false,
            "failure_examples": failures[:max_failure_examples]}


def main():
    causal = load("M41_CAUSALITY_AUDIT.json")
    kv = load("M41_KV_CACHE_AUDIT.json")
    cost = load("M41_COST_ACCOUNTING_AUDIT.json")
    budget = load("M41_BUDGET_AUDIT.json")
    metric = load("M41_METRIC_AUDIT.json")
    data = load("M41_DATA_AUDIT.json")
    repair = load("M41_DATA_REPAIR_AUDIT.json")
    live_budget = BudgetManager().summary()

    model_summaries = {}
    for name, m in causal.get("models", {}).items():
        s = deep_summary(m)
        s["status"] = m.get("status")
        s["causality_lengths"] = sorted(
            (int(k) for k in m.get("causality", {})), key=int)
        s["routing_gradient"] = m.get("routing_gradient")
        model_summaries[name] = s

    causality_all = (causal.get("status") == "PASS"
                     and all(m.get("status") == "PASS"
                             for m in causal.get("models", {}).values())
                     and all(s["n_failed_assertions"] == 0
                             for s in model_summaries.values()))

    incremental_all = causality_all and all(
        all(v.get("pass") for v in m.get("full_vs_incremental", {}).values())
        and all(v.get("pass") for v in m.get("chunked_prefill", {}).values())
        and all(v.get("pass") for v in m.get("prefill_then_generate", {}).values())
        and all(v.get("pass") for v in m.get("batch_consistency", {}).values())
        for m in causal.get("models", {}).values())

    padding_all = causality_all and all(
        all(v.get("pass") for v in m.get("padding_isolation", {}).values())
        for m in causal.get("models", {}).values())

    routing_all = causality_all and all(
        m.get("routing_gradient", {}).get("pass") is True
        for m in causal.get("models", {}).values())

    kv_all = (kv.get("status") == "PASS"
              and all(c.get("pass") for c in kv.get("cases", [])))

    cost_all = cost.get("status") == "PASS" and all(
        v.get("pass") for v in cost.get("hand_checks", {}).values()) and all(
        v.get("finite") for v in cost.get("profiler_cross_check", {}).values())

    metric_all = (metric.get("status") == "PASS"
                  and all(v.get("pass") for v in metric.get("checks", {}).values()))

    data_all = (data.get("status") == "PASS"
                and all(v.get("pass") for v in data.get("checks", {}).values()))

    budget_all = (budget.get("status") == "PASS"
                  and all(v.get("pass") for v in budget.get("checks", {}).values())
                  and live_budget["steps_used"] == 0
                  and live_budget["pending_unconfirmed"] == 0)

    tok = M4Tokenizer()
    rows = control_rows(tok)
    alignment = alignment_report(tok, rows)
    spec = gate_spec()
    gate_ready = (budget_all and alignment["answer_only_alignment"]
                  and len(rows) == 64
                  and spec["tf_accuracy_min"] == 0.95
                  and spec["natural_eos_full_accuracy_min"] == 0.90
                  and spec["sole_training_entry"] == "train_m4.py"
                  and spec["abort_main_on_reference_failure"] is True)

    gates = {
        "CAUSALITY_PASS": causality_all,
        "INCREMENTAL_PARITY_PASS": incremental_all,
        "KV_CACHE_PASS": kv_all,
        "PADDING_ISOLATION_PASS": padding_all,
        "ROUTING_GRADIENT_PASS": routing_all,
        "ACTUAL_COST_ACCOUNTING_PASS": cost_all,
        "STRICT_EOS_METRIC_PASS": metric_all,
        "DATA_LEAKAGE_AUDIT_PASS": data_all,
        "SHARED_BUDGET_PASS": budget_all,
        "POSITIVE_CONTROL_GATE_READY": gate_ready,
    }

    original = (HERE / "M4_MATH_TESTS.json").read_bytes()
    original_sha = hashlib.sha256(original).hexdigest()
    history = load("M41_BASELINE_SNAPSHOT.json")
    history_preserved = original_sha == history["sha256"] and \
        history["n_pass"] == 24 and history["n_tests"] == 24

    cost_samples = {}
    for name, c in cost.get("samples_training_input", {}).items():
        d = c.get("details", {})
        cost_samples[name] = {
            "input_len": c.get("tensor_length"),
            "analytic_fwd_macs": c.get("analytic_fwd_macs"),
            "analytic_bwd_macs_ESTIMATED": c.get("analytic_bwd_macs"),
            "optimizer_macs_ESTIMATED": c.get("optimizer_macs"),
            "score_or_selective_macs": d.get("score", d.get(
                "selective_executed_attention_and_controller")),
            "value_aggregation_macs": d.get("value_aggregation"),
            "actual_compute_ratio": d.get("actual_compute_ratio"),
        }

    baseline_acc = {k: v.get("accuracy") for k, v in
                    data.get("shortcut_baselines", {}).items()}

    report = {
        "status": "PREFLIGHT_PASS_AWAITING_AUTHORIZATION"
                  if all(gates.values()) and history_preserved else "PREFLIGHT_FAILED",
        "optimizer_step": 0,
        "training_status": "AWAITING_AUTHORIZATION",
        "authorization_note": "预检就绪不是训练授权；用户明确授权前不得执行 train_m4.py --authorize。",
        "gates": gates,
        "history_preserved": {"pass": history_preserved,
                              "sha256": original_sha,
                              "expected_sha256": history["sha256"],
                              "n_pass": history["n_pass"],
                              "n_tests": history["n_tests"]},
        "evidence": {
            "causality": {
                "source": "M41_CAUSALITY_AUDIT.json",
                "lengths": causal.get("lengths"),
                "tolerance": causal.get("tolerance"),
                "scope": causal.get("test_scope"),
                "models": model_summaries,
            },
            "incremental_and_kv": {
                "source": ["M41_CAUSALITY_AUDIT.json", "M41_KV_CACHE_AUDIT.json"],
                "kv_cases": [{"B": c["B"], "T": c["T"], "seed": c["seed"],
                              "chunks": c["chunks"],
                              "max_abs_diff_incremental": c["max_abs_diff_incremental"],
                              "max_abs_diff_chunked": c["max_abs_diff_chunked"],
                              "expected_cache_shapes": c["expected_cache_shapes"],
                              "cache_bytes_fp32_per_block": c["cache_bytes_fp32_per_block"]}
                             for c in kv.get("cases", [])],
            },
            "cost": {"source": "M41_COST_ACCOUNTING_AUDIT.json",
                     "precision": cost.get("precision"),
                     "backward": cost.get("backward"),
                     "optimizer": cost.get("optimizer"),
                     "hand_checks": cost.get("hand_checks"),
                     "samples_training_input": cost_samples},
            "metric": {"source": "M41_METRIC_AUDIT.json",
                       "checks": {k: {"pass": v["pass"], "observed": v.get("observed")}
                                  for k, v in metric.get("checks", {}).items()}},
            "data": {"source": "M41_DATA_AUDIT.json",
                     "checks": {k: {"pass": v["pass"], "observed": v.get("observed")}
                                for k, v in data.get("checks", {}).items()},
                     "shortcut_baseline_accuracy": baseline_acc,
                     "repair": {"source": "M41_DATA_REPAIR_AUDIT.json",
                                "before_sha256": repair["before_sha256"],
                                "after_sha256": repair["after_sha256"],
                                "n_repaired": repair["n_repaired"]}},
            "budget": {"source": "M41_BUDGET_AUDIT.json",
                       "checks": budget.get("checks"),
                       "live_ledger": live_budget},
            "positive_control": {"source": "M4_POSITIVE_CONTROL.json",
                                 "alignment": alignment,
                                 "gate_spec": spec,
                                 "ready_but_not_executed": gate_ready,
                                 "note": "梯度、参数更新与 95%/90% 门禁将在授权后的正控制阶段实测。"},
        },
    }
    p = HERE / "M41_PREFLIGHT_REPORT.json"
    p.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[m41-preflight] status={report['status']} gates="
          f"{sum(gates.values())}/{len(gates)} 写出 {p}")
    return report


if __name__ == "__main__":
    main()
