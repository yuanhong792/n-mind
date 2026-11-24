"""M41F 最终训练前准入报告（零梯度）。

只读取本轮新生成的 M41F 审计，不覆盖历史 M41/M4 证据。
写入 M41F_PREFLIGHT_REPORT.json，并记录代码/数据/预登记方案哈希，
供 preflight_guard.py 在 train_m4.py 每次真实梯度更新前复核。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import preflight_guard as guard
from positive_control import N_CONTROL, alignment_report, control_rows, gate_spec
from tokenizer_m4 import M4Tokenizer

HERE = Path(__file__).resolve().parent


def load(name):
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def deep_all_pass(obj):
    bad = []

    def walk(x, path="root"):
        if isinstance(x, dict):
            for k, v in x.items():
                if k == "pass" and v is False:
                    bad.append(path + "." + k)
                walk(v, path + "." + k)
        elif isinstance(x, list):
            for i, v in enumerate(x):
                walk(v, f"{path}[{i}]")

    walk(obj)
    return not bad, bad[:20]


def main():
    causal = load("M41F_CAUSALITY_AUDIT.json")
    kv = load("M41F_KV_CACHE_AUDIT.json")
    cost = load("M41F_COST_ACCOUNTING_AUDIT.json")
    cost_source = load("M41F_COST_SOURCE_AUDIT.json")
    budget = load("M41F_BUDGET_AUDIT.json")
    metric = load("M41F_METRIC_AUDIT.json")
    data = load("M41F_DATA_AUDIT_FULL.json")
    baseline = load("M41F_BASELINE_AUDIT.json")
    split = load("M41F_EVAL_SPLIT_AUDIT.json")
    gather = load("M41F_GATHER_AUDIT.json")
    spec = load("M41F_POSITIVE_CONTROL_SPEC.json")
    live = guard.verify_budget()

    causal_models = causal.get("models", {})
    causal_ok = (causal.get("status") == "PASS"
                 and len(causal_models) == 4
                 and all(m.get("status") == "PASS" for m in causal_models.values()))
    incremental_ok = causal_ok and all(
        all(v.get("pass") for v in m.get("full_vs_incremental", {}).values())
        and all(v.get("pass") for v in m.get("chunked_prefill", {}).values())
        and all(v.get("pass") for v in m.get("prefill_then_generate", {}).values())
        and all(v.get("pass") for v in m.get("batch_consistency", {}).values())
        for m in causal_models.values())
    padding_ok = causal_ok and all(
        all(v.get("pass") for v in m.get("padding_isolation", {}).values())
        for m in causal_models.values())
    routing_ok = causal_ok and all(
        m.get("routing_gradient", {}).get("pass") is True for m in causal_models.values())
    kv_ok = kv.get("status") == "PASS" and all(c.get("pass") for c in kv.get("cases", []))
    cost_ok = (cost.get("status") == "PASS" and cost_source.get("status") == "PASS"
               and all(v.get("pass") for v in cost.get("hand_checks", {}).values()))
    metric_ok = (metric.get("status") == "PASS"
                 and all(v.get("pass") for v in metric.get("checks", {}).values()))
    split_ok = split.get("status") == "PASS" and all(split.get("checks", {}).values())
    data_ok = (data.get("status") == "PASS"
               and all(v.get("pass") for v in data.get("checks", {}).values())
               and split_ok and baseline.get("status") == "PASS")
    budget_ok = budget.get("status") == "PASS" and live["pass"]
    tok = M4Tokenizer()
    rows = control_rows(tok, N_CONTROL)
    align = alignment_report(tok, rows)
    gate = gate_spec()
    positive_ready = (align["answer_only_alignment"] and len(rows) == 64
                      and spec.get("gate", {}).get("tf_accuracy_min") == 0.95
                      and spec.get("gate", {}).get("natural_eos_full_accuracy_min") == 0.90
                      and gate["sole_training_entry"] == "train_m4.py")

    gates = {
        "CAUSALITY_PASS": causal_ok,
        "INCREMENTAL_PARITY_PASS": incremental_ok,
        "KV_CACHE_PASS": kv_ok,
        "PADDING_ISOLATION_PASS": padding_ok,
        "ROUTING_GRADIENT_PASS": routing_ok,
        "ACTUAL_COST_ACCOUNTING_PASS": cost_ok,
        "STRICT_EOS_METRIC_PASS": metric_ok,
        "DATA_LEAKAGE_AUDIT_PASS": data_ok,
        "SHARED_BUDGET_PASS": budget_ok,
        "POSITIVE_CONTROL_GATE_READY": positive_ready,
    }

    old = (HERE / "M4_MATH_TESTS.json").read_bytes()
    old_sha = hashlib.sha256(old).hexdigest()
    baseline_snap = load("M41_BASELINE_SNAPSHOT.json")
    history_ok = old_sha == baseline_snap["sha256"]

    manifests = {"code": guard.code_manifest(), "data": guard.data_manifest(),
                 "plan": guard.plan_manifest()}
    report = {
        "status": "PREFLIGHT_PASS_AWAITING_AUTHORIZATION"
                  if all(gates.values()) and history_ok else "PREFLIGHT_FAILED",
        "optimizer_step": 0,
        "training_status": "AWAITING_AUTHORIZATION",
        "authorization_note": "预检通过不等于训练授权；唯一入口是 python train_m4.py --authorize。",
        "gates": gates,
        "manifests": manifests,
        "history_preserved": {"pass": history_ok, "M4_MATH_TESTS_sha256": old_sha,
                              "expected": baseline_snap["sha256"],
                              "base_commit": "a55c42a"},
        "evidence": {
            "causality": {"source": "M41F_CAUSALITY_AUDIT.json",
                          "lengths": causal.get("lengths"),
                          "models": {k: {"status": v.get("status")}
                                     for k, v in causal_models.items()},
                          "tolerance": causal.get("tolerance")},
            "kv": {"source": "M41F_KV_CACHE_AUDIT.json",
                   "cases": [{"B": c["B"], "T": c["T"],
                              "inc_diff": c["max_abs_diff_incremental"],
                              "chunk_diff": c["max_abs_diff_chunked"],
                              "cache_shapes": c["expected_cache_shapes"]}
                             for c in kv.get("cases", [])]},
            "cost": {"source": ["M41F_COST_ACCOUNTING_AUDIT.json",
                                "M41F_COST_SOURCE_AUDIT.json"],
                     "backward": cost.get("backward"),
                     "optimizer": cost.get("optimizer"),
                     "cost_source_checks": cost_source.get("checks"),
                     "cost_source_observed": cost_source.get("observed")},
            "metric": {"source": "M41F_METRIC_AUDIT.json",
                       "checks": {k: {"pass": v["pass"], "observed": v.get("observed")}
                                  for k, v in metric.get("checks", {}).items()}},
            "data": {"source": ["M41F_DATA_AUDIT_FULL.json",
                                "M41F_BASELINE_AUDIT.json",
                                "M41F_EVAL_SPLIT_AUDIT.json"],
                     "checks": data.get("checks"),
                     "split": split,
                     "baseline_A_positional": baseline.get(
                         "baseline_A_positional_shortcuts"),
                     "baseline_B_semantic": baseline.get("baseline_B_semantic_rules"),
                     "B_rename_original": baseline.get("B_rename_original_audit")},
            "gather": {"source": "M41F_GATHER_AUDIT.json",
                       "status": gather.get("status"), "checks": gather.get("checks"),
                       "elements": gather.get("elements"),
                       "profiler_memory": gather.get("profiler_memory"),
                       "latency": gather.get("latency")},
            "budget": {"source": "M41F_BUDGET_AUDIT.json",
                       "checks": budget.get("checks"), "live": live},
            "positive_control": {"source": "M41F_POSITIVE_CONTROL_SPEC.json",
                                 "alignment": align, "gate_spec": gate,
                                 "ready_but_not_executed": positive_ready,
                                 "must_have_both_reference_models": True},
        },
    }
    p = HERE / "M41F_PREFLIGHT_REPORT.json"
    p.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[m41f-preflight] status={report['status']} gates="
          f"{sum(gates.values())}/{len(gates)} 写出 {p}")
    return report


if __name__ == "__main__":
    main()
