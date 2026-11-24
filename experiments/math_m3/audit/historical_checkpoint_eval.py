"""M2.1 P0.1：加载本地 M2 checkpoint 做 SEP 四组交叉评测。

交叉设计：
    训练协议 ∈ {legacy, sep}  ×  推理协议 ∈ {legacy, sep}
主公平比较：Legacy训练→Legacy推理  vs  SEP训练→SEP推理。
分别评 Final 与 Best。

纪律：
- 历史 Legacy/SEP Best 均由 M2 训练时的 SEP 评测选出，可能受错误协议影响，
  不得直接视为公平选模结果 → 同时报告 Final，并标注 best_selection_contaminated。
- 缺权重标 NOT_AVAILABLE，不伪造。
- 封存集不参与选模；本审计用 val 固定面板（与 M2 训练面板同协议生成）。
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
M3 = HERE.parent
M2ROOT = M3.parent / "math_m2"
for p in (str(M2ROOT), str(M2ROOT / "models"), str(M3)):
    if p not in sys.path:
        sys.path.insert(0, p)

from tokenizer_m2 import M2Tokenizer  # noqa: E402
from m2_protocol_repair import generate, PROTOCOLS  # noqa: E402
from candidate_registry import build  # noqa: E402
from curriculum_m2 import PANEL_TASKS  # noqa: E402

CHECK = M2ROOT / "checkpoints"
DATA = M2ROOT / "data"
SEED = 20261201


def load_rows(name):
    p = DATA / f"m2_{name}.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def build_panel(rows, per_task=25):
    import random
    rng = random.Random(SEED)
    panel = []
    for task in PANEL_TASKS:
        tr = [r for r in rows if r["task"] == task]
        rng.shuffle(tr)
        panel.extend(tr[:per_task])
    return panel


def load_checkpoint(arch, run_id, role):
    """返回 (model, source)；缺权重返回 (None, 'NOT_AVAILABLE')。"""
    p = CHECK / f"{run_id}_{role}.pt"
    if not p.exists():
        return None, "NOT_AVAILABLE"
    model = build(arch, M2Tokenizer().vocab_size)
    blob = torch.load(p, map_location="cpu", weights_only=False)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, str(p.relative_to(M3.parent))


def eval_protocol(model, tok, panel, protocol):
    per = defaultdict(lambda: {"n": 0, "ok": 0, "eos0": 0, "empty": 0})
    details = []
    for r in panel:
        g = generate(model, tok, r["prompt"], protocol)
        ok = g["text"].strip() == str(r["answer"]).strip()
        d = per[r["task"]]
        d["n"] += 1; d["ok"] += int(ok)
        d["eos0"] += int(g["finish_reason"] == "eos" and not g["text"].strip())
        d["empty"] += int(not g["text"].strip())
        details.append({"task": r["task"], "prompt": r["prompt"],
                        "expected": r["answer"], "predicted": g["text"], "ok": ok})
    pt = {t: {"n": d["n"], "ok": d["ok"], "accuracy": round(d["ok"] / max(1, d["n"]), 4),
              "immediate_eos_rate": round(d["eos0"] / max(1, d["n"]), 4)}
          for t, d in sorted(per.items())}
    n = sum(d["n"] for d in per.values()); ok = sum(d["ok"] for d in per.values())
    return {"per_task": pt,
            "macro_accuracy": round(sum(d["accuracy"] for d in pt.values()) / max(1, len(pt)), 4),
            "micro_accuracy": round(ok / max(1, n), 4),
            "total_n": n, "total_ok": ok, "details": details}


def main():
    torch.set_num_threads(12)
    tok = M2Tokenizer()
    val = load_rows("val")
    sealed = load_rows("sealed")
    panel = build_panel(val)
    sealed_panel = build_panel(sealed)
    # run_id 为基名，load_checkpoint 会追加 _{role}.pt
    runs = {
        "legacy": {"best": "S1_R1_legacy", "final": "S1_R1_legacy"},
        "sep": {"best": "S1_R1_sep", "final": "S1_R1_sep"},
    }
    report = {
        "status": "M21_CROSS_PROTOCOL_EVAL_DONE",
        "note": "两个协议各 256 step，同 R1 GRU 架构/同数据/同 seed 20261201；"
                "面板来自 M2 val，封存面板来自 M2 sealed，均不参与选模。",
        "best_selection_contamination": (
            "M2 训练时 Legacy/SEP 的 Best 都由 SEP 推理评测选出；"
            "Legacy Best 可能不是 Legacy 推理下的最优 checkpoint → 主比较同时报告 Final。"),
        "panel_n": len(panel), "sealed_panel_n": len(sealed_panel),
        "cross": {},
        "checkpoint_sources": {},
    }
    for train_proto, roles in runs.items():
        for role, run_id in roles.items():
            model, source = load_checkpoint("R1", run_id, role)
            report["checkpoint_sources"][f"{train_proto}_{role}"] = source
            if model is None:
                report["cross"][f"{train_proto}_{role}"] = {
                    "status": "NOT_AVAILABLE", "source": source}
                continue
            entry = {"status": "OK", "source": source, "by_infer_protocol": {}}
            for infer_proto in PROTOCOLS:
                ev = eval_protocol(model, tok, panel, infer_proto)
                ev_sealed = eval_protocol(model, tok, sealed_panel, infer_proto)
                entry["by_infer_protocol"][infer_proto] = {
                    "val_macro": ev["macro_accuracy"], "val_micro": ev["micro_accuracy"],
                    "val_n": ev["total_n"], "val_ok": ev["total_ok"],
                    "val_immediate_eos": round(
                        sum(d["immediate_eos_rate"] * d["n"] for d in ev["per_task"].values()) /
                        max(1, ev["total_n"]), 4),
                    "val_per_task": ev["per_task"],
                    "sealed_macro": ev_sealed["macro_accuracy"],
                    "sealed_micro": ev_sealed["micro_accuracy"],
                    "sealed_n": ev_sealed["total_n"], "sealed_ok": ev_sealed["total_ok"],
                    "details": ev["details"][:20],
                }
            report["cross"][f"{train_proto}_{role}"] = entry
            del model

    # 主公平比较：各协议自己的 Final 与 Best
    flat = {}
    for train_proto in PROTOCOLS:
        for role in ("final", "best"):
            e = report["cross"].get(f"{train_proto}_{role}", {})
            if e.get("status") == "OK":
                same = e["by_infer_protocol"][train_proto]
                flat[f"{train_proto}_{role}"] = {"val_macro": same["val_macro"],
                                                 "sealed_macro": same["sealed_macro"]}
    report["fair_comparison"] = flat
    if "legacy_final" in flat and "sep_final" in flat:
        a = flat["legacy_final"]["val_macro"]; b = flat["sep_final"]["val_macro"]
        report["fair_verdict"] = {
            "legacy_final_val_macro": a, "sep_final_val_macro": b,
            "sep_advantage": round(b - a, 4),
            "holds_under_correct_protocol": b > a,
        }
    else:
        report["fair_verdict"] = {"status": "NOT_AVAILABLE"}
    (M3 / "M21_CROSS_PROTOCOL_EVAL.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"],
                      "sources": report["checkpoint_sources"],
                      "fair": report["fair_comparison"],
                      "verdict": report["fair_verdict"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
