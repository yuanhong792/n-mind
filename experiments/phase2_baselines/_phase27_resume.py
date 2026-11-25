"""阶段 2.7 B 续跑（机器死机后）：跳过已完成臂，种子优先轮转，剩余预算 3000s。

协议不变：模型/N/划分/种子/LR/停止规则/封存集均沿用 preregister_phase27.json；
仅调度顺序改为 seed-major（每单元先拿 1 个种子），并在报告披露。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

import _phase27_run as R  # noqa: E402
from mqar_run import OUT_DIR, log, atomic_json, _read_arm, _spawn  # noqa: E402
import mqar_task as T  # noqa: E402

REMAIN_S = 3000.0  # 60 分钟总预算已用约 10 分钟（崩溃前的训练）


def completed(aid: str) -> bool:
    r = _read_arm({"arm_id": aid})
    return bool(r and str(r.get("status")).startswith("COMPLETE") and r.get("levels"))


def main():
    lrs = json.loads((OUT_DIR / "phase27_lr_select.json").read_text(encoding="utf-8"))["lrs"]
    t_end = time.time() + REMAIN_S
    log(f"resume: lrs={lrs} deadline {time.ctime(t_end)}")
    seal_seq = {2: T.DATA_DIR / "mqar_N2_heldout_seq_seal2.npz",
                4: T.DATA_DIR / "mqar_N4_heldout_seq_seal2.npz"}
    seal_name = {2: T.DATA_DIR / "mqar_N2_heldout_name_seal3.npz",
                 4: T.DATA_DIR / "mqar_N4_heldout_name_seal3.npz"}
    cells = [(split, N, model) for split in R.SPLITS for N in R.NS for model in R.MODELS]
    arms = []
    skipped = 0
    for seed in R.SEEDS5:
        for split, N, model in cells:
            aid = f"p27_{model}_N{N}_{split}_s{seed}"
            if completed(aid):
                skipped += 1
                continue
            arms.append({"arm_id": aid, "model": model, "lr": lrs[model], "N": N,
                         "split": split, "seed": seed, "gate_all": True,
                         "seal_path": str(seal_seq[N] if split == "heldout_seq" else seal_name[N])})
    log(f"resume: 待跑 {len(arms)}，跳过已完成 {skipped}")
    if arms:
        R.run_pool(arms, t_end)

    # 汇总
    all_ids = [f"p27_{m}_N{n}_{s}_s{seed}" for s in R.SPLITS for n in R.NS
               for m in R.MODELS for seed in R.SEEDS5]
    results = []
    for aid in all_ids:
        r = _read_arm({"arm_id": aid})
        if r is None:
            r = {"arm_id": aid, "status": "INCONCLUSIVE_BUDGET", "steps_done": 0}
        if r.get("status") == "RUNNING":  # 崩溃残留：无最终评测
            r = dict(r, status="INCONCLUSIVE_BUDGET")
        results.append(r)
    table = []
    for split in R.SPLITS:
        for model in R.MODELS:
            for N in R.NS:
                rs = [r for r in results if r.get("split") == split and r.get("model") == model
                      and r.get("N") == N and "test" in r]
                n_gate, firsts = 0, []
                for r in rs:
                    los = [lv.get("dev_wilson_lo", 0) for lv in r.get("levels", [])]
                    if any(x >= 0.90 for x in los):
                        n_gate += 1
                        firsts.append(next(lv["step"] for lv in r["levels"]
                                           if lv["dev_wilson_lo"] >= 0.90))
                ranks = [r["test"]["rank_acc"] for r in rs]
                ems = [r["test"]["em_free"] for r in rs]
                fols = [r["test"]["follow"] for r in rs]
                table.append({
                    "split": split, "model": model, "N": N, "lr": lrs[model],
                    "n_seeds_done": len(rs), "n_gate_seeds": n_gate,
                    "first_gate_steps": firsts,
                    "rank_mean": round(float(np.mean(ranks)), 4) if ranks else None,
                    "em_mean": round(float(np.mean(ems)), 4) if ems else None,
                    "follow_mean": round(float(np.mean(fols)), 4) if fols else None,
                    "seeds": [{"seed": r.get("seed"), "status": r.get("status"),
                               "steps": r.get("steps_done"),
                               "rank": r.get("test", {}).get("rank_acc"),
                               "em": r.get("test", {}).get("em_free"),
                               "follow": r.get("test", {}).get("follow"),
                               "n_clusters": r.get("test", {}).get("n_clusters")}
                              for r in rs]})
    summary = {"status": "DONE" if all(completed(a) or r.get("status") == "INCONCLUSIVE_BUDGET"
                                       for a, r in zip(all_ids, results)) else "PARTIAL",
               "eval_sanity": "PASS 8/8 @ 54af4dc",
               "lrs": lrs, "table": table,
               "resume_note": "机器死机后从 14:23 续跑；跳过已完成；seed-major 调度；剩余预算 3000s",
               "n8_status": "NOT_RUN_BUDGET",
               "n_arms": len(results),
               "n_complete": sum(1 for r in results if str(r.get("status", "")).startswith("COMPLETE")),
               "n_inconclusive": sum(1 for r in results if r.get("status") == "INCONCLUSIVE_BUDGET"),
               "n_crashed": sum(1 for r in results if r.get("status") == "ARM_CRASHED"),
               "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    atomic_json(OUT_DIR / "phase27_summary.json", summary)
    log(f"summary complete={summary['n_complete']} incl={summary['n_inconclusive']} "
        f"crash={summary['n_crashed']}")


if __name__ == "__main__":
    main()
