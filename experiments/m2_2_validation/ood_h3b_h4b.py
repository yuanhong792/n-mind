"""M2.2-D H3b/H4b v2 评测驱动。

评测集（冻结，报告前只评一次；审计见 manifest）：
  H3b holdout：四拓扑族 × 布局实例 seed 2001（与 dev 1001 不同），n=32
  H4b holdout：非退化规则反转，seed 2002，n=32（min_conflict=0.5）
checkpoint：A/C/F1/F10 × 预登记 10 seed（experiments/m2_ablation/）

H3b v2 追加显式障碍图并改用 BFS oracle，输入从 26 维变为 62 维；旧 checkpoint
不能参与该项评分，因此这里只冻结并审计数据，等待同口径重训。

H4b 保留为“反转规则分布偏移”诊断，直接分别统计规则冲突步与巧合步；没有规则提示，
不得解释为模型识别了规则切换。

用法：
  python -u experiments/m2_2_validation/ood_h3b_h4b.py
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from nmind.config import default_config  # noqa: E402
from nmind.learning import ood_layouts as O  # noqa: E402
from nmind.learning.checkpoint import load_checkpoint  # noqa: E402
from nmind.learning.trainer import _assemble  # noqa: E402

HERE = Path(__file__).resolve().parent
SRC = ROOT / "experiments" / "m2_ablation"
ARMS = ("A_dense_ce", "C_sparse_ce", "F1_detach", "F10_ste")
SEEDS = [20261131 + i for i in range(10)]


def main() -> int:
    t0 = time.perf_counter()
    cfg = default_config()
    device = torch.device("cpu")

    h3b_hold = O.make_h3b_split("holdout", n=32, seed=2001)
    h4b_hold = O.make_h4b_split("holdout", n=32, seed=2002)
    h4b_cued_hold = O.make_h4b_cued_split("holdout", n=32, seed=2003)
    audit = {"h3b": h3b_hold["manifest"], "h4b_distribution_shift": h4b_hold["manifest"],
             "h4b_cued": h4b_cued_hold["manifest"]}

    def h4_masks(rows: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
        conflict, coincident = [], []
        for row in rows:
            r_conflict, r_coincident = [], []
            goal = tuple(row["goal"])
            for t in range(row["len"]):
                if t < 4:
                    r_conflict.append(False)
                    r_coincident.append(False)
                    continue
                pos = O.decode_pos(row["obs"][t])
                normal = O.oracle_action(pos, goal, tuple(), 6, reversed_control=False)
                is_conflict = normal != int(row["actions"][t])
                r_conflict.append(is_conflict)
                r_coincident.append(not is_conflict)
            conflict.append(r_conflict)
            coincident.append(r_coincident)
        return torch.tensor(conflict, dtype=torch.bool), torch.tensor(coincident, dtype=torch.bool)

    conflict_mask, coincident_mask = h4_masks(h4b_hold["rows"])
    batch = _assemble(h4b_hold["rows"], device)

    results: dict[str, dict] = {}
    for arm in ARMS:
        for seed in SEEDS:
            ckpt = SRC / f"{arm}_s{seed}.pt"
            if not ckpt.exists():
                print(f"[skip] {ckpt}", flush=True)
                continue
            model, _meta = load_checkpoint(ckpt, cfg=cfg, device="cpu",
                                           allow_config_mismatch=True)
            model.eval()
            entry = results.setdefault(arm, {}).setdefault(str(seed), {})
            with torch.no_grad():
                pred = model.forward_sequence(batch["obs"], batch["actions"],
                                              mem=batch["mem"])["action_logits"].argmax(-1)
            target = batch["actions"]
            entry["h4b_accuracy_conflict"] = round(float(
                (pred[conflict_mask] == target[conflict_mask]).float().mean()), 4)
            entry["h4b_accuracy_coincident"] = round(float(
                (pred[coincident_mask] == target[coincident_mask]).float().mean()), 4)
            occ_mask = conflict_mask | coincident_mask
            entry["h4b_accuracy_total"] = round(float(
                (pred[occ_mask] == target[occ_mask]).float().mean()), 4)
            print(f"[{arm} s{seed}] H4b conflict={entry['h4b_accuracy_conflict']:.3f} "
                  f"coincident={entry['h4b_accuracy_coincident']:.3f}", flush=True)

    # ---- 配对统计（C−F10 与 A−F10，逐 seed） ----
    def paired(key_a: str, key_b: str, metric: str) -> dict:
        used = [s for s in SEEDS if str(s) in results.get(key_a, {})
                and str(s) in results.get(key_b, {})]
        diffs = [results[key_a][str(s)][metric] - results[key_b][str(s)][metric]
                 for s in used]
        rng = np.random.default_rng(1234)
        arr = np.array(diffs)
        means = [arr[rng.integers(0, len(arr), size=len(arr))].mean() for _ in range(10000)]
        lo, hi = np.percentile(means, [2.5, 97.5])
        return {"mean": round(float(arr.mean()), 4),
                "std_ddof1": round(float(arr.std(ddof=1)), 4),
                "ci95": [round(float(lo), 4), round(float(hi), 4)],
                "per_seed": {str(s): round(d, 4) for s, d in zip(used, diffs)}}

    summary: dict[str, dict] = {}
    for arm in ARMS:
        for metric in ("h4b_accuracy_conflict", "h4b_accuracy_coincident",
                       "h4b_accuracy_total"):
            vals = [results[arm][str(s)][metric] for s in SEEDS if str(s) in results[arm]]
            summary[f"{arm}/{metric}"] = {
                "mean": round(statistics.mean(vals), 4),
                "std_ddof1": round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0,
                "worst": min(vals)}
    summary["paired_C_minus_F10"] = {
        m: paired("C_sparse_ce", "F10_ste", m)
        for m in ("h4b_accuracy_conflict", "h4b_accuracy_coincident",
                  "h4b_accuracy_total")}

    payload = {"captured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
               "device": "cpu", "n_seeds": len(SEEDS), "seeds": SEEDS,
               "audit": audit,
               "h3b_model_evaluation": {
                   "status": "not_run_incompatible_input_dim",
                   "old_checkpoint_obs_dim": cfg.model.event_numeric_dim,
                   "h3b_v2_obs_dim": O.OBS_DIM_WITH_MAP,
                   "required_next_step": "用显式障碍图输入和多布局课程重新训练同口径模型"},
               "h4b_interpretation": {
                   "name": "reversed-rule distribution shift",
                   "rule_cue_present": False,
                   "claim_limit": "不得解释为规则切换识别或闭环控制"},
               "h4b_cued_model_evaluation": {
                   "status": "not_run_incompatible_input_dim",
                   "old_checkpoint_obs_dim": cfg.model.event_numeric_dim,
                   "h4b_cued_obs_dim": O.OBS_DIM_WITH_RULE_CUE,
                   "required_next_step": "用正常/反转配对课程和显式规则提示重新训练"},
               "h4b_counts": {"conflict_steps": int(conflict_mask.sum()),
                               "coincident_steps": int(coincident_mask.sum())},
               "results": results, "summary": summary}
    dest = HERE / "ood_h3b_h4b_v2.json"
    tmp = dest.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(dest)
    print(f"→ {dest}（用时 {time.perf_counter() - t0:.0f}s）", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
