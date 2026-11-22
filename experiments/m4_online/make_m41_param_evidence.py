"""从隔离保存的 REJECTED 候选权重重算真实参数变化证据（剔除训练态统计 buffer）。

背景：run_m41_stability.py 的 max_param_abs_delta 把 block_load_count（纯计数器 buffer，
48 步训练累加 ≈6830）算进了参数最大变化——这是口径错误。本脚本从
ckpts/rejected/ 的候选 checkpoint 与父权重逐张量重算，只统计**参数**（不含
block_load_count / router.load_bias 两个训练态 buffer），并二次核验冻结参数逐位不变。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from nmind.config import default_config  # noqa: E402
from nmind.learning.checkpoint import load_checkpoint  # noqa: E402
from nmind.learning.online_adaptation import state_hash  # noqa: E402

HERE = Path(__file__).resolve().parent
PARENT = ROOT / "experiments" / "m2_ablation" / "C_sparse_ce_s20261131.pt"
BUFFERS = {"block_load_count", "router.load_bias"}
TRAINABLE_PREFIXES = ("head.", "action_emb.", "block_to_ws.")   # scope=adapter


def main() -> int:
    cfg = default_config()
    cfg.learning.traj_len = 24
    parent, _meta = load_checkpoint(PARENT, cfg=cfg, device="cpu",
                                    allow_config_mismatch=True)
    pstate = {k: v.detach().cpu() for k, v in parent.state_dict().items()}
    out: dict = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                 "parent_checkpoint": str(PARENT),
                 "note": "从 ckpts/rejected/ 隔离候选重算；剔除训练态统计 buffer；"
                          "block_load_count 的表观大 delta 是计数器累加，不是权重变化",
                 "arms": {}}
    for arm in ("Neural-only", "Hybrid"):
        p = HERE / "ckpts" / "rejected" / f"m4.1-stability-001_{arm}_candidate.pt"
        if not p.exists():
            out["arms"][arm] = {"missing": str(p)}
            continue
        m, _meta = load_checkpoint(p, cfg=cfg, device="cpu", allow_config_mismatch=True)
        cstate = {k: v.detach().cpu() for k, v in m.state_dict().items()}
        per: dict = {}
        for k in sorted(pstate):
            if k in BUFFERS:
                continue
            d = (cstate[k] - pstate[k]).abs()
            per[k] = {"delta_max_abs": round(float(d.max()), 6),
                      "delta_l2": round(float(d.norm()), 6),
                      "changed_elements": int((d > 0).sum().item()),
                      "bitwise_equal": bool(torch.equal(cstate[k], pstate[k]))}
        changed = [k for k, v in per.items() if not v["bitwise_equal"]]
        trainable_changed = [k for k in changed if k.startswith(TRAINABLE_PREFIXES)]
        frozen_violations = [k for k in changed if not k.startswith(TRAINABLE_PREFIXES)]
        mx = max((v["delta_max_abs"], k) for k, v in per.items())
        out["arms"][arm] = {
            "candidate_path": str(p),
            "child_state_sha256": state_hash(m),
            "n_param_tensors": len(per),
            "changed_count": len(changed),
            "changed_params": changed,
            "trainable_changed_count": len(trainable_changed),
            "frozen_violations": frozen_violations,
            "frozen_all_bitwise_unchanged": not frozen_violations,
            "max_param_delta": {"value": mx[0], "key": mx[1]},
            "top_deltas": sorted(((v["delta_max_abs"], k) for k, v in per.items()),
                                  reverse=True)[:5],
            "per_tensor": per,
        }
        print(f"[{arm}] changed={len(changed)} max={mx[0]} ({mx[1]}) "
              f"frozen_ok={not frozen_violations}", flush=True)
    (HERE / "m41_param_evidence.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"→ {HERE / 'm41_param_evidence.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
