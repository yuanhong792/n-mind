"""M2.2 pred×稀疏 机制诊断：逐损失梯度对逐参数组的范数与方向一致性。

回答的问题（提示词 B 节明确要求不能只凭 F1 好转就宣称"梯度冲突"）：
  1. D 配置（pred 目标不 detach）里，pred 损失的梯度到底流到哪些参数组？
     量化：||g_pred|| 相对 ||g_action|| 在 blocks/router 上的占比。
  2. 在这些参数组上，action 梯度与 pred 梯度的方向一致性（cosine）如何？
     频繁强负相关 = 冲突证据；接近 0 = 只是干扰不冲突。
  3. detach（F1 配置）之后，pred 梯度是否只剩 predictor 自己（主干零）？
     —— 确认 detach 没有误伤预测头自己的学习，也没有把 pred 变成常数。

方法：加载已训练 checkpoint（D / F1 / C 各一），model.train() 模式下对同两个
训练批次做前向，逐损失项 autograd.grad(retain_graph=True)，不执行 optimizer.step，
不改动 checkpoint 状态。结果写 gradient_diagnostics.json。

用法：
  python experiments/m2_closeout/gradient_diagnostics.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from nmind.config import default_config  # noqa: E402
from nmind.learning import datasets as D  # noqa: E402
from nmind.learning.checkpoint import load_checkpoint  # noqa: E402
from nmind.learning.losses import (balance_loss, entropy_loss, prediction_loss,  # noqa: E402
                                   variance_regularizer, action_loss)
from nmind.learning.trainer import _assemble, make_data  # noqa: E402

SRC = ROOT / "experiments" / "m2_ablation"
OUT = ROOT / "experiments" / "m2_closeout"
SEED = 20261121

# (标签, checkpoint, detach_pred_target, 说明)
ARMS = [
    ("D_sparse_joint", "D_sparse_joint_s20261121.pt", False, "稀疏+联合，pred 目标不 detach"),
    ("F1_detach", "F1_detach_s20261121.pt", True, "稀疏+联合，pred 目标 detach"),
    ("C_sparse_ce", "C_sparse_ce_s20261121.pt", None, "稀疏+纯CE（对照组，无联合损失）"),
    ("B_dense_joint", "B_dense_joint_s20261121.pt", False, "稠密+联合（对照：STE 软掩码应有任务梯度到达 router）"),
]

GROUPS = ["encoder", "blocks", "router", "workspace", "predictor", "head",
          "action_emb", "block_to_ws"]


def group_named_params(model) -> dict[str, list[torch.nn.Parameter]]:
    return {name: [p for p in getattr(model, name).parameters() if p.requires_grad]
            for name in GROUPS}


def grad_norm(grads: list[torch.Tensor | None]) -> float:
    sq = 0.0
    for g in grads:
        if g is not None:
            sq += float(g.detach().float().pow(2).sum())
    return sq ** 0.5


def grad_vec(grads: list[torch.Tensor | None]) -> torch.Tensor | None:
    parts = [g.detach().float().reshape(-1) for g in grads if g is not None]
    if not parts:
        return None
    return torch.cat(parts)


def cosine(a: torch.Tensor | None, b: torch.Tensor | None) -> float | None:
    if a is None or b is None:
        return None
    na, nb = float(a.norm()), float(b.norm())
    if na == 0.0 or nb == 0.0:
        return None
    return round(float((a * b).sum() / (na * nb)), 4)


def measure_arm(tag: str, ckpt_name: str, detach: bool | None, note: str) -> dict:
    cfg = default_config()
    model, _meta = load_checkpoint(SRC / ckpt_name, cfg=cfg, device="cpu",
                                   allow_config_mismatch=True)
    model.train()                      # 与训练态一致（STE 软掩码路径）
    groups = group_named_params(model)
    data = make_data(cfg, D.GridTask(goal_visible_steps=4, seed=SEED), seed=SEED)
    rows = data["train"]

    blob = torch.load(SRC / ckpt_name, map_location="cpu", weights_only=True)
    probe = None
    if blob.get("probe") is not None and detach is not None:   # 仅 joint 臂（D/F1）；C 的探针未训练，不测
        probe = _make_probe_like(model)
        probe.load_state_dict(blob["probe"])
        probe.train()

    per_batch = []
    for batch_id, pick in enumerate(([0, 1, 2, 3, 4, 5, 6, 7],
                                     [8, 9, 10, 11, 12, 13, 14, 15])):
        b = _assemble([rows[i] for i in pick], torch.device("cpu"))
        # 每个批次都要重新取图：optimizer.zero_grad 的等价物是直接用 autograd.grad
        out = model.forward_sequence(b["obs"], b["actions"], mem=b["mem"], depth=1)
        losses = {
            "action": action_loss(out["action_logits"], b["actions"]),
            "pred": prediction_loss(out["z_hat"], out["z"], detach_target=bool(detach)),
            "var": variance_regularizer(out["z"], target_std=0.25),
            "balance": balance_loss(out["route_probs"], n_blocks=out["route_probs"].shape[-1]),
            "entropy": entropy_loss(out["route_probs"]),
        }
        if probe is not None:
            T = b["obs"].shape[1]
            codes = b["codes"].repeat_interleave(T)
            losses["probe"] = F.cross_entropy(probe(out["ctx"].reshape(-1, out["ctx"].shape[-1])),
                                              codes)

        term_grads: dict[str, dict[str, float | None]] = {}
        vecs: dict[str, dict[str, torch.Tensor | None]] = {}
        for tname, tloss in losses.items():
            term_grads[tname] = {}
            vecs[tname] = {}
            for gname, params in groups.items():
                if not params:
                    continue
                gs = torch.autograd.grad(tloss, params, retain_graph=True,
                                         allow_unused=True)
                term_grads[tname][gname] = round(grad_norm(gs), 6)
                vecs[tname][gname] = grad_vec(gs)

        batch_result = {
            "batch_id": batch_id,
            "loss_values": {k: round(float(v.detach()), 6) for k, v in losses.items()},
            "grad_norms": term_grads,
            "cosine_action_vs_pred": {
                g: cosine(vecs["action"][g], vecs["pred"][g]) for g in GROUPS},
            "cosine_action_vs_balance": {
                g: cosine(vecs["action"][g], vecs["balance"][g]) for g in GROUPS},
            "pred_grad_share_on_blocks": _share(term_grads, "blocks"),
            "pred_grad_share_on_router": _share(term_grads, "router"),
            "pred_health": {
                # pred 分支没有退化成常数：z_hat 方差应与 z 同量级
                "z_std": round(float(out["z"].detach().float().std()), 4),
                "z_hat_std": round(float(out["z_hat"].detach().float().std()), 4),
            },
        }
        per_batch.append(batch_result)
        model.zero_grad(set_to_none=True)

    return {"arm": tag, "checkpoint": ckpt_name, "detach_pred_target": detach,
            "note": note, "batches": per_batch}


def _make_probe_like(model):
    from nmind.learning.trainer import _make_probe
    return _make_probe(model.cfg.block_hidden, 36)


def _share(term_grads: dict, group: str) -> float | None:
    ga = term_grads.get("action", {}).get(group, 0.0) or 0.0
    gp = term_grads.get("pred", {}).get(group, 0.0) or 0.0
    if ga + gp == 0.0:
        return None
    return round(gp / (ga + gp), 4)


def main() -> int:
    results = [measure_arm(tag, ckpt, det, note) for tag, ckpt, det, note in ARMS]
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / "gradient_diagnostics.json"
    dest.write_text(json.dumps({
        "seed": SEED, "batch_size": 8, "mode": "model.train() + autograd.grad，未执行 optimizer.step",
        "interpretation": {
            "question": "pred(不detach)的梯度是否进入 blocks/router 并与 action 梯度方向冲突",
            "how_to_read": "D 臂 pred_grad_share_on_blocks/router>0 且 cosine_action_vs_pred 显著为负、"
                           "F1 臂两处归零（pred 只训练 predictor）＝detach 作用位置的直接证据；"
                           "负 cosine 是冲突证据，不是机制证明——还需训练轨迹对照。",
        },
        "arms": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"→ {dest}")
    for arm in results:
        b0 = arm["batches"][0]
        print(f"[{arm['arm']}] pred_share blocks={b0['pred_grad_share_on_blocks']} "
              f"router={b0['pred_grad_share_on_router']} "
              f"cos(a,pred) blocks={b0['cosine_action_vs_pred']['blocks']} "
              f"router={b0['cosine_action_vs_pred']['router']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
