"""长程能力回归评测：同一模型在新技能（未见 holdout）与旧技能（冻结测试集）上的矩阵。

口径（M4 任务书第 7 节）：
  * 新技能指标只在**未见 holdout** 上计算（test_new_skill_metric_is_unseen_holdout）；
  * 冲突步 = 旧规则 oracle != 新规则标签的遮挡步（排除两规则巧合的白拿分）；
  * sealed_eval 永不进入训练/回放，这里只做冻结评测。
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Any

from ..learning import datasets as D
from ..learning.trainer import _assemble, evaluate


def occluded_acc(model, rows: list[dict], device: torch.device) -> float | None:
    ev = evaluate(model, None, rows, device=device, occluded_only=True)
    return None if "accuracy_occluded" not in ev else round(ev["accuracy_occluded"], 4)


def rule_conflict_masks(rows: list[dict], vis: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    """遮挡期掩码：冲突步 = 旧规则(min-distance) oracle != B 规则标签；其余为巧合步。

    行内 obs 编码沿用 26 维基础任务口径（前 6 维 x one-hot、次 6 维 y one-hot）。
    """
    n, T = len(rows), int(rows[0]["len"])
    conflict = torch.zeros(n, T, dtype=torch.bool)
    coincident = torch.zeros(n, T, dtype=torch.bool)
    for i, r in enumerate(rows):
        goal = (int(r["goal_code"]) // 6, int(r["goal_code"]) % 6)
        for t in range(vis, T):
            row = np.asarray(r["obs"][t])
            pos = (int(np.argmax(row[0:6])), int(np.argmax(row[6:12])))
            base = D.oracle_action(pos, goal, tuple(), 6, reversed_control=False)
            is_c = base != int(r["actions"][t])
            conflict[i, t] = is_c
            coincident[i, t] = not is_c
    return conflict, coincident


def masked_acc(model, rows: list[dict], device: torch.device,
               mask: torch.Tensor) -> float | None:
    if int(mask.sum()) == 0:
        return None
    batch = _assemble(rows, device)
    with torch.no_grad():
        pred = model.forward_sequence(batch["obs"], batch["actions"],
                                      mem=batch["mem"])["action_logits"].argmax(-1)
    tgt = batch["actions"]
    return round(float((pred[mask] == tgt[mask]).float().mean()), 4)


def eval_skill(model, rows: list[dict], device: torch.device, *,
               conflict: torch.Tensor, coincident: torch.Tensor) -> dict[str, Any]:
    return {"acc_occluded_total": occluded_acc(model, rows, device),
            "acc_occluded_conflict": masked_acc(model, rows, device, conflict),
            "acc_occluded_coincident": masked_acc(model, rows, device, coincident),
            "n_conflict": int(conflict.sum()), "n_coincident": int(coincident.sum())}
