"""语言模型（NMindLM）局部更新原语：答案掩码 CE + 硬预算 + 墙钟 + 标签置换负对照。

为什么不复用 `learning.online_trainer.bounded_stream_update`：那个函数的行是网格观测
dict（obs/action/reward），损失走 NModel 的五项联合目标与 `_assemble`，喂不进 token 序列。
语言侧的更新口径在 M5.1/M5.2 已经跑通（只学答案 span、批量前向、预算到点即停），
这里把它抽成可复用实现，供 Bot 候选学习与后续实验共用，避免出现两套训练器。

诚实性约束：
  * `trainable="adapter"` 只放开 head/block_to_ws，`"all"` 为全参——从零/小预算阶段
    必须用全参（M5.1 v1 用局部更新口径训随机主干，TF=0 的实锤教训）；
  * 预算 `AdaptationBudget` 到点抛 BudgetExceeded，墙钟到点直接 break 并如实报告步数；
  * 非有限损失跳过并计数，绝不静默；
  * `label_permutation_seed` 给定时把答案换成别人的答案（负对照），
    若负对照同样"提升"，说明存在泄漏或评测错误。
"""

from __future__ import annotations

import random
import time
from typing import Any

import torch

from ..learning.online_adaptation import AdaptationBudget
from .tokenizer import PAD


def encode_rows(tok, prompts: list[str], answers: list[str]) -> list[tuple[list[int], tuple[int, int]]]:
    """(prompt, answer) → (token ids, 答案 span)。prompt 前缀零梯度，只有答案进损失。"""
    rows: list[tuple[list[int], tuple[int, int]]] = []
    for p, a in zip(prompts, answers):
        pre = tok.encode(p)
        ans = tok.encode(a, add_eos=True)
        rows.append((pre + ans, (len(pre), len(pre) + len(ans))))
    return rows


def answer_masked_loss(model, batch, dev) -> tuple[torch.Tensor, int, int]:
    """批量答案掩码 CE（M5.1 口径：等长左补齐，一次前向）。返回 (loss, 命中, 目标数)。"""
    t_max = max(len(ids) for ids, _s in batch)
    x = torch.full((len(batch), t_max - 1), PAD, dtype=torch.long)
    y = torch.full((len(batch), t_max - 1), PAD, dtype=torch.long)
    span_mask = torch.zeros(len(batch), t_max - 1, dtype=torch.bool)
    for i, (ids, (s, e)) in enumerate(batch):
        seq = torch.tensor(ids, dtype=torch.long)
        x[i, : len(ids) - 1] = seq[:-1]
        y[i, : len(ids) - 1] = seq[1:]
        span_mask[i, max(0, s - 1): min(len(ids) - 1, e)] = True
    logits, _st = model(x.to(dev))
    logp = torch.log_softmax(logits, dim=-1)
    tgt = y.to(dev)
    flat_lp = logp.reshape(-1, logp.shape[-1])
    flat_t = tgt.reshape(-1)
    idx = torch.nonzero((span_mask.reshape(-1) & (flat_t != PAD)).to(dev)).reshape(-1)
    if idx.numel() == 0:
        return torch.zeros((), device=dev, requires_grad=True), 0, 0
    sel_lp, sel_t = flat_lp[idx], flat_t[idx]
    loss = -sel_lp[torch.arange(sel_t.numel()), sel_t].mean()
    return loss, int((sel_lp.argmax(-1) == sel_t).sum()), int(sel_t.numel())


def permuted_answer_rows(rows, *, seed: int):
    """负对照：固定 seed 的置换，把每条样本的答案换成集合内另一条的答案。
    若打乱标签后成绩同样"提升"，说明存在泄漏或评测错误。"""
    n = len(rows)
    order = random.Random(seed).sample(range(n), n) if n else []
    perm_rows = []
    for i, (ids, _span) in enumerate(rows):
        j = order[i]
        ans_ids = rows[j][0][rows[j][1][0]:rows[j][1][1]]
        pre = ids[:rows[i][1][0]]
        perm_rows.append((pre + list(ans_ids), (len(pre), len(pre) + len(ans_ids))))
    return perm_rows


def set_trainable(model, *, trainable: str = "all",
                  adapter_prefixes: tuple[str, ...] = ("head.", "block_to_ws.")) -> list[str]:
    """返回本次可训练参数名；"all" 全参，"adapter" 只放开输出头与工作空间写入。"""
    names: list[str] = []
    for name, p in model.named_parameters():
        ok = trainable == "all" or name.startswith(adapter_prefixes)
        p.requires_grad_(ok)
        if ok:
            names.append(name)
    return names


KNOWN_TRAINABLE_MODES = ("all", "adapter")
KNOWN_ADAPTER_PREFIXES = ("head.", "block_to_ws.")


def set_trainable(model, *, trainable: str = "all",
                  adapter_prefixes: tuple[str, ...] | None = None) -> list[str]:
    """返回本次可训练参数名；"all" 全参，"adapter" 只放开输出头与工作空间写入。

    白名单校验（Gate-B 004）：trainable 只允许 all/adapter；adapter_prefixes 必须是
    KNOWN_ADAPTER_PREFIXES 的子集——未知前缀直接报错，防止诊断实验静默放开意外模块。"""
    if trainable not in KNOWN_TRAINABLE_MODES:
        raise ValueError(f"trainable={trainable!r} 不在白名单 {KNOWN_TRAINABLE_MODES}")
    prefixes = adapter_prefixes if adapter_prefixes is not None else ("head.", "block_to_ws.")
    if trainable == "adapter":
        unknown = [p for p in prefixes if p not in KNOWN_ADAPTER_PREFIXES]
        if unknown:
            raise ValueError(f"adapter_prefixes {unknown} 不在白名单 {KNOWN_ADAPTER_PREFIXES}")
        if not prefixes:
            raise ValueError("adapter 模式下前缀列表不能为空（否则将静默冻结全部参数）")
    names: list[str] = []
    for name, p in model.named_parameters():
        ok = trainable == "all" or name.startswith(prefixes)
        p.requires_grad_(ok)
        if ok:
            names.append(name)
    return names


def masked_stream_update(model, rows, *, updates: int, dev: torch.device,
                         lr: float = 3e-3, seed: int = 0, label: str = "lm-update",
                         wall_limit_s: float = 60.0, batch_size: int = 16,
                         replay_rows=None, replay_ratio: float = 0.0,
                         trainable: str = "all",
                         adapter_prefixes: tuple[str, ...] | None = None,
                         label_permutation_seed: int | None = None,
                         budget: AdaptationBudget | None = None,
                         record_grad_norms: bool = False,
                         batch_mix: tuple[list, list] | None = None,
                         phase_pool: list[tuple[int, list]] | None = None) -> dict[str, Any]:
    """有界流更新：最多 `updates` 次 optimizer.step 且不超过 wall_limit_s。

    adapter_prefixes：None=沿用 set_trainable 默认（head.+block_to_ws.）；传入自定义
    前缀元组可进一步收窄 adapter 范围（Gate-B 003 冻结输出头实验用），默认行为不变。

    课程呈现模式（Gate-B 005 方案 C，两者互斥，同时提供直接报错；默认 None=None=
    既有随机采样行为完全不变）：
      * batch_mix=(pool_a, pool_b)：交错混合——每步 batch 从 pool_a 采一半、pool_b
        采另一半，A/B 行次占比精确可控；
      * phase_pool=[(steps, pool), ...]：顺序分相——按累计步数切换采样池（A→B 两阶段）。
    两种模式下 `rows` 不再参与采样（校验非空仍保留），label_permutation_seed 不生效——
    负对照臂的乱标签置换由调用方对池预先完成（learning_loop 负责）。

    步数口径（Gate-B 004 拆分）：`attempted_steps` = 尝试的梯度步数（含非有限 loss
    跳过，预算按尝试计防止无限循环）；`optimizer_steps_executed` = **真实完成**
    backward+step 的次数——非有限跳过绝不冒充完成优化（001—003 轮两值恰好相等的
    场景下口径无影响，但口径本身必须诚实）。record_grad_norms=True 时额外记录每组
    可训练参数的梯度范数（诊断接口，默认关闭，不影响默认训练行为）。"""
    if not rows:
        raise ValueError("空 rows：没有 verified 经验不得触发更新")
    if batch_mix is not None and phase_pool is not None:
        raise ValueError("batch_mix 与 phase_pool 互斥：课程呈现模式只能二选一")
    budget = budget or AdaptationBudget(updates, label=label)
    if label_permutation_seed is not None:
        rows = permuted_answer_rows(rows, seed=label_permutation_seed)
    if adapter_prefixes is None:
        trainable_names = set_trainable(model, trainable=trainable)
    else:
        trainable_names = set_trainable(model, trainable=trainable,
                                        adapter_prefixes=adapter_prefixes)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    rng = random.Random(seed)
    curve: list[float] = []
    non_finite = 0
    hits = total = 0
    t0 = time.perf_counter()
    stopped_by: str = "budget_exhausted"
    # 共享预算：调用方可把同一个 AdaptationBudget 传给多条臂（如 neural + negative 对照）。
    # 此时必须区分两个上限——updates 是"本臂自己的步数上限"，budget.cap 是"所有臂合计的硬总闸"。
    # 历史缺陷（R1 实验抓到）：条件写成 budget.used < min(updates, budget.cap)，拿"全部臂累计
    # 已用步数"去比"本臂上限"，于是第一臂用完自己的份额后，第二臂（负对照）一步都跑不到，
    # 四臂对照名存实亡。现在本臂计数从 used0 起算，总闸单独判。
    used0 = budget.used
    arm_cap = int(updates)
    steps_done = 0
    grad_norms: dict[str, float] = {}
    grad_norm_peak: dict[str, float] = {}
    model.train()
    while True:
        if budget.used - used0 >= arm_cap:
            stopped_by = "budget_exhausted" if budget.used >= budget.cap else "arm_cap_reached"
            break
        if budget.used >= budget.cap:
            stopped_by = "total_budget_exhausted"
            break
        if time.perf_counter() - t0 > wall_limit_s:
            stopped_by = "wall_limit"
            break
        budget.spend(1, label)
        if batch_mix is not None:
            pool_a, pool_b = batch_mix
            half_a = min(len(pool_a), max(1, batch_size // 2))
            half_b = min(len(pool_b), max(1, batch_size - half_a))
            batch = ([pool_a[i] for i in rng.sample(range(len(pool_a)), half_a)] +
                     [pool_b[i] for i in rng.sample(range(len(pool_b)), half_b)])
        elif phase_pool is not None:
            cur = budget.used - used0
            pool = phase_pool[-1][1]
            acc = 0
            for ph_steps, ph_rows in phase_pool:
                if cur < acc + ph_steps:
                    pool = ph_rows
                    break
                acc += ph_steps
            batch = [pool[i] for i in rng.sample(range(len(pool)), min(batch_size, len(pool)))]
        else:
            batch = [rows[i] for i in rng.sample(range(len(rows)), min(batch_size, len(rows)))]
        loss, h, n = answer_masked_loss(model, batch, dev)
        if replay_rows and replay_ratio > 0:
            rbatch = [replay_rows[i] for i in
                      rng.sample(range(len(replay_rows)), min(4, len(replay_rows)))]
            r_loss, _rh, _rn = answer_masked_loss(model, rbatch, dev)
            loss = loss + float(replay_ratio) * r_loss
        if not torch.isfinite(loss):
            non_finite += 1
            opt.zero_grad(set_to_none=True)
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        if record_grad_norms:
            for pname, p in ((n, prm) for n, prm in model.named_parameters() if prm.requires_grad):
                g = float(p.grad.norm()) if p.grad is not None else 0.0
                grad_norms[pname] = round(g, 6)
                grad_norm_peak[pname] = max(grad_norm_peak.get(pname, 0.0), round(g, 6))
        opt.step()
        steps_done += 1
        curve.append(round(float(loss.detach()), 4))
        hits += h
        total += n
    wall = round(time.perf_counter() - t0, 2)
    model.eval()
    return {"arm": label, "optimizer_steps_executed": steps_done,
             "attempted_steps": budget.used - used0,
             "attempted_vs_executed_note": "attempted 含非有限 loss 跳过（预算按尝试计防无限循环）；"
                                            "executed 只数真实完成 backward+step 的次数",
             "arm_step_cap": arm_cap,
             "budget_used_total": budget.used, "budget_cap": budget.cap,
             "loss_first": curve[0] if curve else None, "loss_last": curve[-1] if curve else None,
             "curve": curve, "non_finite_steps_skipped": non_finite,
             "answer_token_accuracy": round(hits / total, 4) if total else None,
             "trainable": trainable,
             "trainable_prefixes": (list(adapter_prefixes) if adapter_prefixes is not None
                                     else list(KNOWN_ADAPTER_PREFIXES) if trainable == "adapter" else None),
             "trainable_names": trainable_names,
             "trainable_params": len(trainable_names),
             "grad_norm_last": grad_norms if record_grad_norms else None,
             "grad_norm_peak": grad_norm_peak if record_grad_norms else None,
             "wall_s": wall, "stopped_by": stopped_by,
             "label_permutation_seed": label_permutation_seed}


def parameter_audit(before_state: dict[str, torch.Tensor], after_state: dict[str, torch.Tensor]
                    ) -> dict[str, Any]:
    """逐张量参数变化取证（torch.equal，不用容差）：变化数、最大 delta、冻结违规。"""
    changed, unchanged, max_delta, violations = 0, 0, 0.0, []
    for k, v0 in before_state.items():
        v1 = after_state.get(k)
        if v1 is None:
            violations.append(f"missing:{k}")
            continue
        if torch.equal(v0, v1):
            unchanged += 1
        else:
            changed += 1
            if v0.is_floating_point():
                max_delta = max(max_delta, float((v1 - v0).abs().max()))
    return {"tensors_changed": changed, "tensors_unchanged": unchanged,
             "max_abs_delta": round(max_delta, 6), "missing_keys": violations}
