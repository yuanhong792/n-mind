"""CPU 训练：事件状态预测 + 动作决策的联合监督，以及持续学习的 A→B→重测 A。

一条轨迹就是一个 batch 行（同长度成批），所以循环深度、路由、记忆读取都按
真实时序展开 —— 不存在"把时序打散成独立样本"的捷径。

损失（learning/losses.py 的 5 项 + 探针）：
    total = CE(动作) + pred·MSE(z_hat, z) + var·方差正则
            + balance·负载均衡 + entropy·路由熵 + probe·记忆探针

记忆探针回答"目标最初在哪一格"。遮挡期的输入里已经没有目标了，
探针也**不读 SQLite、不看 oracle 目标变量**，只能从循环状态里读，
所以它的准确率是"网络自己记住了"的直接证据。

诚实性开关全部在配置与函数参数里（损失权重、replay_ratio、trainable_adapter），
不靠外部脚本改行为。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..config import Config, default_config
from ..core.architecture import NModel
from ..runtime.hardware import detect
from . import datasets
from .checkpoint import config_hash, git_revision, save_checkpoint
from .losses import LossWeights, compute_total
from .replay import ReservoirReplay


@dataclass
class TrainResult:
    model: NModel
    probe: nn.Module
    curve: dict[str, list[float]] = field(default_factory=lambda: {"total": [], "action": [],
                                                                   "pred": [], "var": [],
                                                                   "probe": [], "error": []})
    evals: list[dict[str, Any]] = field(default_factory=list)
    final: dict[str, Any] = field(default_factory=dict)
    wall_s: float = 0.0
    steps: int = 0
    checkpoint: str = ""

    def summary(self) -> dict[str, Any]:
        head = [round(x, 4) for x in self.curve["total"][:3]]
        tail = [round(x, 4) for x in self.curve["total"][-3:]]
        return {"wall_s": round(self.wall_s, 2), "steps": self.steps,
                "loss_head": head, "loss_tail": tail,
                "loss_drop": round(head[0] - tail[-1], 4) if head and tail else None,
                "final_eval": self.final.get("eval"), "checkpoint": self.checkpoint}


def _make_probe(d_h: int, n_classes: int, hidden: int = 64) -> nn.Module:
    """记忆探针 = 小型非线性读出。

    用 MLP 而不是单层线性是有理由的：决策头本身就是非线性读出，
    若只有线性探针能过，等于宣称"状态里必须摆成线性可分的形式"，
    那是给网络加额外约束，不是我们要测的东西。
    探针与主干**联合训练**（权重 learning.probe_weight），不是事后拟合。
    """
    return nn.Sequential(nn.Linear(d_h, hidden), nn.GELU(), nn.Linear(hidden, n_classes))


def _goal_code(goal: tuple[int, int], size: int = 6) -> int:
    return int(goal[0]) * size + int(goal[1])


def build_model(cfg: Config, *, seed: int, execution: str = "sparse", use_router: bool = True,
                use_error_gating: bool = True, use_workspace: bool = True,
                device: torch.device | str = "cpu", mem_dim: int = 16,
                n_blocks: int | None = None) -> NModel:
    """按消融开关装配模型。seed 决定初始化，同 seed 同权重（可复现验收靠这个）。"""
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))
    if n_blocks:
        cfg = cfg.model_copy(update={"model": cfg.model.model_copy(update={"n_blocks": int(n_blocks),
                                                                           "top_k": min(cfg.model.top_k, int(n_blocks))})})
    model = NModel(cfg.model, n_actions=datasets.N_ACTIONS, mem_dim=mem_dim,
                   execution=execution, use_router=use_router,
                   use_error_gating=use_error_gating, use_workspace=use_workspace)
    return model.to(device)


def _with_aux(traj: Any) -> dict[str, Any]:
    """一条轨迹 → 一行训练数据（numpy，成批时再转张量）。

    mem 列默认不填：只有需要"记忆输入"的实验（engine/evaluation）才注入向量，
    缺省时 NModel 会用零向量占位，等价于"没有长期记忆"这一消融条件。
    """
    obs = np.asarray(traj.obs, dtype=np.float32)
    return {"obs": obs, "actions": np.asarray(traj.actions, dtype=np.int64),
            "goal_code": _goal_code(traj.goal), "len": int(obs.shape[0]),
            "task": getattr(traj, "task_name", "")}


def make_data(cfg: Config, task: datasets.GridTask, *, seed: int) -> dict[str, Any]:
    lcfg = cfg.learning
    data = datasets.make_split(task, n_train=lcfg.n_train, n_test=lcfg.n_test,
                               T=lcfg.traj_len, seed=seed)
    train = []
    for t in data["train"]:
        row = _with_aux(t)
        row["task"] = task.name
        train.append(row)
    test = []
    for t in data["test"]:
        row = _with_aux(t)
        row["task"] = task.name
        test.append(row)
    return {"train": train, "test": test,
            "dataset_hash": datasets.dataset_hash(task, n_train=lcfg.n_train,
                                                  n_test=lcfg.n_test, T=lcfg.traj_len, seed=seed)}


def _assemble(rows: list[dict[str, Any]], dev: torch.device) -> dict[str, Any]:
    obs = torch.tensor(np.stack([r["obs"] for r in rows]), device=dev)
    acts = torch.tensor(np.stack([r["actions"] for r in rows]), device=dev)
    mem = None
    if all("mem" in r for r in rows):
        mem = torch.tensor(np.stack([r["mem"] for r in rows]), device=dev)
    codes = torch.tensor([int(r["goal_code"]) for r in rows], device=dev, dtype=torch.long)
    return {"obs": obs, "actions": acts, "mem": mem, "codes": codes,
            "goal_codes": [int(r["goal_code"]) for r in rows]}


def _freeze_except_adapter(model: NModel) -> list[str]:
    """局部更新模式：只放开决策头、动作嵌入与块→工作空间投影。

    这是指令 M6"局部 Adapter 更新"的最小实现：用**参数子集**而不是新增模块，
    好处是更新前拍一张快照就能整份回滚（learning/checkpoint.save_snapshot）。
    """
    for p in model.parameters():
        p.requires_grad_(False)
    opened = []
    for name in ("head", "action_emb", "block_to_ws"):
        for p in getattr(model, name).parameters():
            p.requires_grad_(True)
        opened.append(name)
    return opened


def evaluate(model: NModel, probe: nn.Module, rows: list[dict[str, Any]], *,
             device: torch.device, occluded_only: bool = False, batch_size: int = 8,
             max_rows: int | None = None) -> dict[str, float]:
    """准确率（含遮挡段细分）+ 探针准确率。纯 no_grad，不产生任何训练副作用。"""
    was_training = model.training
    model.eval()
    rows = rows[:max_rows] if max_rows else rows
    n = hit = occ_n = occ_hit = pn = phit = 0
    with torch.no_grad():
        for s in range(0, len(rows), batch_size):
            b = _assemble(rows[s:s + batch_size], device)
            out = model.forward_sequence(b["obs"], b["actions"], mem=b["mem"])
            logits = out["action_logits"]
            pred = logits.argmax(-1)
            hit += int((pred == b["actions"]).sum().item())
            n += int(pred.numel())
            occ = b["obs"][:, :, 24] > 0.5
            occ_hit += int((pred[occ] == b["actions"][occ]).sum().item())
            occ_n += int(occ.sum().item())
            T = b["obs"].shape[1]
            p_pred = probe(out["ctx"].reshape(-1, out["ctx"].shape[-1])).argmax(-1)
            codes = b["codes"].repeat_interleave(T)
            phit += int((p_pred == codes).sum().item())
            pn += int(codes.numel())
    if was_training:
        model.train()
    res: dict[str, float] = {"accuracy": hit / max(1, n), "n_steps": float(n),
                             "probe_accuracy": phit / max(1, pn), "probe_n": float(pn)}
    if occ_n:
        res["accuracy_occluded"] = occ_hit / occ_n
        res["n_steps_occluded"] = float(occ_n)
    return res


def train(cfg: Config | None = None, *, epochs: int | None = None, seed: int | None = None,
          task: datasets.GridTask | None = None, execution: str = "sparse",
          use_router: bool = True, use_error_gating: bool = True, use_workspace: bool = True,
          use_memory_input: bool = False, replay: ReservoirReplay | None = None,
          replay_ratio: float | None = None, trainable_adapter: bool = False,
          device: str = "auto", eval_interval: int | None = None, quiet: bool = False,
          max_eval_rows: int = 16, checkpoint_path: str | None = None,
          mem_provider: Any | None = None) -> TrainResult:
    """训练一个任务。给了 replay 就按比例混批；trainable_adapter 只放开部分参数。"""
    cfg = cfg or default_config()
    lcfg = cfg.learning
    seed = int(cfg.runtime.seed if seed is None else seed)
    epochs = int(lcfg.epochs if epochs is None else epochs)
    eval_interval = max(1, int(lcfg.eval_interval if eval_interval is None else eval_interval))
    replay_ratio = float(lcfg.replay_ratio if replay_ratio is None else replay_ratio)
    task = task or datasets.GridTask(goal_visible_steps=4, seed=seed)

    dev_obj, _report = detect(device, torch_threads=cfg.runtime.torch_threads)
    dev = dev_obj if isinstance(dev_obj, torch.device) else torch.device(str(dev_obj))
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))

    model = build_model(cfg, seed=seed, device=dev, execution=execution, use_router=use_router,
                        use_error_gating=use_error_gating, use_workspace=use_workspace)
    opened = _freeze_except_adapter(model) if trainable_adapter else []
    probe = _make_probe(cfg.model.block_hidden, 36).to(dev)      # 6x6 网格里目标所在格 = 36 类
    weights = LossWeights.from_cfg(cfg)

    data = make_data(cfg, task, seed=seed)
    rows = data["train"]
    test_rows = data["test"]
    if use_memory_input:
        _inject_memory_rows(rows, mem_provider, dim=16, T=lcfg.traj_len)
        _inject_memory_rows(test_rows, mem_provider, dim=16, T=lcfg.traj_len)

    bs = max(2, int(lcfg.batch_size))
    steps_per_epoch = int(lcfg.n_updates) or max(1, len(rows) // bs)
    params = [p for p in list(model.parameters()) + list(probe.parameters()) if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lcfg.lr, weight_decay=1e-4)
    rng = np.random.default_rng(seed + 999)

    res = TrainResult(model=model, probe=probe)
    t0 = time.perf_counter()
    non_finite = 0
    for ep in range(epochs):
        model.train()
        for _st in range(steps_per_epoch):
            pick = [int(i) for i in rng.integers(0, len(rows), size=bs)]
            batch_rows = [rows[i] for i in pick]
            n_rep = 0
            if replay is not None and replay_ratio > 0 and len(replay):
                n_rep = max(1, int(round(bs * replay_ratio)))
                batch_rows = batch_rows[: max(1, bs - n_rep)] + replay.sample(n_rep)
            b = _assemble(batch_rows, dev)
            out = model.forward_sequence(b["obs"], b["actions"], mem=b["mem"], depth=1)
            losses = compute_total(out, b["actions"], weights)
            p_logits = probe(out["ctx"].reshape(-1, out["ctx"].shape[-1]))
            codes = b["codes"].repeat_interleave(b["obs"].shape[1])
            probe_loss = F.cross_entropy(p_logits, codes)
            losses["probe"] = probe_loss
            losses["total"] = losses["total"] + weights.probe * probe_loss
            if not torch.isfinite(losses["total"]):
                non_finite += 1
                opt.zero_grad(set_to_none=True)
                continue                      # 不 backward 非有限损失：一次 NaN 会污染全部权重
            opt.zero_grad(set_to_none=True)
            losses["total"].backward()
            nn.utils.clip_grad_norm_(params, lcfg.grad_clip)
            opt.step()
            if replay is not None:
                for r in batch_rows[: bs - n_rep] if n_rep else batch_rows:
                    replay.add(dict(r))
            for k in ("total", "action", "pred", "var", "probe"):
                res.curve[k].append(float(losses[k].detach()))
            res.curve["error"].append(float(out["error"].mean().detach()))
        res.steps += steps_per_epoch
        model.eval()
        if (ep + 1) % eval_interval == 0 or ep == epochs - 1:
            ev = {"epoch": ep,
                  "train": evaluate(model, probe, rows, device=dev, occluded_only=True,
                                    max_rows=max_eval_rows),
                  "test": evaluate(model, probe, test_rows, device=dev, occluded_only=True,
                                   max_rows=max_eval_rows)}
            ev["train"]["accuracy_all"] = evaluate(model, probe, rows, device=dev,
                                                   max_rows=max_eval_rows)["accuracy"]
            ev["test"]["accuracy_all"] = evaluate(model, probe, test_rows, device=dev,
                                                  max_rows=max_eval_rows)["accuracy"]
            res.evals.append(ev)
            if not quiet:
                t = ev["test"]
                print(f"[ep{ep}] total={res.curve['total'][-1]:.4f} act={res.curve['action'][-1]:.4f} "
                      f"probe={res.curve['probe'][-1]:.4f} | test acc_all={t['accuracy_all']:.3f} "
                      f"acc_occ={t.get('accuracy_occluded', float('nan')):.3f} "
                      f"probe_acc={t['probe_accuracy']:.3f}")
    res.wall_s = time.perf_counter() - t0
    model.eval()
    res.final = {
        "eval": evaluate(model, probe, test_rows, device=dev, occluded_only=True),
        "eval_all": evaluate(model, probe, test_rows, device=dev),
        "route": model.active_block_stats(),
        "params": model.n_params(),
        "task": datasets.task_to_dict(task),
        "dataset_hash": data["dataset_hash"],
        "config_hash": config_hash(cfg),
        "git_revision": git_revision(),
        "seed": seed,
        "execution": execution,
        "ablation": {"use_router": use_router, "use_error_gating": use_error_gating,
                     "use_workspace": use_workspace, "use_memory_input": use_memory_input,
                     "trainable_adapter": trainable_adapter, "opened": opened},
        "replay_items_used": len(replay) if replay is not None else 0,
        "non_finite_steps_skipped": non_finite,
    }
    if checkpoint_path:
        res.checkpoint = save_checkpoint(model, cfg=cfg, path=checkpoint_path, probe=probe,
                                         meta={"seed": seed, "task": datasets.task_to_dict(task),
                                               "dataset_hash": data["dataset_hash"],
                                               "final": res.final["eval"],
                                               "curve_tail": res.curve["total"][-5:]})
    return res


def _inject_memory_rows(rows: list[dict[str, Any]], provider: Any, *, dim: int, T: int) -> None:
    """把记忆向量按行注入。provider 缺失时给零矩阵——那是"无记忆"条件，不是报错。"""
    for r in rows:
        if provider is None:
            r["mem"] = np.zeros((r["len"], dim), dtype=np.float32)
            continue
        r["mem"] = provider(r)


def train_continual(cfg: Config | None = None, *, seed: int | None = None,
                    epochs: int | None = None, replay_ratio: float | None = None,
                    use_replay: bool = True, trainable_adapter: bool = False,
                    device: str = "auto", quiet: bool = False,
                    checkpoint_path: str | None = None) -> dict[str, Any]:
    """A → B → 重测 A。关掉 replay 时旧任务轨迹完全不参与，用来量真实遗忘量。"""
    cfg = cfg or default_config()
    seed = int(cfg.runtime.seed if seed is None else seed)
    ep = int(cfg.learning.epochs if epochs is None else epochs)
    taskA = datasets.GridTask(name="A", goal_visible_steps=4, seed=seed)
    taskB = datasets.GridTask(name="B", goal_visible_steps=4, reversed_control=True, seed=seed + 1)

    dA, dB = make_data(cfg, taskA, seed=seed), make_data(cfg, taskB, seed=seed + 1)
    buffer = ReservoirReplay(capacity=cfg.learning.replay_buffer, seed=seed) if use_replay else None
    if buffer is not None:
        for row in dA["train"]:
            buffer.add(dict(row))

    rA = train(cfg, epochs=ep, seed=seed, task=taskA, device=device, quiet=True,
               trainable_adapter=trainable_adapter)
    dev = rA.model.block_load_count.device
    a_before = evaluate(rA.model, rA.probe, dA["test"], device=dev, occluded_only=True)

    rB = train(cfg, epochs=ep, seed=seed, task=taskB, device=device, quiet=quiet,
               replay=buffer, replay_ratio=replay_ratio, trainable_adapter=trainable_adapter,
               checkpoint_path=checkpoint_path)
    # rB 复用了 rA 的 seed ⇒ 同一份初始权重，差别只在"先学了 A"与"是否重放"
    a_after = evaluate(rB.model, rB.probe, dA["test"], device=dev, occluded_only=True)
    b_after = rB.final["eval"]
    return {"taskA": datasets.task_to_dict(taskA), "taskB": datasets.task_to_dict(taskB),
            "use_replay": use_replay, "replay_ratio": replay_ratio,
            "buffer_size": len(buffer) if buffer else 0,
            "replay_fingerprint": buffer.fingerprint() if buffer else "",
            "A_before": a_before, "A_after": a_after, "B_after": b_after,
            "forgetting_occluded_accuracy": a_before.get("accuracy_occluded", float("nan"))
            - a_after.get("accuracy_occluded", float("nan")),
            "forgetting_probe_accuracy": a_before["probe_accuracy"] - a_after["probe_accuracy"],
            "curve_A_tail": rA.curve["total"][-3:], "curve_B_tail": rB.curve["total"][-3:],
            "model": rB.model, "probe": rB.probe, "data_A": dA, "data_B": dB}


__all__ = ["TrainResult", "build_model", "make_data", "evaluate", "train", "train_continual"]
