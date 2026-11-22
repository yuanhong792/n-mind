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
                n_blocks: int | None = None, router_ste: bool = False,
                pred_mode: str | None = None) -> NModel:
    """按消融开关装配模型。seed 决定初始化，同 seed 同权重（可复现验收靠这个）。"""
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))
    if n_blocks:
        cfg = cfg.model_copy(update={"model": cfg.model.model_copy(update={"n_blocks": int(n_blocks),
                                                                           "top_k": min(cfg.model.top_k, int(n_blocks))})})
    model = NModel(cfg.model, n_actions=datasets.N_ACTIONS, mem_dim=mem_dim,
                   execution=execution, use_router=use_router,
                   use_error_gating=use_error_gating, use_workspace=use_workspace,
                   router_ste=router_ste,
                   pred_mode=cfg.model.pred_mode if pred_mode is None else pred_mode)
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
                               n_val=lcfg.n_val, T=lcfg.traj_len, seed=seed)

    def rows_of(trajs: list) -> list[dict[str, Any]]:
        out = []
        for t in trajs:
            row = _with_aux(t)
            row["task"] = task.name
            out.append(row)
        return out

    return {"train": rows_of(data["train"]), "test": rows_of(data["test"]),
            "val": rows_of(data.get("val", [])),
            "dataset_hash": datasets.dataset_hash(task, n_train=lcfg.n_train,
                                                  n_test=lcfg.n_test, n_val=lcfg.n_val,
                                                  T=lcfg.traj_len, seed=seed),
            "split_version": datasets.SPLIT_VERSION}


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
            if probe is not None:
                T = b["obs"].shape[1]
                p_pred = probe(out["ctx"].reshape(-1, out["ctx"].shape[-1])).argmax(-1)
                codes = b["codes"].repeat_interleave(T)
                phit += int((p_pred == codes).sum().item())
                pn += int(codes.numel())
    if was_training:
        model.train()
    res: dict[str, float] = {"accuracy": hit / max(1, n), "n_steps": float(n)}
    if probe is not None:
        res["probe_accuracy"] = phit / max(1, pn) if pn else float("nan")
        res["probe_n"] = float(pn)
    if occ_n:
        res["accuracy_occluded"] = occ_hit / occ_n
        res["n_steps_occluded"] = float(occ_n)
    return res


def _extract_ctx(model: NModel, rows: list[dict[str, Any]], device: torch.device,
                 batch_size: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    """冻结主干，抽出 (池化状态, 目标格标签) 供事后探针拟合/评测。"""
    xs, ys = [], []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for s in range(0, len(rows), batch_size):
            b = _assemble(rows[s:s + batch_size], device)
            out = model.forward_sequence(b["obs"], b["actions"], mem=b["mem"])
            ctx = out["ctx"]
            xs.append(ctx.reshape(-1, ctx.shape[-1]))
            ys.append(b["codes"].repeat_interleave(ctx.shape[1]))
    if was_training:
        model.train()
    return torch.cat(xs), torch.cat(ys)


def fit_posthoc_probe(model: NModel, fit_rows: list[dict[str, Any]],
                      eval_rows: list[dict[str, Any]], *, device: torch.device,
                      epochs: int = 60, lr: float = 1e-2, batch_size: int = 8,
                      seed: int = 0) -> dict[str, float]:
    """事后读出探针：主干冻结、特征固定，只训一个 MLP 读"目标最初在哪一格"。

    与联合探针的区别是**它不参与主干训练**，所以探针分数高不能再用
    "探针和主干互相适配"来解释。2x2 消融里四组必须都用这把尺子量记忆，
    否则 CE-only 组没有被训练过的探针，比较就无从谈起。
    """
    torch.manual_seed(seed)
    probe = _make_probe(model.cfg.block_hidden, 36, hidden=64).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-4)
    feats, codes = _extract_ctx(model, fit_rows, device, batch_size)
    probe.train()
    for _ in range(epochs):
        loss = F.cross_entropy(probe(feats), codes)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    probe.eval()
    f_e, c_e = _extract_ctx(model, eval_rows, device, batch_size)
    with torch.no_grad():
        pred = probe(f_e).argmax(-1)
    return {"probe_posthoc_accuracy": float((pred == c_e).float().mean().item()),
            "probe_posthoc_n": float(c_e.numel()),
            "probe_posthoc_loss": round(float(loss.detach()), 4)}


def train(cfg: Config | None = None, *, epochs: int | None = None, seed: int | None = None,
          task: datasets.GridTask | None = None, execution: str = "sparse",
          use_router: bool = True, use_error_gating: bool = True, use_workspace: bool = True,
          use_memory_input: bool = False, replay: ReservoirReplay | None = None,
          replay_ratio: float | None = None, trainable_adapter: bool = False,
          device: str = "auto", eval_interval: int | None = None, quiet: bool = False,
          checkpoint_path: str | None = None,
          mem_provider: Any | None = None, loss_mode: str = "joint",
          posthoc_probe: bool = True, loss_overrides: dict[str, float] | None = None,
           router_ste: bool = False, pred_mode: str | None = None,
           data_override: dict[str, Any] | None = None,
           model: NModel | None = None, probe: nn.Module | None = None) -> TrainResult:
    """训练一个任务。给了 replay 就按比例混批；trainable_adapter 只放开部分参数。

    M4（2026-11-22）新增：`model`/`probe` 显式传入现存模型时**跳过 build_model()**，
    从给定权重继续更新（增量学习唯一合法入口）。传入模型与 cfg 的输入维度、
    execution/router_ste/pred_mode 语义不一致时**报错停止**，绝不静默重建。

    `loss_mode`：`joint` = CE + 预测 + 方差 + 均衡 + 熵 + 联合探针（M1 原配置）；
    `ce` = **只留 CE**，探针不参与训练（2x2 消融的 A/C 组就是这一档）。
    两档走同一条代码路径、同一份数据、同一组初始化，差异只有损失项——
    这样"联合损失是否拖累决策"才是可归因的实验，而不是两个脚本的差异。
    """
    if loss_mode not in ("joint", "ce"):
        raise ValueError(f"loss_mode 只能是 'joint' 或 'ce'，实际 {loss_mode!r}")
    cfg = cfg or default_config()
    lcfg = cfg.learning
    seed = int(cfg.runtime.seed if seed is None else seed)
    if pred_mode is not None and pred_mode != cfg.model.pred_mode:
        # 单一事实源：kwarg 覆盖必须同步进 cfg，否则 checkpoint 里存的 cfg 与实际
        # 装配的模型不一致（F10_next 首轮踩过：存档记 current，重载按 current 重建）。
        cfg = cfg.model_copy(update={"model": cfg.model.model_copy(update={"pred_mode": pred_mode})})
    epochs = int(lcfg.epochs if epochs is None else epochs)
    eval_interval = max(1, int(lcfg.eval_interval if eval_interval is None else eval_interval))
    replay_ratio = float(lcfg.replay_ratio if replay_ratio is None else replay_ratio)
    task = task or datasets.GridTask(goal_visible_steps=4, seed=seed)

    dev_obj, _report = detect(device, torch_threads=cfg.runtime.torch_threads)
    dev = dev_obj if isinstance(dev_obj, torch.device) else torch.device(str(dev_obj))
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))

    if model is not None:
        # M4 续训入口：显式传入的模型优先，禁止静默重新初始化。
        # 任何语义不一致都直接报错——宁可停止实验，也不偷偷重建后仍声称"续训"。
        conflicts: list[str] = []
        if int(model.cfg.event_numeric_dim) != int(cfg.model.event_numeric_dim):
            conflicts.append(f"event_numeric_dim 模型={model.cfg.event_numeric_dim} cfg={cfg.model.event_numeric_dim}")
        if model.execution != execution:
            conflicts.append(f"execution 模型={model.execution} kwarg={execution}")
        if model.use_router != use_router:
            conflicts.append(f"use_router 模型={model.use_router} kwarg={use_router}")
        if bool(model.router_ste) != bool(router_ste):
            conflicts.append(f"router_ste 模型={model.router_ste} kwarg={router_ste}")
        if pred_mode is not None and model.pred_mode != pred_mode:
            conflicts.append(f"pred_mode 模型={model.pred_mode} kwarg={pred_mode}")
        if conflicts:
            raise ValueError("传入模型与训练配置不一致（增量学习禁止静默重建）：" + "；".join(conflicts))
        model = model.to(dev)
        pred_mode_eff = model.pred_mode
        opened = _freeze_except_adapter(model) if trainable_adapter else []
        init_mode = "provided_model"
    else:
        model = build_model(cfg, seed=seed, device=dev, execution=execution, use_router=use_router,
                            use_error_gating=use_error_gating, use_workspace=use_workspace,
                            router_ste=router_ste, pred_mode=pred_mode)
        pred_mode_eff = model.pred_mode   # None 时取 cfg.model.pred_mode
        opened = _freeze_except_adapter(model) if trainable_adapter else []
        init_mode = "fresh_build"
    probe_was_provided = probe is not None
    if probe_was_provided:
        probe = probe.to(dev)
    else:
        probe = _make_probe(cfg.model.block_hidden, 36).to(dev)      # 6x6 网格里目标所在格 = 36 类
    weights = LossWeights.from_cfg(cfg) if loss_mode == "joint" else LossWeights(
        pred=0.0, var=0.0, balance=0.0, entropy=0.0, probe=0.0)
    if loss_overrides:
        for k2, v2 in loss_overrides.items():
            if not hasattr(weights, k2):
                raise ValueError(f"未知损失权重 {k2!r}；可用：{list(weights.__dataclass_fields__)}")
            setattr(weights, k2, float(v2))

    data = make_data(cfg, task, seed=seed) if data_override is None else data_override
    for split in ("train", "test"):
        if not data.get(split):
            raise ValueError(f"data_override 的 {split!r} 不能为空")
        actual_dim = int(np.asarray(data[split][0]["obs"]).shape[-1])
        if actual_dim != cfg.model.event_numeric_dim:
            raise ValueError(
                f"{split} 观测维度 {actual_dim} 与模型 event_numeric_dim "
                f"{cfg.model.event_numeric_dim} 不一致")
    rows = data["train"]
    test_rows = data["test"]
    val_rows = data["val"] or test_rows       # 无 val 时退回 test（并在返回体里标注）
    if use_memory_input:
        _inject_memory_rows(rows, mem_provider, dim=16, T=lcfg.traj_len)
        _inject_memory_rows(test_rows, mem_provider, dim=16, T=lcfg.traj_len)
        if data["val"]:
            _inject_memory_rows(val_rows, mem_provider, dim=16, T=lcfg.traj_len)

    bs = max(2, int(lcfg.batch_size))
    steps_per_epoch = int(lcfg.n_updates) or max(1, len(rows) // bs)
    train_params = list(model.parameters())
    if loss_mode == "joint":
        train_params += list(probe.parameters())   # CE 模式下探针不参与优化，避免偷偷改变量
    params = [p for p in train_params if p.requires_grad]
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
            losses = compute_total(out, b["actions"], weights,
                                   pred_shift=(pred_mode_eff == "next"))
            if loss_mode == "joint":
                p_logits = probe(out["ctx"].reshape(-1, out["ctx"].shape[-1]))
                codes = b["codes"].repeat_interleave(b["obs"].shape[1])
                probe_loss = F.cross_entropy(p_logits, codes)
                losses["probe"] = probe_loss
                losses["total"] = losses["total"] + weights.probe * probe_loss
            else:
                losses["probe"] = losses["total"].new_zeros(())
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
            probe_live = probe if loss_mode == "joint" else None
            ev = {"epoch": ep,
                  "train": evaluate(model, probe_live, rows, device=dev, occluded_only=True),
                  "val": evaluate(model, probe_live, val_rows, device=dev, occluded_only=True)}
            ev["train"]["accuracy_all"] = evaluate(model, probe_live, rows, device=dev)["accuracy"]
            ev["val"]["accuracy_all"] = evaluate(model, probe_live, val_rows, device=dev)["accuracy"]
            res.evals.append(ev)
            if not quiet:
                t = ev["val"]
                print(f"[ep{ep}] total={res.curve['total'][-1]:.4f} act={res.curve['action'][-1]:.4f} "
                      f"probe={res.curve['probe'][-1]:.4f} | val acc_all={t['accuracy_all']:.3f} "
                      f"acc_occ={t.get('accuracy_occluded', float('nan')):.3f} "
                      f"probe_acc={t.get('probe_accuracy', float('nan')):.3f}")
    res.wall_s = time.perf_counter() - t0
    model.eval()
    probe_live = probe if loss_mode == "joint" else None
    res.final = {
        # 兼容旧字段：eval = 测试集（只在训练结束后摸一次）
        "eval": evaluate(model, probe_live, test_rows, device=dev, occluded_only=True),
        "eval_all": evaluate(model, probe_live, test_rows, device=dev),
        "val": evaluate(model, probe_live, val_rows, device=dev, occluded_only=True),
        "route": model.active_block_stats(),
        "params": model.n_params(),
        "task": datasets.task_to_dict(task),
        "dataset_hash": data["dataset_hash"],
        "split_version": data["split_version"],
        "val_is_test_fallback": not bool(data["val"]),
        "config_hash": config_hash(cfg),
        "git_revision": git_revision(),
        "seed": seed,
        "execution": execution,
        "loss_mode": loss_mode,
        "loss_weights": {k2: float(getattr(weights, k2)) for k2 in
                         ("pred", "var", "balance", "entropy", "probe")},
        "ablation": {"use_router": use_router, "use_error_gating": use_error_gating,
                     "use_workspace": use_workspace, "use_memory_input": use_memory_input,
                     "trainable_adapter": trainable_adapter, "opened": opened,
                     "router_ste": router_ste, "pred_mode": pred_mode_eff,
                     "init_mode": init_mode, "provided_probe": probe_was_provided},
        "replay_items_used": len(replay) if replay is not None else 0,
        "non_finite_steps_skipped": non_finite,
    }
    if posthoc_probe:
        pos = fit_posthoc_probe(model, rows, val_rows, device=dev, seed=seed)
        pos_test = fit_posthoc_probe(model, rows, test_rows, device=dev, seed=seed)
        res.final["probe_posthoc_val"] = pos
        res.final["probe_posthoc_test"] = pos_test
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
    """A → B → 重测 A（M4 修复：B 阶段真正继承 A 后权重）。

    2026-11-22 修复前的缺陷：B 阶段调用 train() 时重新 build_model()，同一 seed 只能
    重建"相同初始权重"，A 阶段学到的参数完全没有进入 B 阶段——forgetting 指标无法
    解释为"先学 A 再学 B 之后的遗忘"。
    修复后：A 训练 → 快照 → B 阶段以 A 后权重为唯一起点（逐位克隆 + 内置继承断言）;
    optimizer 为增量阶段新建并明确记录（不继承 A 阶段动量）。
    seed 语义分开记录：模型参数初始化 seed = seed（A/B 同一权重世系）；B 阶段训练数据
    采样 seed = seed + 1（独立数据流，与重新初始化无关）。关掉 replay 时旧任务轨迹
    完全不参与 B 阶段，量到的就是纯参数漂移遗忘。
    """
    from .online_adaptation import clone_snapshot, state_hash, verify_bitwise_inheritance

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

    # ---- M4 修复核心：B 阶段从 A 后权重逐位克隆出发，绝不靠"相同 seed"冒充继承 ----
    a_snap = clone_snapshot(rA.model, rA.probe)
    b_model = build_model(cfg, seed=seed, device=dev)      # 同结构新对象；权重立刻被 A 后快照覆盖
    b_probe = _make_probe(cfg.model.block_hidden, 36).to(dev)
    b_model.load_state_dict(a_snap["model"], strict=True)
    if a_snap.get("probe"):
        b_probe.load_state_dict(a_snap["probe"], strict=True)
    inherit = verify_bitwise_inheritance(a_snap, b_model, b_probe)
    if not inherit["all_bitwise_equal"]:
        raise RuntimeError(f"B 阶段权重继承断言失败：{inherit['mismatched_keys']}")

    b_train = train(cfg, epochs=ep, seed=seed + 1, task=taskB, device=device, quiet=quiet,
                    model=b_model, probe=b_probe, replay=buffer, replay_ratio=replay_ratio,
                    trainable_adapter=trainable_adapter, checkpoint_path=checkpoint_path)
    a_after = evaluate(b_train.model, b_train.probe, dA["test"], device=dev, occluded_only=True)
    b_after = b_train.final["eval"]
    return {"taskA": datasets.task_to_dict(taskA), "taskB": datasets.task_to_dict(taskB),
            "use_replay": use_replay, "replay_ratio": replay_ratio,
            "buffer_size": len(buffer) if buffer else 0,
            "replay_fingerprint": buffer.fingerprint() if buffer else "",
            "A_before": a_before, "A_after": a_after, "B_after": b_after,
            "forgetting_occluded_accuracy": a_before.get("accuracy_occluded", float("nan"))
            - a_after.get("accuracy_occluded", float("nan")),
            "forgetting_probe_accuracy": a_before["probe_accuracy"] - a_after["probe_accuracy"],
            "curve_A_tail": rA.curve["total"][-3:], "curve_B_tail": b_train.curve["total"][-3:],
            "model": b_train.model, "probe": b_train.probe, "data_A": dA, "data_B": dB,
            "lineage": {"model_init_seed": seed, "b_data_sampling_seed": seed + 1,
                        "parent_state_sha256": a_snap["sha256"],
                        "child_state_sha256": state_hash(b_train.model, b_train.probe),
                        "inheritance_verified": inherit["all_bitwise_equal"],
                        "optimizer": "B 阶段新建 AdamW（动量不继承 A 阶段）",
                        "init_mode_b": b_train.final["ablation"]["init_mode"]}}


__all__ = ["TrainResult", "build_model", "make_data", "evaluate", "train", "train_continual"]
