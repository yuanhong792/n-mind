"""指令第 8 节的四个可复现任务：记忆、决策、动态预算，加上统一的读出探针。

为什么探针要**事后拟合**（而不是只靠训练时联合的辅助损失）：
  * 联合探针与动作损失抢梯度，测出来的是"训练逼它编码了多少"；
  * 事后读出冻结主干、只拟合读出头，测出来的是"状态里到底有没有这份信息"。
后者才是"连续事件记忆"该用的证据，前者是有用的辅助信号但会低估。
两个都报，各说各的话。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..config import Config, default_config
from ..core.architecture import NModel
from ..learning import datasets as D
from ..learning.trainer import TrainResult, _assemble, evaluate, make_data


def collect_state(model: NModel, rows: list[dict[str, Any]], *, t: int,
                  device: torch.device, batch_size: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    """取第 t 步的池化块状态 ctx 与该行的目标格编码。"""
    xs, cs = [], []
    model.eval()
    with torch.no_grad():
        for s in range(0, len(rows), batch_size):
            b = _assemble(rows[s:s + batch_size], device)
            out = model.forward_sequence(b["obs"], b["actions"])
            xs.append(out["ctx"][:, min(t, out["ctx"].shape[1] - 1)].cpu())
            cs.append(b["codes"].cpu())
    return torch.cat(xs), torch.cat(cs)


def readout_probe(model: NModel, train_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]],
                  *, t: int, device: torch.device, hidden: int = 64, iters: int = 400,
                  lr: float = 8e-3, seed: int = 0, n_classes: int = 36) -> dict[str, float]:
    """冻结主干，只在状态上拟合一个两层层读出头，报告能否召回目标格。

    train/test 都报：train 高 test 低说明是记住样本而不是状态里真有序列信息。
    """
    Xtr, Ctr = collect_state(model, train_rows, t=t, device=device)
    Xte, Cte = collect_state(model, test_rows, t=t, device=device)
    g = torch.Generator().manual_seed(seed)
    probe = nn.Sequential(nn.Linear(int(Xtr.shape[1]), hidden), nn.GELU(), nn.Linear(hidden, n_classes))
    probe.to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(iters):
        idx = torch.randint(0, Xtr.shape[0], (min(64, Xtr.shape[0]),), generator=g)
        loss = F.cross_entropy(probe(Xtr[idx]), Ctr[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        tr_acc = float((probe(Xtr).argmax(-1) == Ctr).float().mean())
        te_acc = float((probe(Xte).argmax(-1) == Cte).float().mean())
    return {"recall_train": tr_acc, "recall_test": te_acc, "chance": 1.0 / n_classes,
            "probe_loss": float(loss.detach()), "n_test": float(Xte.shape[0]), "step": float(t)}


def memory_span_curve(cfg: Config | None, model: NModel, *, task_seed: int,
                      steps: tuple[int, ...] = (2, 6, 12, 20, 30), device: torch.device,
                      n_rows: int = 24) -> list[dict[str, float]]:
    """逐步拉长"遮挡后还要走多少步"，看召回率随干扰步数怎么掉。

    同一条轨迹上取多个时间点，因此各点用的是同一份状态演化，
    差别只在"距离目标最后可见已经过了多少步"。
    """
    cfg = cfg or default_config()
    task = D.GridTask(goal_visible_steps=2, seed=task_seed)
    data = make_data(cfg, task, seed=task_seed)
    rows_tr, rows_te = data["train"][:n_rows], data["test"][:n_rows]
    out = []
    for t in steps:
        if t >= cfg.learning.traj_len:
            continue
        p = readout_probe(model, rows_tr, rows_te, t=t, device=device)
        ev = evaluate(model, nn.Identity(), rows_te, device=device, occluded_only=True)
        p.update({"distractor_steps": float(t), "policy_acc_at_t": _acc_at_step(model, rows_te, t, device)})
        out.append(p)
    return out


def _acc_at_step(model: NModel, rows: list[dict[str, Any]], t: int, device: torch.device) -> float:
    hits = tot = 0
    model.eval()
    with torch.no_grad():
        for s in range(0, len(rows), 8):
            b = _assemble(rows[s:s + 8], device)
            out = model.forward_sequence(b["obs"], b["actions"])
            pred = out["action_logits"][:, min(t, out["action_logits"].shape[1] - 1)].argmax(-1)
            tgt = b["actions"][:, min(t, b["actions"].shape[1] - 1)]
            hits += int((pred == tgt).sum().item())
            tot += int(pred.numel())
    return hits / max(1, tot)


def decision_metrics(model: NModel, rows: list[dict[str, Any]], *, device: torch.device) -> dict[str, float]:
    """任务 3：动作决策。分"目标可见段/遮挡段/撞墙后一步"三档看，不给一个平均数糊过去。"""
    base = evaluate(model, nn.Identity(), rows, device=device, occluded_only=False)
    model.eval()
    vis_n = vis_hit = post_coll = post_coll_hit = 0
    with torch.no_grad():
        for s in range(0, len(rows), 8):
            b = _assemble(rows[s:s + 8], device)
            out = model.forward_sequence(b["obs"], b["actions"])
            pred = out["action_logits"].argmax(-1)
            vis = b["obs"][:, :, 24] < 0.5
            vis_hit += int((pred[vis] == b["actions"][vis]).sum().item())
            vis_n += int(vis.sum().item())
            coll = torch.zeros_like(b["obs"][:, :, 25], dtype=torch.bool)
            coll[:, 1:] = b["obs"][:, :-1, 25] > 0.5               # 撞墙后的那一步
            post_coll_hit += int((pred[coll] == b["actions"][coll]).sum().item())
            post_coll += int(coll.sum().item())
    res = {"accuracy_all": base["accuracy"], "accuracy_visible": vis_hit / max(1, vis_n),
           "n_visible": float(vis_n), "accuracy_after_collision": post_coll_hit / max(1, post_coll),
           "n_after_collision": float(post_coll)}
    res["accuracy_occluded"] = evaluate(model, nn.Identity(), rows, device=device,
                                        occluded_only=True)["accuracy_occluded"]
    return res


def budget_difficulty_report(model: NModel, *, cfg: Config | None = None,
                             device: torch.device = torch.device("cpu"),
                             T: int = 16, n_each: int = 20, seed: int = 7) -> dict[str, Any]:
    """任务 5：混合难易样本，检查激活块数是否真的随难度变化，并量真实耗时。

    这里**不假设**"误差高就该多算"是好事：报告里同时给出
    "固定 k" 与 "误差门控 k" 两种模式下的准确率与延迟，谁更好看数据。
    """
    cfg = cfg or default_config()
    easy = [D.make_trajectory(D.GridTask(name="easy", goal_visible_steps=T, obstacles=(), seed=seed),
                             np.random.default_rng(7000 + i), T) for i in range(n_each)]
    hard = [D.make_trajectory(D.GridTask(name="hard", goal_visible_steps=1,
                                         obstacles=((2, 2), (2, 3), (3, 2), (4, 1)), seed=seed + 1),
                             np.random.default_rng(8000 + i), T) for i in range(n_each)]
    rows = {k: [_row(tr) for tr in v] for k, v in (("easy", easy), ("hard", hard))}
    out: dict[str, Any] = {"T": T, "n_each": n_each}
    for mode, fixed_k in (("gated", None), ("fixed_k1", 1), ("fixed_kall", cfg.model.n_blocks)):
        stats = {}
        for split, rws in rows.items():
            k_list, acc, ms = _run_with_k(model, rws, device, k=fixed_k)
            stats[split] = {"mean_k": round(float(np.mean(k_list)), 3),
                            "accuracy": round(acc, 4), "mean_step_ms": round(ms, 3)}
        out[mode] = stats
    return out


def _row(tr: Any) -> dict[str, Any]:
    return {"obs": np.asarray(tr.obs, dtype=np.float32), "actions": np.asarray(tr.actions, dtype=np.int64),
            "goal_code": int(tr.goal[0]) * 6 + int(tr.goal[1]), "len": int(tr.obs.shape[0])}


def _run_with_k(model: NModel, rows: list[dict[str, Any]], device: torch.device,
                k: int | None) -> tuple[list[int], float, float]:
    import time

    model.eval()
    ks, hit, tot, times = [], 0, 0, []
    with torch.no_grad():
        for r in rows:
            b = _assemble([r], device)
            t0 = time.perf_counter()
            out = model.forward_sequence(b["obs"], b["actions"], k=k)
            times.append((time.perf_counter() - t0) * 1000.0)
            ks.append(out["blocks_touched"] / max(1, b["obs"].shape[1]))
            pred = out["action_logits"].argmax(-1)
            hit += int((pred == b["actions"]).sum().item())
            tot += int(pred.numel())
    return ks, hit / max(1, tot), float(np.mean(times))


@dataclass
class TaskSuite:
    """把任务 1/2/3/5 的结果收在一起，供报告与消融复用。"""

    results: dict[str, Any] = field(default_factory=dict)

    def run(self, model: NModel, probe: nn.Module, *, cfg: Config | None = None,
            device: torch.device = torch.device("cpu"), task_seed: int = 11,
            T: int = 16) -> dict[str, Any]:
        cfg = cfg or default_config()
        data = make_data(cfg, D.GridTask(goal_visible_steps=4, seed=task_seed), seed=task_seed)
        self.results = {
            "state_prediction": evaluate(model, probe, data["test"], device=device, occluded_only=True),
            "decision": decision_metrics(model, data["test"], device=device),
            "memory_recall_visible_step": readout_probe(model, data["train"], data["test"], t=1, device=device),
            "memory_recall_occluded": readout_probe(model, data["train"], data["test"],
                                                    t=cfg.learning.traj_len - 1, device=device),
            "memory_span_curve": memory_span_curve(cfg, model, task_seed=task_seed, device=device),
            "budget": budget_difficulty_report(model, cfg=cfg, device=device, T=T),
        }
        return self.results
