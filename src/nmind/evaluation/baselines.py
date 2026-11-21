"""对照基线：与 N-Cell 比参数量与比 CPU 时间。

三个基线，都是真实现（不是"照抄论文里的数字"）：

  dense      —— 同一份 NModel，关掉路由、每步全激活。它把"稀疏主干的计算节省"
               与"路由/预测器带来的额外开销"干净地分离开，是最关键的对照组。
  gru        —— 多层 nn.GRU：参数量接近的常规模型，用来回答"这套东西比 GRU 强在哪"。
  shared     —— 前 S 个块跨任务共享、其余块每任务私有。这一档直接测的是
               "共享主干能不能省参数"，而不是靠外部假设。

transformer 基线暂未实现（见 docs/STATUS.md 未完成清单）：宁可少一个对照组，
也不要一个没验证过的实现来充当基线。

公平性口径全部写进返回体：参数量、每步激活元素数、训练后指标、CPU 单序列耗时。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..config import Config, default_config
from ..core.architecture import NModel
from ..learning import datasets as D
from ..learning.trainer import _assemble, evaluate, make_data


# ---------------------------------------------------------------------- 主干定义

class GruBackbone(nn.Module):
    """多层 GRU + 动作头 + 目标读出。没有路由、没有工作空间、没有预测器。"""

    def __init__(self, input_dim: int, hidden: int, n_layers: int, n_actions: int,
                 n_classes: int = 36) -> None:
        super().__init__()
        self.embed = nn.Linear(input_dim, hidden)
        self.gru = nn.GRU(hidden, hidden, num_layers=n_layers, batch_first=True)
        self.head = nn.Linear(hidden, n_actions)
        self.readout = nn.Linear(hidden, n_classes)

    def forward_sequence(self, obs: torch.Tensor, actions: torch.Tensor, **kw: Any) -> dict[str, Any]:
        h, _ = self.gru(F.gelu(self.embed(obs)))
        return {"logits": self.head(h), "ctx": h, "readout_logits": self.readout(h),
                "blocks_touched": 0, "route_stats": {}}

    def active_elements(self, obs: torch.Tensor) -> int:
        """每步实际读写的参数量：GRU 是"全激活"的，所以等于全部参数。"""
        return sum(p.numel() for p in self.parameters())


class SharedTrunkCell(nn.Module):
    """块级主干共享：前 shared 块所有任务都用，后 private 块按任务 id 各一份。

    参数量 = 共享块 + 任务数 × 私有块。这正是"共享能不能省参数"的直接测法：
    固定总块数，比较 full-private / shared / full-shared 三档在同样数据下的表现。
    """

    def __init__(self, z_dim: int, d_h: int, n_blocks: int, shared: int, private: int,
                 n_tasks: int, n_actions: int, n_classes: int = 36) -> None:
        super().__init__()
        if shared + private > n_blocks:
            raise ValueError(f"shared({shared})+private({private}) 不能超过 n_blocks({n_blocks})")
        self.shared, self.private, self.n_tasks = int(shared), int(private), int(n_tasks)
        self.enc = nn.Linear(z_dim, d_h)
        if shared:
            self.trunk = nn.GRUCell(d_h, d_h)
        if private:
            self.priv = nn.ModuleList([nn.GRUCell(d_h, d_h) for _ in range(n_tasks * private)])
        self.head = nn.Linear(d_h, n_actions)
        self.readout = nn.Linear(d_h, n_classes)

    def forward_sequence(self, obs: torch.Tensor, actions: torch.Tensor, *,
                         task_id: int = 0, **kw: Any) -> dict[str, Any]:
        b, T, _ = obs.shape
        d_h = self.trunk.hidden_size if self.shared else self.priv[0].hidden_size
        h = torch.zeros(b, self.shared + self.private, d_h, device=obs.device, dtype=obs.dtype)
        outs, reads = [], []
        for t in range(T):
            x = F.gelu(self.enc(obs[:, t]))
            nxt = []
            if self.shared:
                # 共享槽位是同一个 GRUCell 对 [b*shared] 批量算：输入 x 必须按槽位复制，
                # 否则 input batch b 与 hidden batch b*shared 不匹配
                x_rep = x.unsqueeze(1).expand(-1, self.shared, -1).reshape(-1, x.shape[-1])
                nxt.append(self.trunk(x_rep, h[:, :self.shared].reshape(-1, x.shape[-1]))
                           .view(b, self.shared, -1))
            for j in range(self.private):
                cell = self.priv[task_id * self.private + j]
                nxt.append(cell(x, h[:, self.shared + j].reshape(-1, x.shape[-1]))
                           .view(b, 1, -1))
            h = torch.cat(nxt, dim=1)
            pooled = h.mean(dim=1)
            outs.append(self.head(pooled))
            reads.append(self.readout(pooled))
        return {"logits": torch.stack(outs, dim=1), "ctx": torch.stack(reads, dim=1),
                "readout_logits": torch.stack(reads, dim=1),
                "blocks_touched": (self.shared + self.private) * b, "route_stats": {}}


def build_dense(cfg: Config, *, seed: int, device: torch.device) -> NModel:
    torch.manual_seed(seed)
    return NModel(cfg.model, n_actions=D.N_ACTIONS, mem_dim=16, execution="dense",
                  use_router=False).to(device).eval()


# ---------------------------------------------------------------------- 统一训练与评测

@dataclass
class BaselineRow:
    name: str
    params: int
    active_elements: int
    accuracy: float
    accuracy_occluded: float
    recall: float
    chance: float
    train_s: float
    seq_ms: float
    note: str


def fit(model: nn.Module, *, ce_key: str = "logits", readout_key: str = "readout_logits",
        rows: list[dict[str, Any]], cfg: Config, device: torch.device, epochs: int = 8,
        lr: float = 3e-3, task_id: int = 0, seed: int = 0) -> dict[str, float]:
    """把三种主干用同一套 CE + 读出损失训起来；优化器、batch、步数一致。"""
    torch.manual_seed(seed)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(seed + 5)
    bs = max(2, cfg.learning.batch_size)
    steps = cfg.learning.n_updates
    model.train()
    losses = []
    for _ep in range(epochs):
        for _st in range(steps):
            pick = [int(i) for i in rng.integers(0, len(rows), size=bs)]
            b = _assemble([rows[i] for i in pick], device)
            kw = {"task_id": task_id} if isinstance(model, SharedTrunkCell) else {}
            out = model.forward_sequence(b["obs"], b["actions"], **kw)
            # NModel 返回 action_logits，GRU/SharedTrunk 返回 logits：两键自适应，避免单边假死
            lg = out[ce_key] if ce_key in out else out["action_logits"]
            loss = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), b["actions"].reshape(-1))
            if out.get(readout_key) is not None:
                codes = b["codes"].repeat_interleave(b["obs"].shape[1])
                loss = loss + 0.25 * F.cross_entropy(out[readout_key].reshape(-1, out[readout_key].shape[-1]), codes)
            if not torch.isfinite(loss):
                raise RuntimeError("基线训练出现非有限损失")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            losses.append(float(loss.detach()))
    model.eval()
    return {"loss_head": losses[0], "loss_tail": losses[-1],
            "loss_drop": losses[0] - losses[-1], "steps": len(losses)}


def score(model: nn.Module, rows: list[dict[str, Any]], device: torch.device, *,
          task_id: int = 0) -> dict[str, float]:
    model.eval()
    hit = tot = occ_hit = occ_n = r_hit = r_n = 0
    with torch.no_grad():
        for s in range(0, len(rows), 8):
            b = _assemble(rows[s:s + 8], device)
            kw = {"task_id": task_id} if isinstance(model, SharedTrunkCell) else {}
            out = model.forward_sequence(b["obs"], b["actions"], **kw)
            lg = out["logits"] if "logits" in out else out["action_logits"]
            pred = lg.argmax(-1)
            hit += int((pred == b["actions"]).sum().item())
            tot += int(pred.numel())
            occ = b["obs"][:, :, 24] > 0.5
            occ_hit += int((pred[occ] == b["actions"][occ]).sum().item())
            occ_n += int(occ.sum().item())
            rl = out.get("readout_logits")
            if rl is not None:
                codes = b["codes"].repeat_interleave(b["obs"].shape[1])
                r_hit += int((rl.argmax(-1).reshape(-1) == codes).sum().item())
                r_n += int(codes.numel())
    res = {"accuracy": hit / max(1, tot), "n": float(tot)}
    if occ_n:
        res["accuracy_occluded"] = occ_hit / occ_n
    if r_n:
        res["recall"] = r_hit / r_n
    return res


def measure_seq_ms(model: nn.Module, rows: list[dict[str, Any]], device: torch.device,
                   *, task_id: int = 0, repeats: int = 3) -> float:
    model.eval()
    b = _assemble(rows[:4], device)
    kw = {"task_id": task_id} if isinstance(model, SharedTrunkCell) else {}
    best = float("inf")
    with torch.no_grad():
        for _ in range(repeats):
            t0 = time.perf_counter()
            model.forward_sequence(b["obs"], b["actions"], **kw)
            best = min(best, (time.perf_counter() - t0) * 1000.0)
    return best


def run_baselines(cfg: Config | None = None, *, seed: int | None = None, epochs: int = 8,
                  device: str = "auto", n_blocks: int | None = None,
                  quiet: bool = False) -> dict[str, Any]:
    """在相同数据、相同损失、相同优化预算下比较 dense / gru / shared-trunk 三种主干。"""
    from ..runtime.hardware import detect

    cfg = cfg or default_config()
    seed = int(cfg.runtime.seed if seed is None else seed)
    dev_obj, report = detect(device, torch_threads=cfg.runtime.torch_threads)
    dev = dev_obj if isinstance(dev_obj, torch.device) else torch.device(str(dev_obj))
    task = D.GridTask(goal_visible_steps=4, seed=seed)
    data = make_data(cfg, task, seed=seed)

    rows_out: list[BaselineRow] = []
    hidden = cfg.model.block_hidden

    # 1) 稠密 N-Cell（路由关闭 ⇒ 全激活）
    dense = build_dense(cfg, seed=seed, device=dev) if not n_blocks else None
    if dense is not None:
        log = fit(dense, rows=data["train"], cfg=cfg, device=dev, seed=seed)
        sc = score(dense, data["test"], dev)
        ms = measure_seq_ms(dense, data["test"], dev)
        rows_out.append(BaselineRow("dense_n_cell", sum(p.numel() for p in dense.parameters()),
                                    sum(p.numel() for p in dense.blocks.parameters()),
                                    sc["accuracy"], sc.get("accuracy_occluded", float("nan")),
                                    sc.get("recall", float("nan")), 1 / 36,
                                    float(log["loss_drop"]), ms,
                                    "同架构关闭路由：全块激活"))

    # 2) GRU 主干
    gru = GruBackbone(cfg.model.event_numeric_dim, hidden, 2, D.N_ACTIONS).to(dev)
    log = fit(gru, rows=data["train"], cfg=cfg, device=dev, seed=seed)
    sc = score(gru, data["test"], dev)
    rows_out.append(BaselineRow("gru_2layer", sum(p.numel() for p in gru.parameters()),
                                gru.active_elements(None), sc["accuracy"],
                                sc.get("accuracy_occluded", float("nan")), sc.get("recall", float("nan")),
                                1 / 36, float(log["loss_drop"]), measure_seq_ms(gru, data["test"], dev),
                                "无路由/无工作空间/无预测器；损失只含 CE+读出，与 N-Cell 主损失不同，见 LIMITATIONS"))

    # 3) 共享主干：2 个任务，比较 full-private / 混合 / full-shared
    shared_rows = []
    for label, sh, pr in (("private_only", 0, 4), ("mixed", 4, 2), ("shared_only", 6, 0)):
        net = SharedTrunkCell(cfg.model.event_numeric_dim, hidden, 8, sh, pr,
                              n_tasks=2, n_actions=D.N_ACTIONS).to(dev)
        log = fit(net, rows=data["train"], cfg=cfg, device=dev, seed=seed, task_id=0)
        sc = score(net, data["test"], dev, task_id=0)
        shared_rows.append({"variant": label, "shared_blocks": sh, "private_blocks": pr,
                            "params": sum(p.numel() for p in net.parameters()),
                            "loss_drop": round(log["loss_drop"], 4),
                            "accuracy": round(sc["accuracy"], 4),
                            "accuracy_occluded": round(sc.get("accuracy_occluded", float("nan")), 4),
                            "recall": round(sc.get("recall", float("nan")), 4),
                            "seq_ms": round(measure_seq_ms(net, data["test"], dev, task_id=0), 3)})

    return {"hardware": report.as_dict(), "seed": seed,
            "task": D.task_to_dict(task), "dataset_hash": data["dataset_hash"],
            "rows": [r.__dict__ for r in rows_out], "shared_trunk": shared_rows,
            "params_total": dense.n_params()["total"] if dense else None,
            "note": "transformer 基线未实现；本表结论仅在已实现的三个主干之间成立"}
