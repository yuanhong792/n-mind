"""BlockRouter：以块为单位的 Top-K 动态路由。

要点（对应指令第 3、4 节）：
- 硬 Top-K 的离散索引没有梯度，这里用直通估计 + 软权重把梯度送回打分网络，
  不假装离散选择天然可微。
- 预测误差 E_t 与不确定性 U_t 进入打分，作为"按需增加计算"的假说载体；
  该假说是否成立由 evaluation/ablations.py 的消融实验判定，不预设结论。
- 负载偏置（load bias）用于 anti-collapse，避免永远只选中同一批块。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn


@dataclass
class RouteResult:
    indices: torch.Tensor          # [B, k] 被激活的块编号
    weights: torch.Tensor          # [B, k] 选中块上的归一化软权重（有梯度）
    scores: torch.Tensor           # [B, N] 加入负载偏置前的打分
    probs: torch.Tensor            # [B, N]  softmax 分布（用于熵与均衡损失）
    k: int
    stats: dict = field(default_factory=dict)


class BlockRouter(nn.Module):
    def __init__(
        self,
        z_dim: int,
        d_h: int,
        d_w: int,
        n_blocks: int,
        hidden: int = 64,
        balance_lr: float = 0.05,
        balance_max: float = 2.0,
    ) -> None:
        super().__init__()
        self.n_blocks = n_blocks
        self.balance_lr = balance_lr
        self.balance_max = balance_max

        # 每块状态用一个标量摘要进入打分：摘要 = 状态 L2 范数 / sqrt(d_h)。
        in_dim = z_dim + d_w + n_blocks + 2
        self.score_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, n_blocks)
        )
        # 非参数化的负载均衡偏置（随负载移动，不参与梯度）
        self.register_buffer("load_bias", torch.zeros(n_blocks))

    # ------------------------------------------------------------------ 打分

    def scores(
        self,
        z: torch.Tensor,
        h: torch.Tensor,
        w: torch.Tensor,
        err: torch.Tensor,
        unc: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz = z.shape[0]
        h_summary = h.pow(2).mean(dim=-1)                     # [B, N]
        err = err.reshape(bsz, -1)[:, :1]
        unc = unc.reshape(bsz, -1)[:, :1]
        x = torch.cat([z, w, h_summary, err, unc], dim=-1)
        raw = self.score_mlp(x)
        return raw, torch.softmax(raw + self.load_bias.unsqueeze(0), dim=-1)

    # ------------------------------------------------------------------ 路由

    def route(
        self,
        z: torch.Tensor,
        h: torch.Tensor,
        w: torch.Tensor,
        err: torch.Tensor,
        unc: torch.Tensor,
        k: int,
    ) -> RouteResult:
        k = int(max(1, min(k, self.n_blocks)))
        raw, probs = self.scores(z, h, w, err, unc)
        top_probs, indices = torch.topk(probs, k, dim=-1)      # [B, k]
        weights = top_probs / top_probs.sum(-1, keepdim=True).clamp_min(1e-9)

        mask = torch.zeros_like(probs).scatter_(-1, indices, 1.0)
        active = mask
        load = mask.mean(dim=0) if mask.shape[0] else torch.zeros(self.n_blocks)

        stats = {
            "k": k,
            "activation_ratio": float(mask.sum().item()) / float(mask.numel()),
            "load": [round(float(v), 4) for v in load.detach().tolist()],
            "top_score": round(float(raw.max(dim=-1).values.detach().mean().item()), 4),
            "entropy": round(
                float(-(probs * (probs.clamp_min(1e-12)).log()).sum(-1).mean().detach().item()), 4
            ),
        }
        return RouteResult(
            indices=indices,
            weights=weights,
            scores=raw,
            probs=probs,
            k=k,
            stats=stats,
        )

    # ------------------------------------------------------------------ 损失与均衡

    @staticmethod
    def entropy_loss(probs: torch.Tensor) -> torch.Tensor:
        logp = probs.clamp_min(1e-12).log()
        return -(probs * logp).sum(-1).mean()

    def update_load_bias(self, load: torch.Tensor, target: float | None = None) -> dict:
        """按"低于目标负载就抬、高于就压"的符号规则移动偏置，带硬边界。

        这里的 load 是"每样本平均选中次数"（k 个块 ⇒ 总和为 k），不是概率分布，
        所以目标必须跟着 load 的量纲走：默认取均匀分布时的期望份额 sum(load)/N。
        写死 1/N 会在 k>1 时把所有块都判成"过载"，反塌缩机制悄悄失效。
        """
        if target is None:
            target = float(load.sum()) / float(self.n_blocks)
        with torch.no_grad():
            delta = self.balance_lr * torch.sign(torch.tensor(float(target)) - load)
            self.load_bias = (self.load_bias + delta).clamp(-self.balance_max, self.balance_max)
        return {
            "bias": [round(float(v), 4) for v in self.load_bias.tolist()],
            "target": round(float(target), 4),
            "collapsed": bool((load < 1e-6).float().mean().item() > 0.5),
        }
