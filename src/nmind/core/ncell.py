"""N-Cell：门控循环状态单元（块级稀疏激活）。

形式化定义（与指令第 3 节一致）：

    g = sigmoid(W_g · concat(z, h_prev, w) + b_g)
    c = tanh(W_z·z + W_h·h_prev + W_w·w + b_c)
    h' = (1 - g) ⊙ h_prev + g ⊙ c
    h  = h'  若该块本步被激活，否则 h 保持 h_prev

张量形状约定（全部 batch-first）：
    z        [B, z_dim]        事件潜状态
    h_prev   [B, N, d_h]       N 个块的旧状态
    w        [B, d_w]          工作空间读出向量
    active   [B, N] float(0/1) 块激活掩码（稀疏掩码）
    返回     h_new [B, N, d_h]

本文件刻意与 GRU 的数学相似性保持一致，不宣称新的循环单元数学。
创新假说在"块稀疏状态复用 + 预测误差门控 + 共享工作空间"这一组合上，见 docs/ARCHITECTURE.md。
"""

from __future__ import annotations

import torch
from torch import nn


class NCell(nn.Module):
    """N 个块各自的门控循环状态转移；支持稠密与稀疏两种等价执行方式。"""

    def __init__(self, z_dim: int, d_h: int, d_w: int, n_blocks: int, *,
                 use_reset_gate: bool = False) -> None:
        super().__init__()
        self.use_reset_gate = bool(use_reset_gate)
        self.z_dim = z_dim
        self.d_h = d_h
        self.d_w = d_w
        self.n_blocks = n_blocks

        gate_in = z_dim + d_h + d_w
        # 每个块一份独立权重：这是"块"的含义，也是稀疏能省算力的前提。
        self.W_g = nn.Parameter(torch.empty(n_blocks, gate_in, d_h))
        self.b_g = nn.Parameter(torch.zeros(n_blocks, d_h))
        self.W_z = nn.Parameter(torch.empty(n_blocks, z_dim, d_h))
        self.W_h = nn.Parameter(torch.empty(n_blocks, d_h, d_h))
        self.W_w = nn.Parameter(torch.empty(n_blocks, d_w, d_h))
        self.b_c = nn.Parameter(torch.zeros(n_blocks, d_h))
        # New candidates can add a GRU-like reset gate; legacy state_dict keys stay identical.
        if self.use_reset_gate:
            self.W_r = nn.Parameter(torch.empty(n_blocks, gate_in, d_h))
            self.b_r = nn.Parameter(torch.ones(n_blocks, d_h))
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """按"每块权重矩阵"的 fan_in 做均匀初始化。

        三维参数 [N, in, out] 不能用 xavier_uniform_ 直接套（它会拿前两维算 fan），
        所以这里显式按 in+out 计算边界，逐块等价于 Linear 的默认初始化。
        """
        for w in ((self.W_g, self.W_z, self.W_h, self.W_w, self.W_r)
                  if self.use_reset_gate else (self.W_g, self.W_z, self.W_h, self.W_w)):
            fan_in, fan_out = w.shape[1], w.shape[2]
            bound = 1.0 / (fan_in + fan_out) ** 0.5
            nn.init.uniform_(w, -bound, bound)
        # b_g 保持 0：sigmoid(0)=0.5，状态起步既不锁死也不乱跳。
        nn.init.constant_(self.b_g, 0.0)
        nn.init.constant_(self.b_c, 0.0)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @staticmethod
    def initial_state(batch: int, n_blocks: int, d_h: int, device=None) -> torch.Tensor:
        return torch.zeros(batch, n_blocks, d_h, device=device)

    # ------------------------------------------------------------------ 数学核

    def _propagate(self, z: torch.Tensor, h: torch.Tensor, w: torch.Tensor):
        """稠密计算所有块的候选状态。z/w 已按块广播，形状 [B, N, *]。"""
        concat = torch.cat([z, h, w], dim=-1)
        g = torch.sigmoid(torch.einsum("bni,nid->bnd", concat, self.W_g) + self.b_g)
        h_candidate = h
        if self.use_reset_gate:
            reset = torch.sigmoid(torch.einsum("bni,nid->bnd", concat, self.W_r) + self.b_r)
            h_candidate = reset * h
        c = torch.tanh(
            torch.einsum("bni,nid->bnd", z, self.W_z)
            + torch.einsum("bni,nid->bnd", h_candidate, self.W_h)
            + torch.einsum("bni,nid->bnd", w, self.W_w)
            + self.b_c
        )
        return (1.0 - g) * h + g * c, g

    def forward(
        self,
        z: torch.Tensor,
        h_prev: torch.Tensor,
        w: torch.Tensor,
        active: torch.Tensor,
        soft: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """一步状态更新。

        active: [B, N] 的 0/1 掩码。soft 若给出，则用直通估计（前向按硬掩码、反向按软掩码），
        这样 Top-K 的离散选择不会假装自己有完整梯度。
        返回 (h_new [B,N,d_h], g [B,N,d_h], mask [B,N], probs [B,N])。
        """
        if z.dim() != 2 or z.shape[1] != self.z_dim:
            raise ValueError(f"z 形状应为 [B, {self.z_dim}]，实际 {tuple(z.shape)}")
        if h_prev.shape[1] != self.n_blocks or h_prev.shape[2] != self.d_h:
            raise ValueError(f"h_prev 形状应为 [B, {self.n_blocks}, {self.d_h}]，实际 {tuple(h_prev.shape)}")
        if w.shape[-1] != self.d_w:
            raise ValueError(f"w 形状应为 [B, {self.d_w}]，实际 {tuple(w.shape)}")
        if active.shape != h_prev.shape[:2]:
            raise ValueError(f"active 形状应为 [B, N]，实际 {tuple(active.shape)}")

        z_exp = z[:, None, :].expand(-1, self.n_blocks, -1)
        w_exp = w[:, None, :].expand(-1, self.n_blocks, -1)
        h_prop, g = self._propagate(z_exp, h_prev, w_exp)

        mask = active.to(h_prop.dtype).unsqueeze(-1)
        if soft is not None and self.training:
            # straight-through：数值上仍用硬掩码，梯度路径走软掩码。
            softm = soft.unsqueeze(-1)
            mask = softm + (mask - softm).detach()
        h_new = (1.0 - mask) * h_prev + mask * h_prop
        denom = active.sum(dim=-1, keepdim=True).clamp_min(1.0)
        probs = active.to(h_prop.dtype) / denom          # 供负载均衡统计：选中块之间均分概率质量
        return h_new, g, mask.squeeze(-1), probs

    # ------------------------------------------------------------------ 稀疏执行

    def forward_sparse(
        self,
        z: torch.Tensor,
        h_prev: torch.Tensor,
        w: torch.Tensor,
        indices: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """只对 [B, k] 指定块做计算，未选中的块原样保留。

        与 forward() 的硬掩码路径在数学上等价（用 test_ncell.py 钉死），
        但**只计算被选中的 (样本, 块) 对**，这才是稀疏执行省算力的地方。
        返回 (h_new, g_sel [P,d_h], mask [B,N], probs [B,N], 实际计算的块对数)。
        """
        bsz, k = indices.shape
        rows = torch.arange(bsz, device=indices.device)[:, None].expand(-1, k).reshape(-1)
        flat_blocks = indices.reshape(-1)

        z_sel = z.index_select(0, rows)
        h_sel = h_prev[rows, flat_blocks]
        w_sel = w.index_select(0, rows)

        concat = torch.cat([z_sel, h_sel, w_sel], dim=-1).unsqueeze(1)
        Wg = self.W_g.index_select(0, flat_blocks)
        g = torch.sigmoid(torch.bmm(concat, Wg).squeeze(1) + self.b_g.index_select(0, flat_blocks))
        h_candidate = h_sel
        if self.use_reset_gate:
            Wr = self.W_r.index_select(0, flat_blocks)
            reset = torch.sigmoid(torch.bmm(concat, Wr).squeeze(1)
                                  + self.b_r.index_select(0, flat_blocks))
            h_candidate = reset * h_sel
        # 三处都用 bmm 显式带 batch 维：matmul 对 2D×3D 会做广播，
        # 曾把 [P,d] 静默放大成 [P,P,d]，形状错了也不报错，属于最难查的那类 bug。
        c = torch.tanh(
            torch.bmm(z_sel.unsqueeze(1), self.W_z.index_select(0, flat_blocks)).squeeze(1)
            + torch.bmm(h_candidate.unsqueeze(1), self.W_h.index_select(0, flat_blocks)).squeeze(1)
            + torch.bmm(w_sel.unsqueeze(1), self.W_w.index_select(0, flat_blocks)).squeeze(1)
            + self.b_c.index_select(0, flat_blocks)
        )
        h_upd = (1.0 - g) * h_sel + g * c
        if weights is not None:
            # STE（同稠密路径 mask 的做法）：把权重变量本身变成"前向值=1、∂/∂wk=1"，
            # 插值 wk_ste*h_upd + (1-wk_ste)*h_sel 的前向值 = h_upd（硬采纳，精确），
            # 反向 ∂h/∂wk = h_upd - h_sel → 路由器拿到任务梯度；块梯度不被缩放。
            wk = weights.reshape(-1).unsqueeze(-1).to(h_upd.dtype)
            wk_ste = wk + (1.0 - wk).detach()
            h_upd = wk_ste * h_upd + (1.0 - wk_ste) * h_sel

        h_new = h_prev.clone()
        h_new[rows, flat_blocks] = h_upd
        mask = torch.zeros(bsz, self.n_blocks, device=indices.device, dtype=h_prev.dtype)
        mask.scatter_(-1, indices, 1.0)
        probs = mask / float(max(1, k))
        return h_new, g, mask, probs, int(rows.numel())
