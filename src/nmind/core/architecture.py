"""NModel：把内核各件按指令第 6 节的运行逻辑串成一次可复现的纵向闭环。

    observe -> encode -> update local workspace -> select blocks ->
    retrieve memory -> local decision -> validate -> (optional cloud) ->
    record episode -> consolidation

这里只负责神经计算图本身；记忆读写与云端策略在 runtime/engine.py。
消融开关（不用路由 / 不用误差门控 / 不用工作空间）直接内建，
这样基线与消融跑的是同一份代码路径，不靠外部脚本改行为。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

from ..config import ModelCfg
from .decision_head import ActionHead
from .encoder import EventEncoder
from .ncell import NCell
from .predictor import WorldPredictor
from .router import BlockRouter
from .workspace import GlobalWorkspace


@dataclass
class NState:
    h: torch.Tensor
    ws: dict[str, torch.Tensor]
    last_z_hat: torch.Tensor | None = None
    tick: int = 0
    n_updates: int = 0

    @property
    def batch(self) -> int:
        return int(self.h.shape[0])


@dataclass
class StepOutput:
    z: torch.Tensor
    z_hat: torch.Tensor
    error: torch.Tensor
    uncertainty: torch.Tensor
    decision: dict[str, Any]
    ctx: torch.Tensor
    route_probs: torch.Tensor
    route_stats: dict[str, Any] = field(default_factory=dict)
    ws_pressure: dict[str, Any] = field(default_factory=dict)
    overflow: list[dict] = field(default_factory=list)
    k: int = 0
    depth: int = 1
    blocks_touched: int = 0


class NModel(nn.Module):
    def __init__(
        self,
        cfg: ModelCfg,
        n_actions: int = 4,
        mem_dim: int = 16,
        execution: str = "sparse",
        use_router: bool = True,
        use_error_gating: bool = True,
        use_workspace: bool = True,
        top_k: int | None = None,
        router_ste: bool = False,
    ) -> None:
        super().__init__()
        if execution not in ("sparse", "dense"):
            raise ValueError("execution 只能是 'sparse' 或 'dense'")
        self.cfg = cfg
        self.n_actions = n_actions
        self.mem_dim = mem_dim
        self.execution = execution
        self.use_router = use_router
        self.use_error_gating = use_error_gating
        self.use_workspace = use_workspace
        self.top_k = int(top_k or cfg.top_k)
        # 稀疏模式下给路由器接上任务梯度的开关（M2 收尾实验 F10）。
        # 关闭时 route.weights 不参与前向（历史行为，router 只被均衡/熵正则训练）；
        # 打开时选中的块更新按 route.weights 插值，梯度经插值路径回传路由器（STE 风格）。
        self.router_ste = bool(router_ste)

        self.encoder = EventEncoder(
            numeric_dim=cfg.event_numeric_dim,
            text_dim=cfg.event_text_dim,
            z_dim=cfg.z_dim,
            hidden=cfg.encoder_hidden,
        )
        self.blocks = NCell(cfg.z_dim, cfg.block_hidden, cfg.workspace_dim, cfg.n_blocks)
        self.router = BlockRouter(
            z_dim=cfg.z_dim,
            d_h=cfg.block_hidden,
            d_w=cfg.workspace_dim,
            n_blocks=cfg.n_blocks,
            hidden=max(16, cfg.z_dim // 2),
        )
        self.workspace = GlobalWorkspace(
            z_dim=cfg.z_dim,
            d_w=cfg.workspace_dim,
            slots=cfg.workspace_slots,
        )
        self.predictor = WorldPredictor(
            z_dim=cfg.z_dim,
            action_dim=cfg.action_dim,
            d_w=cfg.workspace_dim,
            hidden=cfg.predictor_hidden,
        )
        self.head = ActionHead(cfg.z_dim, cfg.workspace_dim, mem_dim, n_actions)

        self.action_emb = nn.Embedding(n_actions, cfg.action_dim)
        self.block_to_ws = nn.Linear(cfg.block_hidden, cfg.workspace_dim)
        self.mem_summary_dim = mem_dim
        # 共享主干消融用：前 shared_trunk_blocks 个块在"共享"模式下被全任务复用
        self.register_buffer("block_load_count", torch.zeros(cfg.n_blocks))

    # ------------------------------------------------------------------ 状态

    def initial_state(self, batch: int = 1, device=None) -> NState:
        return NState(
            h=NCell.initial_state(batch, self.cfg.n_blocks, self.cfg.block_hidden, device),
            ws=self.workspace.initial_state(batch, device),
        )

    # ------------------------------------------------------------------ 单步

    @staticmethod
    def _indices_to_mask(indices: torch.Tensor, n_blocks: int) -> torch.Tensor:
        mask = torch.zeros(indices.shape[0], n_blocks, device=indices.device)
        mask.scatter_(-1, indices, 1.0)
        return mask

    def step(
        self,
        numeric: torch.Tensor,
        state: NState,
        action_prev: torch.Tensor,
        text_feat: torch.Tensor | None = None,
        mem: torch.Tensor | None = None,
        k: int | None = None,
        depth: int = 1,
    ) -> tuple[StepOutput, NState]:  # noqa: PLR0915 - 单步就是整条推理图，拆开反而失真
        bsz = numeric.shape[0]
        z = self.encoder(numeric, text_feat)
        if mem is None:
            mem = torch.zeros(bsz, self.mem_dim, device=z.device)
        if mem.shape[-1] != self.mem_dim:
            raise ValueError(f"mem 维度应为 {self.mem_dim}，实际 {mem.shape[-1]}")

        # 预测误差：上一拍的 z_hat 与本拍真实 z 比较。第一步无历史，误差取均值基准 0.5。
        if state.last_z_hat is not None:
            err = WorldPredictor.normalized_error(z, state.last_z_hat)
        else:
            err = torch.full((bsz, 1), 0.5, device=z.device)
        if not self.use_error_gating:
            err = torch.full_like(err, 0.5)          # 消融：门控看不到真实误差
        unc = err.detach().mean(dim=0, keepdim=True).expand_as(err)

        w_read = (
            self.workspace.read(state.ws, z)
            if self.use_workspace
            else SimpleNamespace(
                value=torch.zeros(bsz, self.cfg.workspace_dim, device=z.device),
                attn=torch.full(
                    (bsz, self.cfg.workspace_slots),
                    1.0 / self.cfg.workspace_slots,
                    device=z.device,
                ),
                occupied=state.ws["occupied"],
            )
        )

        k_eff = self.cfg.n_blocks if not self.use_router else int(k or self.top_k)
        route = self.router.route(z, state.h, w_read.value, err, unc, k=k_eff)

        # 反塌缩偏置只在**训练态**更新：推理态必须是无状态的纯函数，
        # 否则"同一权重 + 同一输入 → 同一输出"这条复现验收会被偷偷破坏。
        if self.training and self.use_router:
            load = torch.zeros(self.cfg.n_blocks, device=z.device)
            load.scatter_add_(0, route.indices.reshape(-1),
                              torch.ones_like(route.indices.reshape(-1), dtype=load.dtype))
            load = load / max(1.0, float(route.indices.numel()))
            with torch.no_grad():
                self.router.update_load_bias(load)

        # 执行块更新：稀疏与稠密两条路径必须等价（tests/test_ncell.py 有专门断言）
        depth = max(1, int(depth))
        h = state.h
        touched = 0
        if self.execution == "dense":
            mask = self._indices_to_mask(route.indices, self.cfg.n_blocks)
            # 软掩码保留梯度：NCell 内部用直通估计，前向按硬掩码、反向按软掩码。
            soft = torch.zeros_like(mask).scatter_(-1, route.indices, route.weights)
            for _ in range(depth):
                h, _g, _m, _p = self.blocks(z, h, w_read.value, mask, soft=soft)
            touched = int(mask.sum().item())
        else:
            # 硬选择语义与稠密掩码路径保持一致：选中即整体采纳 h'，不按软权重缩放。
            # route.weights 默认不参与前向（router_ste=False）：此时稀疏模式下路由器
            # 收不到任何任务梯度（gradient_diagnostics.json 实测 action/pred/probe→router 全 0），
            # 只有均衡/熵正则在训练它 —— 这是 D 臂失败的候选机制之一。
            # router_ste=True 时传入软权重：前向仍是硬选择加权的插值，梯度经插值回传。
            idx = route.indices
            w_sel = route.weights if self.router_ste else None
            for _d in range(depth):
                h, _g, _m, _p, n_pairs = self.blocks.forward_sparse(z, h, w_read.value, idx,
                                                                    weights=w_sel)
                touched += int(n_pairs)
            h = h.clone()

        # 把被激活块的更新后状态写入工作空间（竞争写入，不是直接覆盖）
        overflow: list[dict] = []
        ws_pressure = {}
        ws_state = state.ws
        if self.use_workspace:
            sel = route.indices.reshape(-1)
            rows = torch.arange(bsz, device=z.device)[:, None].expand(-1, route.indices.shape[1]).reshape(-1)
            payload = self.block_to_ws(h[rows, sel]).view(bsz, route.indices.shape[1], -1)
            wr = self.workspace.write(ws_state, payload)
            ws_state = {"content": wr.state["content"], "occupied": wr.occupied, "utility": wr.utility}
            ws_pressure = self.workspace.pressure(wr.occupied)
            overflow = wr.overflow

        a_idx = action_prev.long()
        if a_idx.dim() == 2 and a_idx.shape[-1] == 1:
            a_idx = a_idx.squeeze(-1)              # [B,1] 是最常见的误传，与 [B] 等价
        if a_idx.dim() != 1 or a_idx.shape[0] != bsz:
            raise ValueError(f"action_prev 形状应为 [B]（B={bsz}），实际 {tuple(action_prev.shape)}")
        action_e = self.action_emb(a_idx)
        z_hat = self.predictor(z, action_e, w_read.value)

        ctx_z = h.mean(dim=1)                        # 块状态池化，作为决策主输入
        decision = self.head.decide(ctx_z, w_read.value, mem)

        out = StepOutput(
            z=z,
            z_hat=z_hat,
            error=err,
            uncertainty=unc,
            decision=decision,
            ctx=ctx_z,
            # 负载均衡梯度挂在 router 的软概率上，而不是某个深度迭代的块内掩码：
            # 后者会随 depth 变化而让辅助损失含义漂移。
            route_probs=route.probs,
            route_stats=route.stats,
            ws_pressure=ws_pressure,
            overflow=overflow,
            k=k_eff,
            depth=depth,
            blocks_touched=touched,
        )
        new_state = NState(
            h=h,
            ws=ws_state,
            last_z_hat=z_hat.detach(),
            tick=state.tick + 1,
            n_updates=state.n_updates + touched,
        )
        with torch.no_grad():
            self.block_load_count.scatter_add_(
                0, route.indices.reshape(-1), torch.ones_like(route.indices.reshape(-1), dtype=self.block_load_count.dtype)
            )
        return out, new_state

    # ------------------------------------------------------------------ 序列

    def forward_sequence(
        self,
        numeric: torch.Tensor,             # [B, T, D]
        actions: torch.Tensor,             # [B, T] 本步实际执行的动作（作为下一步的条件）
        text_feat: torch.Tensor | None = None,   # [B, T, text_dim] 或 None
        mem: torch.Tensor | None = None,        # [B, T, mem_dim] 或 None
        k: int | None = None,
        depth: int = 1,
    ) -> dict[str, torch.Tensor]:
        if numeric.dim() != 3:
            raise ValueError(f"numeric 应为 [B,T,D]，实际 {tuple(numeric.shape)}")
        bsz, T, _ = numeric.shape
        state = self.initial_state(bsz, numeric.device)
        zs, hats, errs, preds, acts, ctxs, probs = [], [], [], [], [], [], []
        touched_total = 0
        route_stats: dict[str, Any] = {}
        for t in range(T):
            prev_action = torch.zeros(bsz, dtype=torch.long, device=numeric.device) if t == 0 else actions[:, t - 1]
            out, state = self.step(
                numeric[:, t],
                state,
                prev_action,
                None if text_feat is None else text_feat[:, t],
                None if mem is None else mem[:, t],
                k=k,
                depth=depth,
            )
            zs.append(out.z)
            hats.append(out.z_hat)
            errs.append(out.error)
            acts.append(out.decision["logits"])
            preds.append(out.decision["probabilities"])
            ctxs.append(out.ctx)
            probs.append(out.route_probs)
            touched_total += out.blocks_touched
            route_stats = out.route_stats
        if T == 0:
            raise ValueError("序列长度 T 不能为 0")
        return {
            "z": torch.stack(zs, dim=1),                          # [B,T,z_dim]
            "z_hat": torch.stack(hats, dim=1),
            "error": torch.stack(errs, dim=1),
            "ctx": torch.stack(ctxs, dim=1),                      # [B,T,d_h] 块状态池化
            "route_probs": torch.stack(probs, dim=1),             # [B,T,N]
            "action_logits": torch.stack(acts, dim=1),
            "action_probs": torch.stack(preds, dim=1),
            "route_stats": route_stats,
            "blocks_touched": touched_total,
        }

    # ------------------------------------------------------------------ 元信息

    def forward(self, numeric: torch.Tensor, actions: torch.Tensor, **kw) -> dict[str, torch.Tensor]:
        """nn.Module 接口：与 forward_sequence 同一条路径，不另设第二份实现。"""
        return self.forward_sequence(numeric, actions, **kw)

    def n_params(self) -> dict[str, int]:
        parts = {
            "encoder": self.encoder,
            "blocks": self.blocks,
            "router": self.router,
            "workspace": self.workspace,
            "predictor": self.predictor,
            "head": self.head,
            "action_emb": self.action_emb,
            "block_to_ws": self.block_to_ws,
        }
        counts = {k: sum(p.numel() for p in m.parameters()) for k, m in parts.items()}
        counts["total"] = sum(counts.values())
        return counts

    def active_block_stats(self) -> dict[str, Any]:
        load = self.block_load_count / self.block_load_count.sum().clamp_min(1.0)
        return {
            "load_per_block": [round(float(v), 4) for v in load.tolist()],
            "dead_blocks": int((self.block_load_count < 1).sum().item()),
            "top1_block_share": round(float(load.max().item()), 4),
        }
