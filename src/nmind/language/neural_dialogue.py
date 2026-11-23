"""基于 NCell/Router/Workspace 的字符级自回归语言模型（M5-Language 任务书 P0/P2）。

* 全部权重随机初始化：token embedding 与语言头均为新建层，无预训练词向量/教师/蒸馏。
* 序列计算主体 = NCell 块稀疏更新 + BlockRouter Top-K 选择 + GlobalWorkspace 读写
  （复用 core/ 既有组件，不重写网络数学）。
* 递归状态 (h, ws) 跨 token 流动，也可跨轮携带（会话状态管理，见 session.py）。
* 诚实口径（M4 教训）：err/unc 门控信号 v1 固定 0.5（无 predictor，如实记录）；
  router_ste 默认关闭（C 臂口径，任务盲路由是已知事实）；训练态 buffer
  （load_bias/block_load_count）沿用 M4 审计口径，评测前复位。
* checkpoint 版本化格式 nmind-lm/1：词表哈希不匹配直接拒绝加载，禁止静默加载。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from ..core.ncell import NCell
from ..core.router import BlockRouter
from ..core.workspace import GlobalWorkspace
from .tokenizer import CharTokenizer

LM_FORMAT = "nmind-lm/1"


@dataclass
class LMState:
    h: torch.Tensor                       # [B, N, d_h]
    ws: dict[str, torch.Tensor]


@dataclass
class LMConfig:
    vocab_size: int
    d_model: int = 96
    n_blocks: int = 8
    block_hidden: int = 96
    workspace_dim: int = 64
    workspace_slots: int = 4
    top_k: int = 2
    query_readout: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in
                ("vocab_size", "d_model", "n_blocks", "block_hidden",
                 "workspace_dim", "workspace_slots", "top_k", "query_readout")}


class NMindLM(nn.Module):
    """字符级自回归语言模型；权重随机初始化；递归状态显式携带。"""

    def __init__(self, cfg: LMConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = NCell(cfg.d_model, cfg.block_hidden, cfg.workspace_dim, cfg.n_blocks)
        self.router = BlockRouter(z_dim=cfg.d_model, d_h=cfg.block_hidden,
                                   d_w=cfg.workspace_dim, n_blocks=cfg.n_blocks,
                                   hidden=max(16, cfg.d_model // 2))
        self.workspace = GlobalWorkspace(z_dim=cfg.d_model, d_w=cfg.workspace_dim,
                                          slots=cfg.workspace_slots)
        self.block_to_ws = nn.Linear(cfg.block_hidden, cfg.workspace_dim)
        self.head = nn.Linear(cfg.block_hidden, cfg.vocab_size)
        self.query_readout = bool(getattr(cfg, "query_readout", False))
        if self.query_readout:
            self.query_readout_proj = nn.Linear(
                cfg.block_hidden + cfg.d_model + cfg.workspace_dim,
                cfg.block_hidden,
            )
        self.register_buffer("block_load_count", torch.zeros(cfg.n_blocks))  # 训练态统计
        self.top_k = cfg.top_k

    def initial_state(self, batch: int, device) -> LMState:
        return LMState(
            h=torch.zeros(batch, self.cfg.n_blocks, self.cfg.block_hidden, device=device),
            ws=self.workspace.initial_state(batch, device))

    def forward(self, tokens: torch.Tensor,
                state: LMState | None = None) -> tuple[torch.Tensor, LMState]:
        """tokens [B,T] → logits [B,T,V]。state 携带时从该状态续算（增量推理）。"""
        B, T = tokens.shape
        dev = tokens.device
        st = state or self.initial_state(B, dev)
        h, ws = st.h, st.ws
        err = torch.full((B, 1), 0.5, device=dev)
        unc = torch.full((B, 1), 0.5, device=dev)
        rows = torch.arange(B, device=dev)[:, None].expand(-1, self.top_k).reshape(-1)
        logits: list[torch.Tensor] = []
        for t in range(T):
            z = self.tok_emb(tokens[:, t])
            w_read = self.workspace.read(ws, z)
            route = self.router.route(z, h, w_read.value, err, unc, k=self.top_k)
            h, _g, _m, _p, _n = self.blocks.forward_sparse(z, h, w_read.value, route.indices)
            sel = route.indices.reshape(-1)
            payload = self.block_to_ws(h[rows, sel]).view(B, self.top_k, -1)
            wr = self.workspace.write(ws, payload)
            ws = {"content": wr.state["content"], "occupied": wr.occupied,
                   "utility": wr.utility}
            if self.training and self.use_router_load_update:
                load = torch.zeros(self.cfg.n_blocks, device=dev)
                load.scatter_add_(0, route.indices.reshape(-1),
                                   torch.ones_like(route.indices.reshape(-1), dtype=load.dtype))
                load = load / max(1.0, float(route.indices.numel()))
                with torch.no_grad():
                    self.router.update_load_bias(load)
                    self.block_load_count.scatter_add_(
                        0, route.indices.reshape(-1),
                        torch.ones_like(route.indices.reshape(-1),
                                         dtype=self.block_load_count.dtype))
            decoded = h.mean(dim=1)
            if self.query_readout:
                decoded = self.query_readout_proj(
                    torch.cat([decoded, z, w_read.value], dim=-1)
                )
            logits.append(self.head(decoded))
        return torch.stack(logits, dim=1), LMState(h=h, ws=ws)

    use_router_load_update: bool = True      # 类属性开关（评测/训练态由调用方控制 training flag）

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def save_lm(path: str | Path, *, model: NMindLM, tokenizer: CharTokenizer, cfg: LMConfig,
            steps: int, seed: int, parent_sha: str, manifest_hash: str) -> str:
    blob = {"format": LM_FORMAT, "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
             "config": cfg.to_dict(),
             "vocab_size": tokenizer.vocab_size, "vocab_hash": tokenizer.vocab_hash(),
             "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
             "steps": steps, "seed": seed, "parent_checkpoint_sha256": parent_sha,
             "data_manifest_hash": manifest_hash}
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, p)
    return str(p)


def load_lm(path: str | Path, tokenizer: CharTokenizer) -> tuple[NMindLM, dict[str, Any]]:
    """严格加载：格式/词表哈希/形状不匹配一律拒绝，绝不静默降级。"""
    blob = torch.load(Path(path), map_location="cpu", weights_only=True)
    if blob.get("format") != LM_FORMAT:
        raise ValueError(f"不是 nmind 语言模型 checkpoint（format={blob.get('format')!r}）")
    if blob.get("vocab_hash") != tokenizer.vocab_hash():
        raise ValueError("词表哈希不匹配：checkpoint 与 tokenizer 不兼容，拒绝加载")
    cfg = LMConfig(**blob["config"])
    if cfg.vocab_size != tokenizer.vocab_size:
        raise ValueError("vocab_size 与 tokenizer 不一致")
    model = NMindLM(cfg)
    model.load_state_dict(blob["state_dict"], strict=True)
    model.eval()                            # 必须显式 eval：train 模式首推理就会漂移 buffer（M4 教训）
    meta = {"steps": blob.get("steps"), "seed": blob.get("seed"),
             "parent_checkpoint_sha256": blob.get("parent_checkpoint_sha256"),
             "saved_at": blob.get("saved_at"),
             "data_manifest_hash": blob.get("data_manifest_hash")}
    return model, meta
