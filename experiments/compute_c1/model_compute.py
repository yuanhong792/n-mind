"""C1.0 计算模型：复用 NCell/BlockRouter/GlobalWorkspace，随机初始化。

- top_k < n_blocks（默认 4 块选 2），确保 Router 真有选择空间。
- token embedding 与输出 head 全部自主学习，无预训练权重。
- 提供参数量接近的普通 GRU 参考模型（仅作对照，不替代神经核心）。
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from nmind.core.ncell import NCell
from nmind.core.router import BlockRouter
from nmind.core.workspace import GlobalWorkspace
from tokenizer_compute import ComputeTokenizer, EOS, PAD, vocab_sha


@dataclass
class ComputeConfig:
    vocab_size: int = 99
    d_model: int = 64
    n_blocks: int = 4
    block_hidden: int = 64
    workspace_dim: int = 32
    workspace_slots: int = 8
    top_k: int = 2
    def to_dict(self) -> dict[str, Any]:
        return dict(vocab_size=self.vocab_size, d_model=self.d_model,
                    n_blocks=self.n_blocks, block_hidden=self.block_hidden,
                    workspace_dim=self.workspace_dim,
                    workspace_slots=self.workspace_slots, top_k=self.top_k)


class ComputeLM(nn.Module):
    """字符级计算模型；递归状态 (h, ws) 显式携带。"""

    def __init__(self, cfg: ComputeConfig):
        super().__init__()
        if not (1 <= cfg.top_k < cfg.n_blocks):
            raise ValueError(f"top_k 必须 < n_blocks 才有稀疏空间: {cfg.top_k}/{cfg.n_blocks}")
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=PAD)
        self.blocks = NCell(cfg.d_model, cfg.block_hidden, cfg.workspace_dim, cfg.n_blocks)
        self.router = BlockRouter(z_dim=cfg.d_model, d_h=cfg.block_hidden,
                                  d_w=cfg.workspace_dim, n_blocks=cfg.n_blocks,
                                  hidden=max(16, cfg.d_model // 2))
        self.workspace = GlobalWorkspace(z_dim=cfg.d_model, d_w=cfg.workspace_dim,
                                          slots=cfg.workspace_slots)
        self.block_to_ws = nn.Linear(cfg.block_hidden, cfg.workspace_dim)
        self.head = nn.Linear(cfg.block_hidden, cfg.vocab_size)
        self.register_buffer("block_load_count", torch.zeros(cfg.n_blocks))
        self.use_router_load_update = True
        self.router_gradient_path_enabled = True
        # 评测用统计（不参与梯度）
        self.last_route_indices: list[list[int]] = []
        self.last_workspace_occupied: list[float] = []

    @property
    def top_k(self) -> int:
        return self.cfg.top_k

    def initial_state(self, batch: int, device):
        return {"h": torch.zeros(batch, self.cfg.n_blocks, self.cfg.block_hidden, device=device),
                "ws": self.workspace.initial_state(batch, device)}

    def forward(self, tokens: torch.Tensor, state: dict | None = None,
                collect_stats: bool = False):
        B, T = tokens.shape
        dev = tokens.device
        st = state if state is not None else self.initial_state(B, dev)
        h, ws = st["h"], st["ws"]
        err = torch.full((B, 1), 0.5, device=dev)
        unc = torch.full((B, 1), 0.5, device=dev)
        rows = torch.arange(B, device=dev)[:, None].expand(-1, self.top_k).reshape(-1)
        logits: list[torch.Tensor] = []
        route_log: list[list[int]] = []
        occ_log: list[float] = []
        for t in range(T):
            z = self.tok_emb(tokens[:, t])
            w_read = self.workspace.read(ws, z)
            route = self.router.route(z, h, w_read.value, err, unc, k=self.top_k)
            w_arg = route.weights if (self.router_gradient_path_enabled and self.training) else None
            h, _g, _m, _p, _n = self.blocks.forward_sparse(
                z, h, w_read.value, route.indices, weights=w_arg)
            payload = self.block_to_ws(h[rows, route.indices.reshape(-1)]).view(B, self.top_k, -1)
            wr = self.workspace.write(ws, payload)
            ws = {"content": wr.state["content"], "occupied": wr.occupied, "utility": wr.utility}
            if self.training and self.use_router_load_update:
                load = torch.zeros(self.cfg.n_blocks, device=dev)
                load.scatter_add_(0, route.indices.reshape(-1),
                                  torch.ones_like(route.indices.reshape(-1), dtype=load.dtype))
                load = load / max(1.0, float(route.indices.numel()))
                with torch.no_grad():
                    self.router.update_load_bias(load)
                    self.block_load_count.scatter_add_(
                        0, route.indices.reshape(-1),
                        torch.ones_like(route.indices.reshape(-1), dtype=self.block_load_count.dtype))
            decoded = h.mean(dim=1)
            logits.append(self.head(decoded))
            if collect_stats:
                route_log.append(route.indices[0].detach().cpu().tolist())
                occ_log.append(float(ws["occupied"][0].mean().detach()))
        if collect_stats:
            self.last_route_indices = route_log
            self.last_workspace_occupied = occ_log
        return torch.stack(logits, dim=1), {"h": h, "ws": ws}

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class GRUReference(nn.Module):
    """参数量接近的普通 GRU 对照；不是 N-MIND 神经核心。"""
    def __init__(self, cfg: ComputeConfig, hidden: int = 128, layers: int = 1):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=PAD)
        self.gru = nn.GRU(cfg.d_model, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, cfg.vocab_size)
        self.hidden_size = hidden

    def forward(self, tokens, state=None, collect_stats=False):
        z = self.emb(tokens)
        out, h = self.gru(z, state)
        return self.head(out), h

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def save_compute(path, model: ComputeLM, tok: ComputeTokenizer, *,
                 step: int, optimizer=None, rng=None, extra: dict | None = None) -> str:
    import os
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "format": "nmind-compute-lm/1",
        "version": "compute-c1",
        "tokenizer_version": tok.version,
        "tokenizer_sha256": tok.vocab_sha,
        "vocab_size": tok.vocab_size,
        "config": model.cfg.to_dict(),
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "optimizer_steps_executed": int(step),
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "rng_state": rng,
        "torch_rng_state": torch.get_rng_state(),
        "extra": extra or {},
    }
    tmp = p.with_suffix(p.suffix + ".tmp")
    torch.save(blob, tmp)
    with open(tmp, "r+b") as f:
        os.fsync(f.fileno())
    tmp.replace(p)
    check = torch.load(p, map_location="cpu", weights_only=False)
    for k, v in blob["state_dict"].items():
        if not torch.equal(check["state_dict"][k], v):
            raise RuntimeError(f"checkpoint 张量核验失败: {k}")
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_compute(path, tok: ComputeTokenizer):
    blob = torch.load(Path(path), map_location="cpu", weights_only=False)
    if blob.get("format") != "nmind-compute-lm/1":
        raise ValueError("不是 nmind 计算模型 checkpoint")
    if blob.get("tokenizer_sha256") != tok.vocab_sha:
        raise ValueError("tokenizer SHA 不匹配")
    model = ComputeLM(ComputeConfig(**blob["config"]))
    model.load_state_dict(blob["state_dict"], strict=True)
    model.eval()
    return model, blob


def model_sha(model) -> str:
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
