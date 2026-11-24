"""M0：C1.0 N-Cell/Router/Workspace 参考模型（随机初始化）。"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
C1 = HERE.parent.parent / "compute_c1"
sys.path.insert(0, str(C1))

from model_compute import ComputeConfig, ComputeLM


class NCellReference(nn.Module):
    def __init__(self, vocab_size: int, d_model=64, n_blocks=4, block_hidden=64,
                 workspace_dim=32, workspace_slots=8, top_k=2):
        super().__init__()
        cfg = ComputeConfig(vocab_size=vocab_size, d_model=d_model, n_blocks=n_blocks,
                            block_hidden=block_hidden, workspace_dim=workspace_dim,
                            workspace_slots=workspace_slots, top_k=top_k)
        self.core = ComputeLM(cfg)
        self.cfg = cfg

    def forward(self, tokens, state=None, valid_mask=None, collect_stats=False):
        return self.core(tokens, state, collect_stats=collect_stats, valid_mask=valid_mask)

    @property
    def use_router_load_update(self):
        return self.core.use_router_load_update

    @use_router_load_update.setter
    def use_router_load_update(self, value):
        self.core.use_router_load_update = bool(value)

    @property
    def router_gradient_path_enabled(self):
        return self.core.router_gradient_path_enabled

    @router_gradient_path_enabled.setter
    def router_gradient_path_enabled(self, value):
        self.core.router_gradient_path_enabled = bool(value)

    @property
    def block_load_count(self):
        return self.core.block_load_count

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def state_elements(self):
        c = self.cfg
        return c.n_blocks * c.block_hidden + c.workspace_slots * c.workspace_dim + 2 * c.workspace_slots

    def config_dict(self):
        return {"model_type": "M0_ncell_reference", **self.cfg.to_dict()}

    def load_state_dict(self, state_dict, strict=True):
        return self.core.load_state_dict(state_dict, strict=strict)

    def state_dict(self, *args, **kwargs):
        return self.core.state_dict(*args, **kwargs)
