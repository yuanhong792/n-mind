"""M2 模型共享基类：tokenizer、PAD、状态管理、工具函数。"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
M2ROOT = HERE.parent
LAB = M2ROOT / "math_lab"
if str(M2ROOT) not in sys.path:
    sys.path.insert(0, str(M2ROOT))
if str(LAB) not in sys.path:
    sys.path.insert(0, str(LAB))

PAD = 0


def masked_update(new, old, valid):
    """valid:[B] bool/float；padding 位置保持旧状态。"""
    v = valid.to(new.dtype)
    if v.dim() == 1:
        v = v.view(-1, *([1] * (new.dim() - 1)))
    return v * new + (1.0 - v) * old


class M2Base(nn.Module):
    """统一接口：forward(tokens, state, valid_mask, collect_stats) -> (logits, state)。"""

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def state_elements(self):
        return 0

    def config_dict(self):
        return {"model_type": self.__class__.__name__, **getattr(self, "cfg", {})}
