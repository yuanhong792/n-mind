"""EventEncoder：把结构化事件（和可选的轻量文本特征）映射到潜状态 z_t。

这里刻意不使用任何预训练语言模型。文本走确定性哈希袋特征，因此 z_t 表达的
只是"这条事件在特征空间里的位置"，不得据此宣称系统具备通用语言能力。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import nn


@dataclass
class Event:
    """一条可持久化的事件观测。

    numeric: 结构化数值特征（网格坐标、速度、遮挡、碰撞等），长度须等于 event_numeric_dim
    text:    短文本事实；None 表示本事件没有文本部分
    """

    tick: int = 0
    numeric: np.ndarray | None = None
    text: str | None = None
    source: str = "synthetic"
    meta: dict[str, Any] = field(default_factory=dict)

    def payload(self) -> dict[str, Any]:
        return {
            "tick": self.tick,
            "numeric": None if self.numeric is None else np.asarray(self.numeric).tolist(),
            "text": self.text,
            "source": self.source,
            "meta": self.meta,
        }


class EventEncoder(nn.Module):
    """numeric / text 双通道拼接后映射到 z_t。缺哪个通道就把对应投影置零，保证仍可运行。"""

    def __init__(
        self,
        numeric_dim: int,
        text_dim: int,
        z_dim: int,
        hidden: int = 64,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        self.numeric_dim = numeric_dim
        self.text_dim = text_dim
        self.z_dim = z_dim
        self.normalize = normalize

        self.num_mlp = nn.Sequential(
            nn.Linear(numeric_dim, hidden), nn.GELU(), nn.Linear(hidden, z_dim)
        ) if numeric_dim > 0 else None
        self.txt_mlp = nn.Sequential(
            nn.Linear(text_dim, hidden), nn.GELU(), nn.Linear(hidden, z_dim)
        ) if text_dim > 0 else None
        fuse_in = (z_dim if self.num_mlp else 0) + (z_dim if self.txt_mlp else 0)
        self.fuse = nn.Linear(fuse_in, z_dim) if fuse_in != z_dim else None
        self.out_norm = nn.LayerNorm(z_dim) if normalize else None

    def forward(
        self, numeric: torch.Tensor | None, text_feat: torch.Tensor | None
    ) -> torch.Tensor:
        if numeric is None and text_feat is None:
            raise ValueError("EventEncoder 需要至少一个输入通道")
        parts = []
        if self.num_mlp is not None:
            if numeric is None:
                # 设备跟另一个输入走，否则 GPU 模式下会造出 CPU 张量（CUDA 冒烟实验暴露）
                numeric = torch.zeros(1, self.numeric_dim, device=text_feat.device)
            if numeric.shape[-1] != self.numeric_dim:
                raise ValueError(
                    f"numeric 维度应为 {self.numeric_dim}，实际 {numeric.shape[-1]}"
                )
            parts.append(self.num_mlp(numeric.float()))
        if self.txt_mlp is not None:
            if text_feat is None:
                b = parts[0].shape[0] if parts else 1
                # 设备跟 numeric 走：训练/推理在 CUDA 上时这里必须在同一设备
                text_feat = torch.zeros(b, self.text_dim, device=numeric.device)
            if text_feat.shape[-1] != self.text_dim:
                raise ValueError(
                    f"text_feat 维度应为 {self.text_dim}，实际 {text_feat.shape[-1]}"
                )
            parts.append(self.txt_mlp(text_feat.float()))
        h = parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)
        if self.fuse is not None:
            h = self.fuse(h)
        if self.out_norm is not None:
            h = self.out_norm(h)
        return h

    @staticmethod
    def text_features(text: str | None, dim: int) -> torch.Tensor | None:
        """确定性哈希袋特征（feature hashing）：无词汇表、无预训练、无外部依赖。

        对每个字符一元与词元取 sha 前缀，映射到 [0,dim) 的槽位并按符号累加，
        最后按 L2 归一化。它只能表达"这串文本里出现过哪些片段"，
        **不具备语义理解能力**——所以文档里绝不把它说成语言表征。
        """
        if not text:
            return None
        import hashlib
        import re

        vec = torch.zeros(dim, dtype=torch.float32)
        toks: list[str] = re.findall(r"[a-z0-9]+", text.lower())
        cjk = [c for c in re.sub(r"\s+", "", text) if not c.isascii()]
        toks += cjk
        toks += [a + b for a, b in zip(cjk, cjk[1:])]                    # 中文二元：比单字更接近词
        for t in toks:
            if not t:
                continue
            d = hashlib.sha1(t.encode("utf-8", "surrogatepass")).digest()
            slot = int.from_bytes(d[:4], "big") % dim
            sign = 1.0 if d[4] & 1 else -1.0
            vec[slot] += sign
        n = float(vec.norm())
        return (vec / n if n > 0 else vec).unsqueeze(0)

    @staticmethod
    def to_tensor(values: list[float] | np.ndarray | None, dim: int) -> torch.Tensor | None:
        """把不规则事件列表补齐成 [B, dim]；缺失填 0，长度不符直接报错而不是静默截断。"""
        if values is None:
            return None
        if isinstance(values, np.ndarray):
            values = values.tolist()
        if not isinstance(values, (list, tuple)):
            raise TypeError("numeric 必须是 list/ndarray")
        if len(values) != dim:
            raise ValueError(f"事件数值特征长度应为 {dim}，实际 {len(values)}")
        return torch.tensor([[float(v) for v in values]], dtype=torch.float32)
