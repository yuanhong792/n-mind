"""可选的本地预训练文本模型适配器（M4）。

三条不可妥协的规则：
  1. **绝不自动下载**：加载时强制 local_files_only=True，权重必须由用户自己放到位；
  2. **绝不执行仓库代码**：trust_remote_code=False，只用 transformers 已注册的架构；
  3. **依赖是可选项**：没装 transformers 或没放权重，就返回 None 并说明原因，
     系统整体照常用哈希文本特征跑（规则 6：缺能力要能降级）。

拿到的小模型特征只作为编码器的 text 输入通道使用，不改动 N-Core 的任何结构。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class AdapterStatus:
    requested: bool
    available: bool
    reason: str
    path: str = ""
    detail: dict[str, Any] = field(default_factory=dict)


class LocalTextAdapter:
    """把文本变成长度固定的特征向量；不可用时调用方自行退回哈希特征。"""

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.lang = getattr(cfg, "language", None)
        self._model: Any = None
        self._tok: Any = None
        self.status = self._probe()

    def _probe(self) -> AdapterStatus:
        if self.lang is None or self.lang.source != "local_adapter":
            return AdapterStatus(False, False, "配置未启用本地语言适配器（默认），使用哈希文本特征")
        raw = (self.lang.adapter_path or "").strip()
        if not raw:
            return AdapterStatus(True, False, "language.source=local_adapter 但未给 adapter_path；"
                                              "系统不会自行下载权重")
        p = Path(raw)
        if not p.exists():
            return AdapterStatus(True, False, f"适配器路径不存在：{p}")
        try:
            import torch                     # noqa: F401
            from transformers import AutoModel, AutoTokenizer   # 只在真要用的时候 import
        except ImportError:
            return AdapterStatus(True, False, "未安装 transformers；结构化实验不依赖它，"
                                              "如需通用语言请先 pip install transformers 并放入权重")
        try:
            torch_ = __import__("torch")
            self._tok = AutoTokenizer.from_pretrained(str(p), local_files_only=True,
                                                     trust_remote_code=False)
            self._model = AutoModel.from_pretrained(str(p), local_files_only=True,
                                                   trust_remote_code=False,
                                                   torch_dtype=torch_.float32)
            self._model.eval()
            hidden = int(getattr(self._model.config, "hidden_size", 0) or 0)
            return AdapterStatus(True, True, "本地权重已加载（离线，未下载任何东西）",
                                 str(p), {"hidden_size": hidden,
                                          "max_text_chars": self.lang.max_text_chars,
                                          "attention": getattr(self._model.config,
                                                                "_attn_implementation", "unknown")})
        except Exception as e:              # noqa: BLE001 - 加载失败必须降级而不是崩溃
            return AdapterStatus(True, False, f"权重加载失败：{type(e).__name__}: {e}", str(p))

    # ------------------------------------------------------------------ 特征

    def encode(self, texts: list[str]) -> np.ndarray | None:
        """返回 [len(texts), hidden] 的均值池化特征；不可用返回 None。"""
        if not self.status.available:
            return None
        import torch

        clipped = [t[: self.lang.max_text_chars] for t in texts]
        with torch.no_grad():
            enc = self._tok(clipped, return_tensors="pt", padding=True, truncation=True,
                            max_length=512)
            out = self._model(**enc)
            hidden = out.last_hidden_state
            mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1.0)
        return pooled.detach().cpu().numpy().astype(np.float32)

    def project_dim(self) -> int:
        return int(self.status.detail.get("hidden_size") or 0)
