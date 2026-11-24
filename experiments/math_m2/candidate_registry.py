"""M2 候选注册表：R1/R2 参考 + M2-A/B/C/D/E。"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS = HERE / "models"
if str(MODELS) not in sys.path:
    sys.path.insert(0, str(MODELS))

from gru_reference import GRUReferenceM2          # noqa: E402
from transformer_reference import TinyTransformerM2  # noqa: E402
from dual_memory import DualMemoryM2, build_model    # noqa: E402

CANDIDATES = ["R1", "R2", "M2-A", "M2-B", "M2-C", "M2-D", "M2-E"]


def build(name: str, vocab_size: int):
    if name == "R1":
        return GRUReferenceM2(vocab_size)
    if name == "R2":
        return TinyTransformerM2(vocab_size)
    if name in DualMemoryM2.VARIANTS:
        return DualMemoryM2(vocab_size, variant=name)
    raise ValueError(f"unknown candidate: {name}")


def candidate_names():
    return list(CANDIDATES)


def family(name: str) -> str:
    if name in ("R1", "R2"):
        return "reference"
    return "dual_memory"
