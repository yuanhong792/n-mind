"""FLY-Phase0 机制模块：随机稀疏扩展 + k-WTA + 局部三因素可塑性。

本模块只用 NumPy，不导入 torch，不调用 autograd/backward。
所有权重随机或零初始化；不使用预训练权重、外部 LLM 输出或 N-MIND stable 权重。
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

NAME_CHARS = "小明红号位置哪里收到好了白了"
# MQAR 的 NAME_CHARS 实际含 14 个字符位置、13 个唯一字符；预注册固定 D=690，
# 因此保留 15 个单字符槽位，其中未出现的槽位恒为 0（不影响编码与哈希）。
N_CHARS = 15
CHAR_ID = {ch: i for i, ch in enumerate(dict.fromkeys(NAME_CHARS))}
BIGRAM_POS = 3
D_INPUT = N_CHARS + BIGRAM_POS * N_CHARS * N_CHARS


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encode_name(name: str) -> np.ndarray:
    """4 字名字 -> 单字符 + 位置相关相邻二元组多热向量。"""
    if len(name) != 4:
        raise ValueError(f"FLY 名字长度固定为 4: {name!r}")
    h = np.zeros(D_INPUT, dtype=np.float32)
    ids = []
    for ch in name:
        if ch not in CHAR_ID:
            raise ValueError(f"名字字符不在 NAME_CHARS: {ch!r}")
        ids.append(CHAR_ID[ch])
    for i in ids:
        h[i] = 1.0
    for pos in range(BIGRAM_POS):
        base = N_CHARS + pos * N_CHARS * N_CHARS
        h[base + ids[pos] * N_CHARS + ids[pos + 1]] = 1.0
    return h


def encode_names(names: Sequence[str]) -> np.ndarray:
    return np.stack([encode_name(x) for x in names]).astype(np.float32, copy=False)


def ranked_indices(z: np.ndarray, k: int) -> np.ndarray:
    """返回预激活最高的 k 个索引；并列按索引升序。"""
    k = max(0, min(int(k), z.size))
    if k == 0:
        return np.empty(0, dtype=np.int64)
    if k == z.size:
        return np.arange(z.size, dtype=np.int64)
    cut = np.partition(z, z.size - k)[z.size - k]
    above = np.flatnonzero(z > cut)
    equal = np.flatnonzero(z == cut)
    need = k - above.size
    chosen = np.concatenate([above, equal[:need]])
    return np.sort(chosen.astype(np.int64, copy=False))


@dataclass(frozen=True)
class ExpansionConfig:
    kind: str
    M: int
    fan_in: int | None = None
    sparsity: float | None = None
    seed: int = 20261125

    @property
    def k(self) -> int | None:
        if self.sparsity is None:
            return None
        return max(1, int(round(self.sparsity * self.M)))


class SparseExpansion:
    """固定随机稀疏二值投影 + 全局 k-WTA。"""

    def __init__(self, M: int, fan_in: int, sparsity: float, seed: int):
        if M <= 0 or fan_in <= 0 or fan_in > D_INPUT:
            raise ValueError("M/fan_in 非法")
        if not (0.0 < sparsity <= 1.0):
            raise ValueError("sparsity 必须在 (0, 1]")
        self.cfg = ExpansionConfig("sparse_wta", M, fan_in, sparsity, seed)
        self.M = M
        self.fan_in = fan_in
        self.sparsity = sparsity
        self.k = max(1, int(round(sparsity * M)))
        rng = np.random.default_rng(seed)
        self.R = np.zeros((M, D_INPUT), dtype=np.float32)
        for r in range(M):
            cols = rng.choice(D_INPUT, size=fan_in, replace=False)
            self.R[r, cols] = 1.0

    def encode(self, name_or_h: str | np.ndarray) -> np.ndarray:
        h = encode_name(name_or_h) if isinstance(name_or_h, str) else np.asarray(name_or_h, dtype=np.float32)
        z = self.R @ h
        out = np.zeros(self.M, dtype=np.float32)
        out[ranked_indices(z, self.k)] = 1.0
        return out

    def encode_many(self, names: Sequence[str]) -> np.ndarray:
        return np.stack([self.encode(x) for x in names])

    def digest(self) -> str:
        return sha256_bytes(self.R.tobytes() + str(self.cfg).encode("utf-8"))


class DenseProjection:
    """同维度稠密随机投影，无 k-WTA；用于 FLY-01 与稠密对照。"""

    def __init__(self, M: int, seed: int, scale: float = 1.0):
        self.cfg = ExpansionConfig("dense_projection", M, None, None, seed)
        self.M = M
        rng = np.random.default_rng(seed)
        self.R = rng.normal(0.0, 1.0 / math.sqrt(D_INPUT), size=(M, D_INPUT)).astype(np.float32) * scale
        self.k = M

    def encode(self, name_or_h: str | np.ndarray) -> np.ndarray:
        h = encode_name(name_or_h) if isinstance(name_or_h, str) else np.asarray(name_or_h, dtype=np.float32)
        return self.R @ h

    def encode_many(self, names: Sequence[str]) -> np.ndarray:
        return np.stack([self.encode(x) for x in names])

    def digest(self) -> str:
        return sha256_bytes(self.R.tobytes() + str(self.cfg).encode("utf-8"))


class IdentityExpansion:
    """仅用于单元测试的 1-hot 恒等编码。"""

    def __init__(self, M: int):
        self.M = M
        self.k = M
        self.cfg = ExpansionConfig("identity", M, None, None, 0)

    def encode(self, idx_or_h: int | np.ndarray) -> np.ndarray:
        if isinstance(idx_or_h, (int, np.integer)):
            v = np.zeros(self.M, dtype=np.float32)
            v[int(idx_or_h)] = 1.0
            return v
        return np.asarray(idx_or_h, dtype=np.float32)

    def encode_many(self, items: Sequence[int]) -> np.ndarray:
        return np.stack([self.encode(i) for i in items])

    def digest(self) -> str:
        return sha256_bytes(f"identity:{self.M}".encode("utf-8"))


@dataclass
class PlasticConfig:
    M: int = 2048
    fan_in: int = 6
    sparsity: float = 0.05
    eta: float = 0.2
    decay: float = 0.99
    clip: float = 1.0
    variant: str = "A"
    seed: int = 20261125

    def signature(self) -> str:
        return (f"M={self.M}|fan_in={self.fan_in}|s={self.sparsity:.4g}|"
                f"eta={self.eta:.4g}|decay={self.decay:.4g}|clip={self.clip:.4g}|"
                f"variant={self.variant}|seed={self.seed}")


class FlyMemory:
    """局部三因素可塑记忆。A=delta 规则；B=多巴胺式抑制。"""

    def __init__(self, cfg: PlasticConfig, expansion=None):
        if cfg.variant not in {"A", "B"}:
            raise ValueError("variant 只能是 A 或 B")
        if cfg.decay <= 0 or cfg.decay > 1:
            raise ValueError("decay 必须在 (0, 1]")
        if cfg.clip <= 0:
            raise ValueError("clip 必须 > 0")
        self.cfg = cfg
        self.expansion = expansion or SparseExpansion(cfg.M, cfg.fan_in, cfg.sparsity, cfg.seed)
        if self.expansion.M != cfg.M:
            raise ValueError("expansion.M 与 cfg.M 不一致")
        self.W = np.zeros((cfg.M, cfg.M), dtype=np.float32)

    def reset(self) -> None:
        self.W.fill(0.0)

    def store_pairs(self, pairs: Sequence[tuple[str, str]], epochs: int = 1) -> None:
        """按固定轮数重复呈现同一批键值对；每轮仍逐条调用 write。"""
        if epochs < 1:
            raise ValueError("epochs 必须 >= 1")
        for _ in range(epochs):
            for key, value in pairs:
                self.write(key, value)

    def keys(self) -> set[str]:
        return set(self.__dict__.keys())

    def write(self, key: str, value: str) -> None:
        a = self.expansion.encode(key)
        b = self.expansion.encode(value)
        if self.cfg.decay != 1.0:
            self.W *= self.cfg.decay
        if self.cfg.variant == "A":
            delta = b - self.W.T @ a
            self.W += self.cfg.eta * np.outer(a, delta)
        else:
            # d=+1 表示正确写入的正向调制；对活跃前/后突触做抑制（反赫布对照）。
            d = 1.0
            if d > 0:
                self.W -= self.cfg.eta * np.outer(a, b)
        np.clip(self.W, -self.cfg.clip, self.cfg.clip, out=self.W)

    def read_scores(self, key: str, candidates: Sequence[str] | np.ndarray) -> np.ndarray:
        a = self.expansion.encode(key)
        if isinstance(candidates, np.ndarray) and candidates.ndim == 2 and candidates.shape[1] == self.cfg.M:
            C = candidates
        else:
            C = np.stack([self.expansion.encode(v) for v in candidates])
        proj = a @ self.W
        return proj @ C.T

    def read(self, key: str, candidates: Sequence[str] | np.ndarray) -> int:
        scores = self.read_scores(key, candidates)
        return int(np.argmax(scores))

    def recall(self, pairs: Sequence[tuple[str, str]]) -> float:
        return recall_pairs(self, pairs)

    def digest(self) -> str:
        return sha256_bytes(self.W.tobytes() + self.cfg.signature().encode("utf-8"))


class LinearAttentionMemory:
    """线性注意力式外积快权重基线：W += eta · b aᵀ，读出同 FLY。"""

    def __init__(self, cfg: PlasticConfig, expansion=None):
        self.cfg = cfg
        self.expansion = expansion or SparseExpansion(cfg.M, cfg.fan_in, cfg.sparsity, cfg.seed)
        self.W = np.zeros((cfg.M, cfg.M), dtype=np.float32)

    def reset(self) -> None:
        self.W.fill(0.0)

    def write(self, key: str, value: str) -> None:
        a = self.expansion.encode(key)
        b = self.expansion.encode(value)
        self.W += self.cfg.eta * np.outer(b, a)
        np.clip(self.W, -self.cfg.clip, self.cfg.clip, out=self.W)

    def read_scores(self, key: str, candidates: Sequence[str] | np.ndarray) -> np.ndarray:
        a = self.expansion.encode(key)
        C = (candidates if isinstance(candidates, np.ndarray) and candidates.ndim == 2
             else self.expansion.encode_many(candidates))
        return (a @ self.W.T) @ C.T

    def read(self, key: str, candidates: Sequence[str] | np.ndarray) -> int:
        return int(np.argmax(self.read_scores(key, candidates)))


def random_names(rng: np.random.Generator, n: int) -> list[str]:
    return ["".join(str(NAME_CHARS[i]) for i in rng.choice(N_CHARS, size=4, replace=True))
            for _ in range(n)]


def recall_pairs(mem: FlyMemory | LinearAttentionMemory, pairs: Sequence[tuple[str, str]]) -> float:
    """按 pairs 顺序作为候选集合，计算键->值召回。"""
    values = [v for _, v in pairs]
    C = mem.expansion.encode_many(values)
    correct = 0
    for idx, (key, _) in enumerate(pairs):
        if mem.read(key, C) == idx:
            correct += 1
    return correct / max(1, len(pairs))
