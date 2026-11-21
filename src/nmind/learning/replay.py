"""经验回放：蓄水池采样保存旧轨迹，供持续学习混批。

为什么用蓄水池而不是"最近 N 条"：持续学习的失败模式恰恰是**旧任务被新数据挤光**，
保留窗口必须与到达顺序无关。Algorithm-R 能在不知道总量的情况下给出均匀样本，
代价是一个等大的数组，本地 CPU 完全可接受。

这里存的是**轨迹级**条目（一整段 obs/actions），不是单帧：
任务 A/B 的区别体现在"同一状态空间、不同决策规则"上，只存单帧会把
上下文信息丢掉，重放出来的梯度也就没有意义。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


class ReservoirReplay:
    def __init__(self, capacity: int = 2000, *, seed: int = 0) -> None:
        if capacity < 1:
            raise ValueError("capacity 至少为 1")
        self.capacity = int(capacity)
        self.rng = np.random.default_rng(seed)
        self._items: list[dict[str, Any]] = []
        self._seen = 0
        self.dropped = 0

    # ------------------------------------------------------------------ 写入

    def add(self, item: dict[str, Any]) -> bool:
        """蓄水池插入：未满直接放；满了以 capacity/seen 的概率替换随机一项。"""
        self._validate(item)
        self._seen += 1
        if len(self._items) < self.capacity:
            self._items.append(item)
            return True
        j = int(self.rng.integers(0, self._seen))
        if j < self.capacity:
            self._items[j] = item
            return True
        self.dropped += 1
        return False

    def extend(self, items: Iterable[dict[str, Any]]) -> int:
        return sum(int(self.add(i)) for i in items)

    @staticmethod
    def _validate(item: dict[str, Any]) -> None:
        for key in ("obs", "actions"):
            if key not in item:
                raise KeyError(f"重放条目缺少 '{key}' 字段")
        obs = np.asarray(item["obs"])
        acts = np.asarray(item["actions"])
        if obs.ndim != 2:
            raise ValueError(f"obs 应为 [T,D]，实际 shape={obs.shape}")
        if acts.ndim != 1 or acts.shape[0] != obs.shape[0]:
            raise ValueError(f"actions 应为 [T] 且与 obs 等长：obs {obs.shape} vs actions {acts.shape}")

    # ------------------------------------------------------------------ 取样

    def sample(self, n: int) -> list[dict[str, Any]]:
        if not self._items or n <= 0:
            return []
        n = min(int(n), len(self._items))
        idx = self.rng.choice(len(self._items), size=n, replace=False)
        return [self._items[int(i)] for i in idx]

    def all(self) -> list[dict[str, Any]]:
        return list(self._items)

    def tasks(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for it in self._items:
            key = str(it.get("task", "?"))
            out[key] = out.get(key, 0) + 1
        return out

    def __len__(self) -> int:
        return len(self._items)

    # ------------------------------------------------------------------ 持久化

    def save(self, path: str | Path) -> str:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {"capacity": self.capacity, "seen": self._seen, "dropped": self.dropped,
                   "items": [{k: (np.asarray(v).tolist() if isinstance(v, np.ndarray)
                                  or str(type(v)).startswith("<class 'numpy") else v)
                              for k, v in it.items()} for it in self._items]}
        p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return str(p)

    @classmethod
    def load(cls, path: str | Path, *, capacity: int | None = None, seed: int = 0) -> "ReservoirReplay":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        rep = cls(capacity or payload.get("capacity") or len(payload.get("items", [])), seed=seed)
        for it in payload.get("items", []):
            it = {**it, "obs": np.asarray(it["obs"], dtype=np.float32),
                  "actions": np.asarray(it["actions"], dtype=np.int64)}
            if "z_target" in it:
                it["z_target"] = np.asarray(it["z_target"], dtype=np.float32)
            if "mem" in it:
                it["mem"] = np.asarray(it["mem"], dtype=np.float32)
            rep._items.append(it)
        rep._seen = int(payload.get("seen", len(rep._items)))
        rep.dropped = int(payload.get("dropped", 0))
        return rep

    def fingerprint(self) -> str:
        """内容指纹：让"这轮重放用了哪些旧经验"可复现、可写进训练记录。"""
        h = hashlib.sha256()
        for it in sorted(self._items, key=lambda x: str(x.get("task", ""))):
            h.update(np.asarray(it["actions"]).tobytes())
            h.update(np.asarray(it["obs"]).astype(np.float32).tobytes())
        return h.hexdigest()[:16]


def mix_batch(new_items: list[dict[str, Any]], replay: ReservoirReplay, ratio: float,
              n: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """按比例把重放样本插进本轮批次：返回 (新任务部分, 重放部分)。

    ratio<=0 或缓冲为空时重放部分为空 —— 消融"不用经验回放"就是这条路径，
    代码分支只有一处，不靠外部脚本改行为。
    """
    if not new_items:
        raise ValueError("新任务样本不能为空")
    if ratio <= 0 or len(replay) == 0:
        return new_items[:n], []
    n_rep = min(int(round(n * ratio)), len(replay))
    n_new = max(1, n - n_rep)
    picked = [new_items[i % len(new_items)] for i in np.random.default_rng(0).choice(
        len(new_items), size=n_new, replace=n_new > len(new_items))]
    return picked, replay.sample(n_rep)
