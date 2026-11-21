"""ComputeGovernor：把"算多少"变成外部可审计的硬约束。

约束来源是配置文件与设备实测，而不是模型自己的判断：模型可以建议加深计算，
但上限由 Governor 给定，且必须有有限停止条件。任何超时/超内存都会降档并记录，
不允许静默放宽预算。
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import psutil
import torch


@dataclass
class BudgetDecision:
    k: int
    depth: int
    reason: str
    step_ms: float
    rss_mb: float
    degraded: bool = False


@dataclass
class GovernorReport:
    steps: int = 0
    degraded_steps: int = 0
    k_hist: list[int] = field(default_factory=list)
    depth_hist: list[int] = field(default_factory=list)
    last_step_ms: float = 0.0
    peak_step_ms: float = 0.0
    peak_rss_mb: float = 0.0
    stopped_reason: str = ""

    def summary(self) -> dict:
        n = max(1, self.steps)
        return {
            "steps": self.steps,
            "degraded_ratio": round(self.degraded_steps / n, 4),
            "mean_k": round(sum(self.k_hist) / n, 3),
            "max_depth": max(self.depth_hist) if self.depth_hist else 0,
            "last_step_ms": round(self.last_step_ms, 3),
            "peak_step_ms": round(self.peak_step_ms, 3),
            "peak_rss_mb": round(self.peak_rss_mb, 1),
            "stopped_reason": self.stopped_reason,
        }


class ComputeGovernor:
    def __init__(
        self,
        k_default: int = 2,
        k_min: int = 1,
        k_max: int = 8,
        max_step_ms: float = 50.0,
        max_rss_mb: float = 4096.0,
        max_depth: int = 1,
        window: int = 8,
    ) -> None:
        if k_min < 1 or k_min > k_max:
            raise ValueError("需要 1 <= k_min <= k_max")
        self.k = int(max(k_min, min(k_default, k_max)))
        self.k_min, self.k_max = k_min, k_max
        self.max_step_ms = float(max_step_ms)
        self.max_rss_mb = float(max_rss_mb)
        self.max_depth = int(max(1, max_depth))
        self.depth = 1
        self.durations: deque[float] = deque(maxlen=window)
        self.report = GovernorReport()
        self._t0: float | None = None

    # ------------------------------------------------------------------ 计时

    def start_step(self) -> None:
        self._t0 = time.perf_counter()

    def end_step(self) -> float:
        if self._t0 is None:
            return 0.0
        ms = (time.perf_counter() - self._t0) * 1000.0
        self._t0 = None
        self.durations.append(ms)
        self.report.steps += 1
        self.report.last_step_ms = ms
        self.report.peak_step_ms = max(self.report.peak_step_ms, ms)
        return ms

    # ------------------------------------------------------------------ 资源

    @staticmethod
    def rss_mb() -> float:
        try:
            return psutil.Process().memory_info().rss / 1e6
        except Exception:
            return 0.0

    def available_mb(self) -> float:
        try:
            return psutil.virtual_memory().available / 1e6
        except Exception:
            return float("inf")

    # ------------------------------------------------------------------ 决策

    def allocate(self, err: float = 0.0, reason_hint: str = "") -> BudgetDecision:
        """按最近耗时与内存决定本步 k 与额外迭代深度。

        规则刻意保持简单可预测：慢/紧就降档，快且预测误差高才考虑加深，
        并且加深受 max_depth 硬上限约束。误差高不等于一定要多算——是否有利
        由 ablations 的 no_error_gating 对照来回答。
        """
        rss = self.rss_mb()
        self.report.peak_rss_mb = max(self.report.peak_rss_mb, rss)
        mean_ms = sum(self.durations) / len(self.durations) if self.durations else 0.0
        degraded = False
        reason = reason_hint or "nominal"
        k, depth = self.k, self.depth

        if rss > self.max_rss_mb:
            k, depth, degraded, reason = self.k_min, 1, True, f"rss>{self.max_rss_mb:.0f}MB"
        elif mean_ms > self.max_step_ms:
            k = max(self.k_min, k - 1)
            depth = 1
            degraded, reason = True, f"step_ms>{self.max_step_ms:.0f}"
        elif err > 0.75 and mean_ms < 0.5 * self.max_step_ms:
            depth = min(self.max_depth, depth + 1)
            k = min(self.k_max, k + 1)
            reason = "high_error_with_headroom"
        else:
            depth = max(1, self.max_depth if err > 0.5 else 1)

        self.k, self.depth = k, depth
        self.report.k_hist.append(k)
        self.report.depth_hist.append(depth)
        if degraded:
            self.report.degraded_steps += 1
        return BudgetDecision(k=k, depth=depth, reason=reason, step_ms=mean_ms,
                              rss_mb=rss, degraded=degraded)

    def should_stop(self, steps_done: int, hard_step_limit: int) -> bool:
        """有限停止：达到硬步数上限就停，绝不依赖"模型觉得差不多了"。"""
        if steps_done >= hard_step_limit:
            self.report.stopped_reason = f"hard_step_limit={hard_step_limit}"
            return True
        if self.rss_mb() > self.max_rss_mb * 1.25:
            self.report.stopped_reason = "rss_guard"
            return True
        return False

    def set_torch_threads(self, n: int) -> None:
        if n and n > 0:
            torch.set_num_threads(n)
