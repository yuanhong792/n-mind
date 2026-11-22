"""调度：用户任务优先 + 空闲探索 + 冷却（复用 online.scheduler.decide 的判定口径）。

冷却时间来源唯一化（历史 bug）：原来读 SchedulerConfig.min_cooldown_seconds，
而这个 dataclass 只有 cooldown_s，第一次调用侥幸走 `is None` 分支，第二次即 AttributeError。
现在冷却秒数由 BotPolicy.min_cooldown_seconds 显式注入，SchedulerConfig 只负责候选更新触发条件。
"""

from __future__ import annotations

import time
from typing import Any

from ..online.scheduler import SchedulerConfig, decide


class BotScheduler:
    def __init__(self, cfg: SchedulerConfig | None = None, *,
                 cooldown_s: float = 60.0,
                 clock=time.monotonic) -> None:
        self.cfg = cfg or SchedulerConfig()
        self.cooldown_s = float(cooldown_s)
        self.clock = clock
        self._last_idle_ts: float | None = None

    def should_idle_learn(self, *, now: float | None = None,
                          user_queue_len: int = 0) -> bool:
        if user_queue_len > 0:
            return False                                  # 用户任务优先（SERVING）
        if self._last_idle_ts is None:
            return True
        return (now if now is not None else self.clock()) - self._last_idle_ts >= self.cooldown_s

    def cooldown_remaining(self, *, now: float | None = None) -> float:
        if self._last_idle_ts is None:
            return 0.0
        left = self.cooldown_s - ((now if now is not None else self.clock()) - self._last_idle_ts)
        return round(max(0.0, left), 3)

    def mark_idle_done(self, now: float | None = None) -> None:
        self._last_idle_ts = now if now is not None else self.clock()

    def decide(self, *, verified_count: int, budget_remaining: int,
               now: float | None = None) -> dict[str, Any]:
        """候选更新触发判定：verified 计数 + 硬预算 + 冷却，三者缺一不可。"""
        return decide(verified_count=verified_count, rejected_count=0, uncertain_count=0,
                       budget_remaining=budget_remaining,
                       cooldown_ok=self.cooldown_remaining(now=now) <= 0.0, cfg=self.cfg)
