"""资源采样：Bot 空闲循环的 CPU 时间与 RSS（任务书 §7 上限核对 + 交付的 CPU 消耗证据）。

口径说明（如实）：
  * cpu_s 用 time.process_time()，是**本进程累计 CPU 时间**（含所有线程），不是墙钟；
  * cpu_percent = Δcpu / Δwall × 100，以单核为 100% 基准，与策略里 max_idle_cpu_percent
    （单核基准）同口径；换算是估算，多线程时可能 >100%；
  * RSS 依赖 psutil（软依赖）：没有 psutil 就只报时间，不假装测了内存（返回 None）。
"""

from __future__ import annotations

import time
from typing import Any


def rss_mb() -> float | None:
    try:
        import psutil
        return round(psutil.Process().memory_info().rss / 2**20, 1)
    except Exception:               # noqa: BLE001 - 软依赖缺失
        return None


def cpu_count() -> int:
    import os
    return max(1, os.cpu_count() or 1)


class Window:
    """一段工作的资源窗口：start() → stop(label) 给出用量。"""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._cpu0 = time.process_time()
        self._wall0 = time.perf_counter()
        self.rss_peak_mb: float | None = None

    def start(self) -> "Window":
        self.reset()
        return self

    def tick(self) -> None:
        cur = rss_mb()
        if cur is not None:
            self.rss_peak_mb = cur if self.rss_peak_mb is None else max(self.rss_peak_mb, cur)

    def stop(self, label: str = "") -> dict[str, Any]:
        wall = time.perf_counter() - self._wall0
        cpu = time.process_time() - self._cpu0
        self.tick()
        cores = cpu_count()
        return {"label": label,
                "wall_s": round(wall, 3),
                "cpu_s": round(cpu, 3),
                "cpu_percent_single_core": round(100.0 * cpu / wall, 1) if wall > 0 else None,
                "cpu_percent_of_machine": (round(100.0 * cpu / (wall * cores), 1)
                                            if wall > 0 else None),
                "cpu_cores": cores,
                "rss_mb": rss_mb(),
                "rss_peak_mb": self.rss_peak_mb,
                "note": "cpu_s 是本进程所有线程的累计 CPU 时间；"
                        "cpu_percent_of_machine = cpu_s /(wall×逻辑核数)，与策略上限同口径"}
