"""学习触发调度：显式、可观测的触发条件（人工设置的调度逻辑，非"自主意识"）。

M4 任务书第 4 节：触发条件如 verified_experiences >= N、budget_remaining > 0、
cooldown_satisfied。只有 verified 经验计数参与触发（rejected/uncertain 不算）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

DECISIONS = ("MEMORY_ONLY", "CANDIDATE_WEIGHT_UPDATE")


@dataclass
class SchedulerConfig:
    verified_threshold: int = 32       # 需要积累的 verified 经验数
    cooldown_s: float = 0.0            # 两次候选更新之间的最小间隔
    max_updates_per_batch: int = 120   # 单次候选更新的硬预算（optimizer.step）


def decide(*, verified_count: int, rejected_count: int, uncertain_count: int,
           budget_remaining: int, cooldown_ok: bool,
           cfg: SchedulerConfig | None = None,
           novelty_score: float = 1.0, novelty_threshold: float = 0.0) -> dict[str, Any]:
    cfg = cfg or SchedulerConfig()
    reasons: list[str] = []
    if verified_count < cfg.verified_threshold:
        reasons.append(f"verified_count {verified_count} < {cfg.verified_threshold}")
    if budget_remaining <= 0:
        reasons.append("budget_remaining <= 0")
    if not cooldown_ok:
        reasons.append("cooldown 未满足")
    if novelty_score < novelty_threshold:
        reasons.append(f"novelty {novelty_score} < {novelty_threshold}")
    if reasons:
        return {"decision": "MEMORY_ONLY", "reasons": reasons,
                "verified_count": verified_count, "rejected_count": rejected_count,
                "uncertain_count": uncertain_count}
    return {"decision": "CANDIDATE_WEIGHT_UPDATE", "reasons": ["触发条件全部满足"],
            "verified_count": verified_count, "rejected_count": rejected_count,
            "uncertain_count": uncertain_count,
            "note": "rejected/uncertain 经验不参与触发计数，也不进入学习流"}
