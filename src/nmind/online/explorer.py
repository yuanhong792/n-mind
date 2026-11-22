"""确定性探索策略（第一阶段）：在可验证任务族中按 verified 经验数挑下一个任务。

诚实口径：这是人工设置的确定性调度（最少经验优先 + 字典序平局），不是
"模型自主形成的探索意识"。策略本身的指标单独评估（M4 任务书第 4 节）。
"""

from __future__ import annotations

from typing import Any


def pick_task_family(verified_counts: dict[str, int]) -> dict[str, Any]:
    """返回最少 verified 经验的任务族（平局取字典序最小），附完整依据。"""
    if not verified_counts:
        raise ValueError("verified_counts 为空")
    ranked = sorted(verified_counts.items(), key=lambda kv: (kv[1], kv[0]))
    family, count = ranked[0]
    return {"chosen_family": family, "verified_count": count,
            "rule": "最少 verified 经验优先，平局取字典序（确定性，无随机）",
            "ranking": [{"family": f, "count": c} for f, c in ranked]}
