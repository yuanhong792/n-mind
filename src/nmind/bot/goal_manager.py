"""目标管理：能力缺口 → 候选学习目标；任务池受 allowlist 硬白名单限制。"""

from __future__ import annotations

from dataclasses import dataclass

from .capability_tracker import CapabilityTracker


@dataclass(frozen=True)
class LearningGoal:
    goal_id: str
    task_family: str
    selection_reason: str
    expected_gap: float | None
    validator_id: str
    max_attempts: int
    failure_rate: float = 0.0


class GoalManager:
    """目标池白名单化：不在 allowlist 内的族永远不能成为目标（外部网页指令不可新增目标）。"""

    def __init__(self, *, allowlist: tuple[str, ...], validator_ids: dict[str, str],
                 max_attempts_per_goal: int = 8, rotation_exclude: int = 2) -> None:
        self.allowlist = tuple(allowlist)
        self.validator_ids = dict(validator_ids)
        self.max_attempts = max_attempts_per_goal
        self.rotation_exclude = int(rotation_exclude)

    def candidates(self, tracker: CapabilityTracker) -> list[LearningGoal]:
        """按（缺口、失败率、族名）确定性排序的目标列表——selection_source=rule。

        历史 bug 两连：
          1. 用 summary()（键是 capability_id）按 family 查 → 恒为 None，
             所有族都被当作"未测试"，目标永远停在第一个族；
          2. 排序只看 gap 不看失败率 → 全错的族（如没有任何工具的 grid_rules）
             与全对的族同分，缺口发现实际不生效。
        """
        goals: list[LearningGoal] = []
        summary = tracker.family_summary()          # 必须按 task_family 取，见 capability_tracker
        for family in self.allowlist:
            rec = summary.get(family)
            failures = rec["failures"] if rec else 0
            attempts = rec["attempts"] if rec else 0
            successes = rec["verified_successes"] if rec else 0
            heldout = rec["heldout_score"] if rec else None
            if attempts == 0:
                gap, reasons = 1.0, ["未测试能力（最大缺口）"]
            else:
                gap = 0.0 if heldout is None else round(max(0.0, 1.0 - heldout), 4)
                if heldout is None:
                    reasons = [f"{attempts} 次尝试、{successes} 次通过，封存集未测（NOT_RUN）"]
                else:
                    reasons = [f"holdout 缺口 {gap}"]
                if failures:
                    reasons.append(f"{failures} 次失败")
                if not failures and gap <= 0.1:
                    reasons.append("维持性练习")
            failure_rate = round(failures / max(1, attempts), 4)
            goals.append(LearningGoal(
                goal_id=f"goal-{family}", task_family=family,
                selection_reason="；".join(reasons) + "（selection_source=rule）",
                expected_gap=gap, validator_id=self.validator_ids.get(family, "unknown"),
                max_attempts=self.max_attempts, failure_rate=failure_rate))
        goals.sort(key=lambda g: (-(g.expected_gap or 0.0), -g.failure_rate, g.goal_id))
        return goals

    def next_goal(self, tracker: CapabilityTracker, *, exclude: tuple[str, ...] = ()) -> LearningGoal:
        """确定性选择：缺口/失败率优先；排除最近练过的族，保证多任务族轮转。

        exclude 传最近练过的族（调用方负责给"足够少"的个数：排太多会把整个
        allowlist 排空、每次都落回兜底第一项，轮转就形同不存在）。
        注意：这是任务书 §4.3 允许的确定性规则选择器（rule），
        不代表 Bot 已具备"学习型探索策略"。
        """
        cands = self.candidates(tracker)
        pool = [g for g in cands if g.task_family not in exclude] or cands
        if pool:
            return pool[0]
        fallback = self.allowlist[0] if self.allowlist else "unknown"
        return LearningGoal(goal_id=f"goal-{fallback}", task_family=fallback,
                            selection_reason="目标池为空，兜底第一族（selection_source=rule）",
                            expected_gap=None,
                            validator_id=self.validator_ids.get(fallback, "unknown"),
                            max_attempts=self.max_attempts)
