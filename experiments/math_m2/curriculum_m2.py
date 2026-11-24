"""M2 课程：单字符→多字符连续生成→位置/反转→变量绑定→干扰→距离。

所有候选/种子使用同一课程；按 run 的 total_steps 归一化阶段边界。
PANEL_TASKS 为固定验证面板任务集合（每任务等额抽样）。
"""
from __future__ import annotations

import random
from collections import defaultdict

STAGES = [
    {"id": "M1_single", "start": 0.00, "end": 0.12,
     "tasks": ["S1_copy1", "S5_repeat", "B1_bind"]},
    {"id": "M2_continuous", "start": 0.12, "end": 0.34,
     "tasks": ["S2_copyN", "S3_reverse", "S1_copy1"]},
    {"id": "M3_index", "start": 0.34, "end": 0.48,
     "tasks": ["S4_index", "S2_copyN", "S3_reverse"]},
    {"id": "M4_binding", "start": 0.48, "end": 0.66,
     "tasks": ["B1_bind", "B2_bind", "B_swap", "B4_bind"]},
    {"id": "M5_interference", "start": 0.66, "end": 0.82,
     "tasks": ["C_reassign", "D_indirect", "B_swap", "B8_bind"]},
    {"id": "M6_distance", "start": 0.82, "end": 1.01,
     "tasks": ["E_distance", "B8_bind", "S2_copyN", "S3_reverse"]},
]

GATES = {
    "S1_copy1": 0.95, "S1_copy1_unseen": 0.85, "S2_copyN": 0.85,
    "S3_reverse": 0.80, "S4_index": 0.85, "S5_repeat": 0.85,
    "B1_bind": 0.95, "B2_bind": 0.85, "B4_bind": 0.75, "B8_bind": 0.65,
    "B_swap": 0.75, "C_reassign": 0.75, "D_indirect": 0.70, "E_distance": 0.75,
}

PANEL_TASKS = ["S1_copy1", "S2_copyN", "S3_reverse", "S4_index", "S5_repeat",
               "B1_bind", "B2_bind", "B4_bind", "B8_bind", "B_swap",
               "C_reassign", "D_indirect", "E_distance"]


def stage_for(step: int, total_steps: int) -> dict:
    frac = step / max(1, total_steps)
    for s in STAGES:
        if s["start"] <= frac < s["end"]:
            return s
    return STAGES[-1]


class GateTracker:
    def __init__(self):
        self.history = defaultdict(list)
        self.stable = set()

    def update(self, per_task_acc: dict) -> list[str]:
        for task, acc in per_task_acc.items():
            if task not in GATES:
                continue
            self.history[task].append(acc >= GATES[task])
            if len(self.history[task]) > 4:
                self.history[task].pop(0)
            if len(self.history[task]) >= 2 and all(self.history[task][-2:]):
                self.stable.add(task)
            else:
                self.stable.discard(task)
        return sorted(self.stable)

    def failing(self) -> list[str]:
        return [t for t in PANEL_TASKS if t in GATES and t not in self.stable]


class BalancedSampler:
    def __init__(self, rows_by_task, seed=20261201):
        self.pool = rows_by_task
        self.rng = random.Random(seed)

    def sample(self, focus, failing, stable, batch_size):
        focus = [t for t in focus if self.pool.get(t)]
        failing = [t for t in failing if t not in focus and self.pool.get(t)]
        replay = [t for t in stable if t not in focus and self.pool.get(t)]
        n_focus = int(batch_size * 0.6)
        n_fail = int(batch_size * 0.2)
        n_replay = batch_size - n_focus - n_fail
        if not replay:
            n_focus += n_replay; n_replay = 0
        if not failing:
            n_focus += n_fail; n_fail = 0
        out = []
        for tasks, n in ((focus, n_focus), (failing, n_fail), (replay, n_replay)):
            for i in range(n):
                out.append(self.rng.choice(self.pool[tasks[i % len(tasks)]]))
        self.rng.shuffle(out)
        return out
