"""M1 距离课程：两变量→四变量→干扰→距离→重赋值→间接引用。

所有候选模型、所有种子使用同一课程；按 run 的 total_steps 归一化阶段边界。
"""
from __future__ import annotations

import random
from collections import defaultdict

STAGES = [
    {"id": "Q1_two_vars", "start": 0.00, "end": 0.16,
     "tasks": ["B1_bind1", "B2_bind", "B2_query_switch", "A1_copy1"]},
    {"id": "Q2_four_vars", "start": 0.16, "end": 0.32,
     "tasks": ["B4_bind", "A2_copyN", "A3_index"]},
    {"id": "Q3_interference", "start": 0.32, "end": 0.50,
     "tasks": ["B_swap", "A4_reverse", "A5_repeat", "B4_bind"]},
    {"id": "Q4_distance", "start": 0.50, "end": 0.68,
     "tasks": ["E_distance"]},
    {"id": "Q5_reassign", "start": 0.68, "end": 0.84,
     "tasks": ["C_reassign", "E_distance"]},
    {"id": "Q6_indirect", "start": 0.84, "end": 1.01,
     "tasks": ["D_indirect", "B8_bind", "B_rename", "C_reassign"]},
]

GATES = {
    "A1_copy1": 0.98, "A2_copyN": 0.95, "A3_index": 0.90,
    "A4_reverse": 0.90, "B1_bind1": 0.95, "B2_bind": 0.90,
    "B4_bind": 0.80, "B8_bind": 0.70, "B2_query_switch": 0.85,
    "B_swap": 0.80, "B_rename": 0.75, "C_reassign": 0.85,
    "D_indirect": 0.80, "E_distance": 0.80,
}

PANEL_TASKS = ["A1_copy1", "A2_copyN", "A3_index", "A4_reverse", "A5_repeat",
               "B1_bind1", "B2_bind", "B4_bind", "B8_bind", "B2_query_switch",
               "B_swap", "B_rename", "C_reassign", "D_indirect", "E_distance"]


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
    def __init__(self, rows_by_task, seed=20261124):
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
                task = tasks[i % len(tasks)]
                out.append(self.rng.choice(self.pool[task]))
        self.rng.shuffle(out)
        return out
