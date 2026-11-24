"""M3 课程：单字符→连续生成→绑定→状态更新→距离干扰。四架构共用。"""

import random
from collections import defaultdict

STAGES = [
    {"id": "Q1_single", "start": 0.00, "end": 0.14,
     "tasks": ["A1_copy1", "A2_copy2", "A5_repeat"]},
    {"id": "Q2_continuous", "start": 0.14, "end": 0.36,
     "tasks": ["A3_copyN", "A4_reverse", "A1_copy1"]},
    {"id": "Q3_index", "start": 0.36, "end": 0.50,
     "tasks": ["A6_index", "A3_copyN", "A4_reverse"]},
    {"id": "Q4_binding", "start": 0.50, "end": 0.68,
     "tasks": ["B2_bind", "B_rename", "B4_bind", "A3_copyN"]},
    {"id": "Q5_state", "start": 0.68, "end": 0.84,
     "tasks": ["C_reassign", "C_history_current", "C_indirect", "B8_bind"]},
    {"id": "Q6_distance", "start": 0.84, "end": 1.01,
     "tasks": ["D_distance", "D_interference", "B8_bind", "A4_reverse"]},
]

GATES = {
    "A1_copy1": 0.95, "A2_copy2": 0.90, "A3_copyN": 0.85, "A4_reverse": 0.80,
    "A5_repeat": 0.85, "A6_index": 0.85, "B2_bind": 0.85, "B4_bind": 0.75,
    "B8_bind": 0.65, "B_rename": 0.75, "C_reassign": 0.75,
    "C_history_current": 0.75, "C_indirect": 0.70, "D_distance": 0.75,
    "D_interference": 0.75,
}
PANEL_TASKS = ["A1_copy1", "A2_copy2", "A3_copyN", "A4_reverse", "A5_repeat",
               "A6_index", "B2_bind", "B4_bind", "B8_bind", "B_rename",
               "C_reassign", "C_history_current", "C_indirect", "D_distance",
               "D_interference"]


def stage_for(step, total_steps):
    frac = step / max(1, total_steps)
    for s in STAGES:
        if s["start"] <= frac < s["end"]:
            return s
    return STAGES[-1]


class BalancedSampler:
    def __init__(self, rows_by_task, seed=20261201):
        self.pool = rows_by_task
        self.rng = random.Random(seed)

    def sample(self, focus, failing, stable, batch_size):
        focus = [t for t in focus if self.pool.get(t)]
        failing = [t for t in failing if t not in focus and self.pool.get(t)]
        replay = [t for t in stable if t not in focus and self.pool.get(t)]
        n_focus = int(batch_size * 0.6); n_fail = int(batch_size * 0.2)
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
