"""C1.1 课程：任务分层均衡采样 + 能力晋级门槛 + 历史回放。

采样比例：当前重点60% / 未达标基础20% / 已学回放20%（无已学时转给当前）。
能力必须连续两个评估点达标才认定稳定；不达标就缩小任务难度。
"""
from __future__ import annotations

import random
from collections import defaultdict

# 支线A 序列学习（由易到难）
LINE_A = ["A1_copy1", "A2_copyN", "A3_index", "A4_reverse"]
# 支线B 变量记忆（由易到难）
LINE_B = ["B2_bind", "B5_reassign", "B6_indirect", "B2_counter", "B4_bind"]

# 晋级门槛（任务 -> 阈值）
GATES = {
    "A1_copy1": 0.98,
    "A2_copyN": 0.95,   # 2-4字符
    "A3_index": 0.90,
    "A4_reverse": 0.90,
    "B2_bind": 0.90,
    "B4_bind": 0.80,
    "B2_counter": 0.80,
    "B5_reassign": 0.85,
    "B6_indirect": 0.80,
}

STAGES = [
    # 阶段1：最小学习能力（128步）
    {"id": "S1", "until": 128,
     "focus": ["A1_copy1", "A2_copyN", "B2_bind"],
     "note": "确认参数更新/Loss下降/非空输出"},
    # 阶段2：基础序列（累计640步）
    {"id": "S2", "until": 640,
     "focus": ["A1_copy1", "A2_copyN", "A3_index", "A4_reverse"],
     "note": "复制/位置/双字符反转；变量回放"},
    # 阶段3：变量绑定（累计1536步）
    {"id": "S3", "until": 1536,
     "focus": ["B2_bind", "B4_bind", "B2_counter", "B5_reassign", "B6_indirect"],
     "note": "多变量/反事实/重赋值/间接引用；序列回放"},
    # 阶段4：巩固与均未达标补齐（累计2048）
    {"id": "S4", "until": 2048,
     "focus": [],  # 动态：所有未达标任务
     "note": "补齐未达标基础能力，不进入数学/编程"},
]


def stage_for(step):
    for s in STAGES:
        if step < s["until"]:
            return s
    return STAGES[-1]


class GateTracker:
    """连续两个评估点达标才认定稳定。"""
    def __init__(self):
        self.history = defaultdict(list)
        self.stable = set()

    def update(self, per_task_acc):
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
        return dict(self.stable)

    def failing(self, all_tasks):
        return [t for t in all_tasks if t in GATES and t not in self.stable]


class BalancedSampler:
    """按任务分层采样：60%当前 / 20%未达标基础 / 20%已学回放。"""
    def __init__(self, rows_by_task, seed=20261124):
        self.pool = rows_by_task
        self.rng = random.Random(seed)

    def sample(self, focus, failing, stable, batch_size):
        focus = list(focus)
        failing_base = [t for t in failing if t not in focus]
        replay = [t for t in stable if t not in focus]

        n_focus = int(batch_size * 0.6)
        n_fail = int(batch_size * 0.2)
        n_replay = batch_size - n_focus - n_fail

        # 无已学任务时回放配额转给当前
        if not replay:
            n_focus += n_replay
            n_replay = 0
        # 无未达标任务时转给当前
        if not failing_base:
            n_focus += n_fail
            n_fail = 0

        idxs = []
        idxs += self._draw(focus, n_focus)
        idxs += self._draw(failing_base, n_fail)
        idxs += self._draw(replay, n_replay)
        self.rng.shuffle(idxs)
        return idxs

    def _draw(self, tasks, n):
        if n <= 0 or not tasks:
            return []
        # 任务间均匀
        out = []
        for i in range(n):
            t = tasks[i % len(tasks)]
            pool = self.pool.get(t, [])
            if not pool:
                continue
            out.append(self.rng.choice(pool))
        return out
