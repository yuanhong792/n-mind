"""合成数据：二维网格里物体的位置、移动、遮挡与碰撞。

任务设计（指令第 8 节任务 1/2/3/5 共用一套环境，避免"为每个实验临时造答案"）：

  * 环境 6x6 网格，一个物体、一个目标。每步物体按动作移动一格。
  * 目标位置**只在开头 goal_visible_steps 步可见**，之后被遮挡。
    因此后半段要选对动作，网络必须把目标留在内部状态里（或从记忆里检索回来），
    单靠当前输入不可能算出正确答案 —— 这就是"连续事件记忆"的考点。
  * 撞墙/撞障碍时位置不变，collision 位置 1，作为"碰撞简化状态"。

观测向量（事件编码器的 numeric 输入，长度 26）：
    [0:6]   物体 x  one-hot
    [6:12]  物体 y  one-hot
    [12:18] 目标 x  one-hot（遮挡后全 0）
    [18:24] 目标 y  one-hot（遮挡后全 0）
    [24]    occluded 标志
    [25]    collision 标志

标签：动作 = 4 方向之一，由 oracle 策略给出（曼哈顿距离下降方向，
平局按 上/右/下/左 固定顺序）。遮挡期间 oracle 沿用"记忆中的目标"，
所以标签是可计算的，不是查表硬编码的答案。

训练集/测试集用不同 seed 的轨迹族，绝不重叠。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch

OBS_DIM = 26
N_ACTIONS = 4
DIRS = ((0, -1), (1, 0), (0, 1), (-1, 0))       # 上 右 下 左
DIR_NAMES = ("up", "right", "down", "left")


@dataclass
class GridTask:
    """一个任务定义 = 网格大小 + 障碍布局 + 目标分布 + 规则变体。

    任务 B 通过 reversed_control 换掉动作规则：同一个环境，
    最优动作方向相反（先远离目标再靠近），用来测持续学习的遗忘。
    """
    name: str = "A"
    size: int = 6
    goal_visible_steps: int = 5
    obstacles: tuple[tuple[int, int], ...] = ()
    reversed_control: bool = False
    seed: int = 0

    def key(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:16]


@dataclass
class Trajectory:
    obs: np.ndarray        # [T, OBS_DIM] float32
    actions: np.ndarray    # [T] int64   —— 本步执行的动作（oracle）
    goal: tuple[int, int]
    start: tuple[int, int]
    memory_target: np.ndarray  # [T] int64 —— 遮挡期仍需记住的目标编码，供记忆探针用

    def __len__(self) -> int:
        return int(self.obs.shape[0])


def _onehot(idx: int, size: int) -> np.ndarray:
    v = np.zeros(size, dtype=np.float32)
    if 0 <= idx < size:
        v[idx] = 1.0
    return v


def _encode(x: int, y: int, gx: int, gy: int, occluded: bool, collided: bool) -> np.ndarray:
    """遮挡只作用于**目标**维度：物体自身位置始终可观测。

    否则网络连自己在哪都不知道，任务变得不可解，测出来的差异也就没有解释力。
    """
    out = np.concatenate([
        _onehot(x, 6),
        _onehot(y, 6),
        _onehot(-1 if occluded else gx, 6),
        _onehot(-1 if occluded else gy, 6),
        np.array([1.0 if occluded else 0.0, 1.0 if collided else 0.0], dtype=np.float32),
    ])
    return out.reshape(OBS_DIM)


def oracle_action(state_xy: tuple[int, int], goal: tuple[int, int], obstacles, size: int,
                  *, reversed_control: bool, last_action: int | None = None) -> int:
    """可解释的规则策略：选让曼哈顿距离最小的方向，越界/障碍方向排除。

    遮挡期间仍然用**真实目标**（这是任务标签，不是网络的记忆）。
    reversed_control 时改为最大化距离 —— 表示"同一输入空间、不同决策规则"，
    持续学习实验里就是靠它制造任务冲突。
    """
    x, y = state_xy
    cand: list[tuple[int, int]] = []
    for a, (dx, dy) in enumerate(DIRS):
        nx, ny = x + dx, y + dy
        if not (0 <= nx < size and 0 <= ny < size):
            continue
        if (nx, ny) in obstacles:
            continue
        d = abs(nx - goal[0]) + abs(ny - goal[1])
        cand.append((d, a))
    if not cand:
        return 0 if last_action is None else int(last_action)
    cand.sort(key=lambda t: (t[0] if not reversed_control else -t[0], t[1]))
    return int(cand[0][1])


def make_trajectory(task: GridTask, rng: np.random.Generator, T: int,
                    *, used_pairs: set | None = None) -> Trajectory:
    size = task.size
    obstacles = set(task.obstacles)
    for _ in range(200):
        start = (int(rng.integers(0, size)), int(rng.integers(0, size)))
        goal = (int(rng.integers(0, size)), int(rng.integers(0, size)))
        if start != goal and goal not in obstacles and start not in obstacles:
            # used_pairs 用来保证 train/val/test 的 (起点,目标) 组合互不重叠：
            # 只靠"不同的 RNG 流"不够，36x35 的空间里仍然会撞出同一条轨迹。
            if used_pairs is None or (start, goal) not in used_pairs:
                if used_pairs is not None:
                    used_pairs.add((start, goal))
                break
    obs = np.zeros((T, OBS_DIM), dtype=np.float32)
    acts = np.zeros((T,), dtype=np.int64)
    memory = np.zeros((T,), dtype=np.int64)          # goal 的线性编码；遮挡后仍是同一个值
    pos = start
    collided = False
    last: int | None = None
    vis = max(1, task.goal_visible_steps)
    for t in range(T):
        occluded = t >= vis
        obs[t] = _encode(pos[0], pos[1], goal[0], goal[1], occluded, collided)
        a = oracle_action(pos, goal, obstacles, size, reversed_control=task.reversed_control,
                          last_action=last)
        acts[t] = a
        memory[t] = goal[0] * size + goal[1]
        dx, dy = DIRS[a]
        nx, ny = pos[0] + dx, pos[1] + dy
        if 0 <= nx < size and 0 <= ny < size and (nx, ny) not in obstacles:
            collided = False
            pos = (nx, ny)
        else:
            collided = True                          # 撞墙：位置不变，标志置 1
        last = a
    return Trajectory(obs=obs, actions=acts, goal=goal, start=start, memory_target=memory)


SPLIT_VERSION = "v2-disjoint-pairs"   # 进 dataset_hash：生成器规则变了，旧 hash 自动失效


def _split_rng(seed: int, split: int, i: int) -> np.random.Generator:
    """每个 (split, i) 一条独立的 SeedSequence 流。

    旧实现是 seed*1000 + 1 + i / + 2 + i：test[i] 的流恰好等于 train[i+1] 的流，
    于是测试轨迹与训练轨迹逐字节相同（实测 32/32），0.93 的"泛化"其实是背书。
    SeedSequence([seed, split, i]) 把三个坐标拼成互不重叠的熵源，从构造上杜绝错位。
    """
    return np.random.default_rng(np.random.SeedSequence([int(seed), int(split), int(i)]))


def make_split(task: GridTask, *, n_train: int, n_test: int, T: int, seed: int,
               n_val: int = 0) -> dict[str, Any]:
    """train/val/test 用互不相交的 RNG 流 + 互不重叠的 (起点,目标) 组合。

    验收口径：三个集合的轨迹签名两两无交集，起点-目标对两两无交集
    （tests/test_datasets.py 钉住）。同一个模板只换随机数的"泛化"不算泛化。
    """
    used: set = set()
    tr = [make_trajectory(task, _split_rng(seed, 1, i), T, used_pairs=used)
          for i in range(n_train)]
    va = [make_trajectory(task, _split_rng(seed, 2, i), T, used_pairs=used)
          for i in range(n_val)]
    te = [make_trajectory(task, _split_rng(seed, 3, i), T, used_pairs=used)
          for i in range(n_test)]
    out = {"train": tr, "test": te, "task": task, "T": T, "seed": seed,
           "split_version": SPLIT_VERSION}
    if n_val:
        out["val"] = va
    return out


def stack(trajs: list[Trajectory]) -> dict[str, torch.Tensor]:
    """把一批等长轨迹拼成 [B,T,*] 张量。"""
    if not trajs:
        raise ValueError("空轨迹列表")
    lens = {len(t) for t in trajs}
    if len(lens) != 1:
        raise ValueError(f"轨迹长度不一致：{lens}；请分桶后再批量")
    return {
        "obs": torch.tensor(np.stack([t.obs for t in trajs]), dtype=torch.float32),
        "actions": torch.tensor(np.stack([t.actions for t in trajs]), dtype=torch.long),
        "memory": torch.tensor(np.stack([t.memory_target for t in trajs]), dtype=torch.long),
    }


def batch_tensor(trajs: list[Trajectory], idx: list[int]) -> dict[str, torch.Tensor]:
    return {k: v[idx] for k, v in stack(trajs).items()}


def dataset_hash(task: GridTask, *, n_train: int, n_test: int, T: int, seed: int,
                 n_val: int = 0) -> str:
    payload = {"task": asdict(task), "n_train": n_train, "n_test": n_test, "n_val": n_val,
               "T": T, "seed": seed, "split_version": SPLIT_VERSION}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def difficulty_mix(*, n_easy: int, n_hard: int, seed: int, T: int = 24) -> list[Trajectory]:
    """动态预算实验用：简单样本目标可见很久、几乎不用绕；困难样本早早遮挡且障碍多。

    路由应当在这两组上表现出不同的激活量，第 8 节任务 5 就是量这个。
    """
    easy = GridTask(name="easy", goal_visible_steps=T, obstacles=(), seed=seed)
    hard = GridTask(name="hard", goal_visible_steps=1,
                    obstacles=((2, 2), (2, 3), (3, 2), (4, 1)), seed=seed + 1)
    out = [make_trajectory(easy, np.random.default_rng(7000 + i), T) for i in range(n_easy)]
    out += [make_trajectory(hard, np.random.default_rng(8000 + i), T) for i in range(n_hard)]
    rng = np.random.default_rng(seed)
    rng.shuffle(out)
    return out


def task_to_dict(task: GridTask) -> dict[str, Any]:
    return asdict(task)


__all__ = ["OBS_DIM", "N_ACTIONS", "DIR_NAMES", "GridTask", "Trajectory",
           "make_split", "make_trajectory", "stack", "batch_tensor", "dataset_hash",
           "difficulty_mix", "oracle_action", "task_to_dict"]
