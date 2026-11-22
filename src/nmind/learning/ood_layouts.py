"""H3b/H4b：非退化 OOD 评测集生成器（M2.2-D，任务书第 6 节）。

H3b —— 未见障碍拓扑（版本化 h3b-v1）：
    四个模板族（每族一个可参数化布局，dev/holdout 用不同实例种子）：
      wall_gap  竖墙留一个缺口（阻断型，start/goal 强制异侧）
      corridor  两条平行横墙形成窄走廊
      pocket    U 形死胡同（凹口朝向随机）
      scattered 散布 3-4 个障碍
    每个 episode 的 (start,goal) 必须 BFS 可解（4 连通绕障碍）；
    密度=障碍数/36 记录在案。标签仍由 datasets.oracle_action 计算（可审计）。

H4b —— 非退化规则反转：
    逐 episode 过滤：遮挡期内"原规则 oracle 动作 != 反转规则 oracle 动作"的步数
    占比必须 ≥ min_conflict（默认 0.5），否则重采样（上限 max_attempts）。
    同时输出 coincident_frac（两规则巧合相同的占比）——
    这是"模型在 H4b 上的准确率下限参照"（巧合步上两规则给出同一动作）。

拆分策略：dev（调参可用）/ holdout（封存）用不同的实例种子与生成种子，
模板族相同、布局实例不同；holdout 在报告前只评一次。
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import asdict
from typing import Any

import numpy as np

from . import datasets as D
from .datasets import GridTask, Trajectory, _encode, oracle_action

TOPOLOGY_VERSION = "h3b-v2-observable-bfs"
FAMILIES = ("wall_gap", "corridor", "pocket", "scattered")
SIZE = 6
OBS_DIM_WITH_MAP = D.OBS_DIM + SIZE * SIZE
OBS_DIM_WITH_RULE_CUE = D.OBS_DIM + 1


def make_topology(family: str, rng: np.random.Generator) -> tuple[tuple[int, int], ...]:
    """一个族的布局实例。返回障碍格列表。"""
    if family == "wall_gap":
        gap_y = int(rng.integers(1, SIZE - 1))
        cells = [(3, y) for y in range(SIZE) if y != gap_y]
        return tuple(cells)
    if family == "corridor":
        y1 = int(rng.integers(1, 3))
        y2 = y1 + 2
        x_lo, x_hi = 1, 4
        cells = [(x, y1) for x in range(x_lo, x_hi + 1)] + \
                [(x, y2) for x in range(x_lo, x_hi + 1)]
        return tuple(cells)
    if family == "pocket":
        # U 形死胡同：凹口方向随机（0=上 1=右 2=下 3=左），三边围合
        opening = int(rng.integers(0, 4))
        cells: list[tuple[int, int]] = []
        x0, y0 = int(rng.integers(1, SIZE - 3)), int(rng.integers(1, SIZE - 3))
        for dx in range(3):
            for dy in range(3):
                edge = dx in (0, 2) or dy == 2
                # 去掉凹口方向的边中点
                if opening == 0 and dx == 1 and dy == 0:
                    edge = False
                if opening == 1 and dx == 2 and dy == 1:
                    edge = False
                if opening == 2 and dx == 1 and dy == 2:
                    edge = False
                if opening == 3 and dx == 0 and dy == 1:
                    edge = False
                if edge:
                    cells.append((x0 + dx, y0 + dy))
        return tuple(cells)
    if family == "scattered":
        n = int(rng.integers(3, 5))
        cells: list[tuple[int, int]] = []
        while len(cells) < n:
            c = (int(rng.integers(0, SIZE)), int(rng.integers(0, SIZE)))
            if c not in cells:
                cells.append(c)
        return tuple(cells)
    raise ValueError(f"未知拓扑族 {family!r}；可选 {FAMILIES}")


def solvable(start: tuple[int, int], goal: tuple[int, int],
             obstacles: tuple[tuple[int, int], ...], size: int = SIZE) -> bool:
    """BFS（4 连通绕障碍）：start 能否到达 goal。"""
    if start == goal:
        return True
    obst = set(obstacles)
    seen = {start}
    q = deque([start])
    while q:
        x, y = q.popleft()
        for dx, dy in D.DIRS:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < size and 0 <= ny < size) or (nx, ny) in obst:
                continue
            if (nx, ny) == goal:
                return True
            if (nx, ny) not in seen:
                seen.add((nx, ny))
                q.append((nx, ny))
    return False


def shortest_path_action(state_xy: tuple[int, int], goal: tuple[int, int],
                         obstacles: tuple[tuple[int, int], ...],
                         size: int = SIZE) -> int:
    """返回确定性的 BFS 最短路首步；动作编号的并列顺序沿用 ``D.DIRS``。"""
    if state_xy == goal:
        # 环境没有 stay 动作；到达目标后沿用基础任务的确定性邻居选择。
        return oracle_action(state_xy, goal, set(obstacles), size,
                             reversed_control=False)
    obst = set(obstacles)
    q = deque([state_xy])
    first: dict[tuple[int, int], int | None] = {state_xy: None}
    while q:
        x, y = q.popleft()
        for action, (dx, dy) in enumerate(D.DIRS):
            nxt = (x + dx, y + dy)
            if not (0 <= nxt[0] < size and 0 <= nxt[1] < size):
                continue
            if nxt in obst or nxt in first:
                continue
            first[nxt] = action if first[(x, y)] is None else first[(x, y)]
            if nxt == goal:
                return int(first[nxt])
            q.append(nxt)
    raise ValueError(f"不可达的 start/goal：{state_xy} -> {goal}")


def encode_observable_map(x: int, y: int, gx: int, gy: int, *, occluded: bool,
                          collided: bool,
                          obstacles: tuple[tuple[int, int], ...]) -> np.ndarray:
    """基础观测后追加完整 6×6 障碍图，避免 oracle 使用模型不可见的信息。"""
    grid = np.zeros((SIZE * SIZE,), dtype=np.float32)
    for ox, oy in obstacles:
        grid[ox * SIZE + oy] = 1.0
    return np.concatenate([_encode(x, y, gx, gy, occluded, collided), grid])


def trajectory_with(task: GridTask, start: tuple[int, int], goal: tuple[int, int],
                    rng: np.random.Generator, T: int) -> Trajectory:
    """与 datasets.make_trajectory 相同的编码/标签逻辑，但 start/goal 由调用方指定
    （H3b 需要"异侧/可解"这类布局相关的采样约束，不能让生成器内部随机）。"""
    obstacles = set(task.obstacles)
    obs = np.zeros((T, OBS_DIM_WITH_MAP), dtype=np.float32)
    acts = np.zeros((T,), dtype=np.int64)
    memory = np.zeros((T,), dtype=np.int64)
    pos = start
    collided = False
    last: int | None = None
    vis = max(1, task.goal_visible_steps)
    for t in range(T):
        occluded = t >= vis
        obs[t] = encode_observable_map(
            pos[0], pos[1], goal[0], goal[1], occluded=occluded,
            collided=collided, obstacles=task.obstacles)
        acts[t] = shortest_path_action(pos, goal, task.obstacles, task.size)
        memory[t] = goal[0] * task.size + goal[1]
        dx, dy = D.DIRS[acts[t]]
        nx, ny = pos[0] + dx, pos[1] + dy
        if 0 <= nx < task.size and 0 <= ny < task.size and (nx, ny) not in obstacles:
            collided = False
            pos = (nx, ny)
        else:
            collided = True
        last = int(acts[t])
    return Trajectory(obs=obs, actions=acts, goal=goal, start=start,
                      memory_target=memory)


def sample_solvable_pair(obstacles: tuple[tuple[int, int], ...], rng,
                         attempts: int = 100) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """采样一对 BFS 可解的 (start, goal)。wall_gap 的异侧约束由调用方另行校验。"""
    for _ in range(attempts):
        s = (int(rng.integers(0, SIZE)), int(rng.integers(0, SIZE)))
        g = (int(rng.integers(0, SIZE)), int(rng.integers(0, SIZE)))
        if s == g or s in set(obstacles) or g in set(obstacles):
            continue
        if not solvable(s, g, obstacles):
            continue
        return s, g
    return None


def make_h3b_split(split: str, *, n: int, seed: int, T: int = 24,
                   goal_visible_steps: int = 4) -> dict[str, Any]:
    """一个 H3b split：四族各 n//4 条，每族一个布局实例（由实例种子决定）。"""
    if split not in {"dev", "holdout"}:
        raise ValueError("split 必须是 'dev' 或 'holdout'")
    if n < len(FAMILIES) or n % len(FAMILIES):
        raise ValueError(f"n 必须是 {len(FAMILIES)} 的正整数倍，实际 {n}")
    rng = np.random.default_rng(seed)
    per_family = max(1, n // len(FAMILIES))
    episodes: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for family in FAMILIES:
        layout = make_topology(family, rng)
        if family in {"corridor", "pocket"} and split == "holdout":
            # 离散模板实例有限；用方向分区保证 dev/holdout 不重合。
            layout = tuple((y, x) for x, y in layout)
        task = GridTask(name=f"h3b_{family}", goal_visible_steps=goal_visible_steps,
                        obstacles=layout, seed=seed)
        made = 0
        tries = 0
        while made < per_family and tries < per_family * 20:
            tries += 1
            if family == "wall_gap":
                # 定向采样：start/goal 强制分居 x=3 两侧（缺口在 y 轴上仍需绕行）
                s = (int(rng.integers(0, 3)), int(rng.integers(0, SIZE)))
                g = (int(rng.integers(4, SIZE)), int(rng.integers(0, SIZE)))
                if s in set(layout) or g in set(layout) or not solvable(s, g, layout):
                    continue
            else:
                pair = sample_solvable_pair(layout, rng)
                if pair is None:
                    continue
                s, g = pair
            tr = trajectory_with(task, s, g, rng, T)
            episodes.append((family, tr))
            made += 1
        audit.append({"family": family, "layout": [list(c) for c in layout],
                      "n_obstacles": len(layout),
                      "density": round(len(layout) / (SIZE * SIZE), 3),
                      "n_episodes": made, "attempts": tries})
    rows = [_row_of(tr, f"h3b_{family}") for family, tr in episodes]
    manifest = {"kind": "H3b", "split": split, "topology_version": TOPOLOGY_VERSION,
                "seed": seed, "T": T, "goal_visible_steps": goal_visible_steps,
                "obs_dim": OBS_DIM_WITH_MAP, "oracle": "bfs_shortest_path",
                "audit": audit, "n": len(rows)}
    manifest["manifest_hash"] = _content_hash(manifest, rows)
    return {"manifest": manifest, "rows": rows}


def _row_of(tr: Trajectory, task_name: str) -> dict[str, Any]:
    obs = np.asarray(tr.obs, dtype=np.float32)
    return {"obs": obs, "actions": np.asarray(tr.actions, dtype=np.int64),
            "goal_code": int(tr.goal[0]) * 6 + int(tr.goal[1]),
            "len": int(obs.shape[0]), "task": task_name,
            "goal": list(tr.goal), "start": list(tr.start)}


def _content_hash(manifest: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    """对实际样本内容而非仅元数据取指纹。"""
    metadata = {k: v for k, v in manifest.items() if k != "manifest_hash"}
    h = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode())
    for row in rows:
        h.update(np.asarray(row["obs"], dtype=np.float32).tobytes())
        h.update(np.asarray(row["actions"], dtype=np.int64).tobytes())
        h.update(json.dumps({"start": row["start"], "goal": row["goal"],
                             "task": row["task"]}, sort_keys=True).encode())
    return h.hexdigest()[:16]


def conflict_fraction(tr: Trajectory, *, vis: int = 4, size: int = 6) -> tuple[float, float]:
    """一个反转轨迹上：遮挡期内 原规则动作 != 反转动作 的占比，
    以及两规则恰好给出同一动作的占比（巧合占比——H4b 准确率的下限参照）。"""
    conflict = coincident = occ = 0
    for t in range(len(tr)):
        if t < vis:
            continue
        pos = decode_pos(tr.obs[t])
        a_normal = oracle_action(pos, tr.goal, tuple(), size, reversed_control=False)
        conflict += int(a_normal != int(tr.actions[t]))
        coincident += int(a_normal == int(tr.actions[t]))
        occ += 1
    return (conflict / max(1, occ), coincident / max(1, occ))


def decode_pos(obs_row: np.ndarray) -> tuple[int, int]:
    return int(np.argmax(obs_row[0:6])), int(np.argmax(obs_row[6:12]))


def make_h4b_split(split: str, *, n: int, seed: int, T: int = 16,
                   goal_visible_steps: int = 4, min_conflict: float = 0.5,
                   max_attempts: int = 60) -> dict[str, Any]:
    """非退化规则反转 split：逐 episode 过滤低冲突样本。"""
    if split not in {"dev", "holdout"}:
        raise ValueError("split 必须是 'dev' 或 'holdout'")
    rng = np.random.default_rng(seed)
    task = GridTask(name="h4b", goal_visible_steps=goal_visible_steps,
                    reversed_control=True, seed=seed)
    episodes: list[Trajectory] = []
    fracs: list[float] = []
    attempts_used = []
    for _ in range(n):
        best: Trajectory | None = None
        best_frac = -1.0
        used = 0
        for _try in range(max_attempts):
            used += 1
            tr = D.make_trajectory(task, rng, T)
            f, _c = conflict_fraction(tr)
            if f > best_frac:
                best, best_frac = tr, f
            if f >= min_conflict:
                break
        episodes.append(best)
        fracs.append(best_frac)
        attempts_used.append(used)
    rows = [_row_of(tr, "h4b") for tr in episodes]
    manifest = {"kind": "H4b", "split": split, "seed": seed, "T": T,
                "min_conflict": min_conflict,
                "conflict_frac_mean": round(float(np.mean(fracs)), 4),
                "conflict_frac_min": round(float(np.min(fracs)), 4),
                "coincident_frac_mean": round(float(np.mean(
                    [conflict_fraction(e)[1] for e in episodes])), 4),
                "attempts_mean": round(float(np.mean(attempts_used)), 1),
                "n": len(rows)}
    manifest["manifest_hash"] = _content_hash(manifest, rows)
    return {"manifest": manifest, "rows": rows}


def _rule_trajectory(start: tuple[int, int], goal: tuple[int, int], *,
                     reversed_control: bool, T: int, goal_visible_steps: int) -> Trajectory:
    """无障碍规则任务；最后一维在每一步显式给出规则提示。"""
    obs = np.zeros((T, OBS_DIM_WITH_RULE_CUE), dtype=np.float32)
    acts = np.zeros((T,), dtype=np.int64)
    memory = np.zeros((T,), dtype=np.int64)
    pos = start
    last: int | None = None
    for t in range(T):
        occluded = t >= max(1, goal_visible_steps)
        base = _encode(pos[0], pos[1], goal[0], goal[1], occluded, False)
        obs[t] = np.concatenate([base, np.array([float(reversed_control)], dtype=np.float32)])
        action = oracle_action(pos, goal, tuple(), SIZE,
                               reversed_control=reversed_control, last_action=last)
        acts[t] = action
        memory[t] = goal[0] * SIZE + goal[1]
        dx, dy = D.DIRS[action]
        pos = (pos[0] + dx, pos[1] + dy)
        last = int(action)
    return Trajectory(obs=obs, actions=acts, goal=goal, start=start,
                      memory_target=memory)


def make_h4b_cued_split(split: str, *, n: int, seed: int, T: int = 16,
                        goal_visible_steps: int = 4,
                        min_conflict: float = 0.5) -> dict[str, Any]:
    """显式规则提示的 H4b v2；正常/反转样本成对共享 start/goal。"""
    if split not in {"dev", "holdout"}:
        raise ValueError("split 必须是 'dev' 或 'holdout'")
    if n < 2 or n % 2:
        raise ValueError("n 必须是大于等于 2 的偶数")
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    pairs = 0
    attempts = 0
    while pairs < n // 2:
        attempts += 1
        start = (int(rng.integers(0, SIZE)), int(rng.integers(0, SIZE)))
        goal = (int(rng.integers(0, SIZE)), int(rng.integers(0, SIZE)))
        if start == goal:
            continue
        normal = _rule_trajectory(start, goal, reversed_control=False, T=T,
                                  goal_visible_steps=goal_visible_steps)
        reversed_tr = _rule_trajectory(start, goal, reversed_control=True, T=T,
                                       goal_visible_steps=goal_visible_steps)
        fraction, _ = conflict_fraction(reversed_tr, vis=goal_visible_steps)
        if fraction < min_conflict:
            continue
        rows.append(_row_of(normal, "h4b_normal"))
        rows.append(_row_of(reversed_tr, "h4b_reversed"))
        pairs += 1
    manifest = {"kind": "H4b-cued-v2", "split": split, "seed": seed, "T": T,
                "goal_visible_steps": goal_visible_steps, "obs_dim": OBS_DIM_WITH_RULE_CUE,
                "rule_cue_index": D.OBS_DIM, "paired_start_goal": True,
                "min_reversed_conflict": min_conflict, "n": len(rows),
                "attempts": attempts}
    manifest["manifest_hash"] = _content_hash(manifest, rows)
    return {"manifest": manifest, "rows": rows}


def dataset_manifest() -> dict[str, Any]:
    return {"topology_version": TOPOLOGY_VERSION, "families": list(FAMILIES),
            "size": SIZE, "h3b_obs_dim": OBS_DIM_WITH_MAP,
            "h3b_oracle": "bfs_shortest_path", "h4b_min_conflict": 0.5,
            "h4b_cued_obs_dim": OBS_DIM_WITH_RULE_CUE}


__all__ = ["TOPOLOGY_VERSION", "FAMILIES", "OBS_DIM_WITH_MAP", "OBS_DIM_WITH_RULE_CUE",
           "make_topology", "solvable",
           "shortest_path_action", "encode_observable_map",
           "trajectory_with", "sample_solvable_pair", "make_h3b_split",
           "make_h4b_split", "make_h4b_cued_split", "conflict_fraction", "dataset_manifest"]
