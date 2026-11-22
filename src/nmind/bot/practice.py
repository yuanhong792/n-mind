"""安全练习环境：三类程序可验证任务（算术 / JSON 字段提取 / 网格规则）。

* 每个任务实例由程序生成、由独立 oracle 判分——成功/失败/无法验证三态；
* 神经尝试与工具执行**分开计量**（任务书 §5 分层量化）；
* 工具仅调用白名单实现（skills.toolbox），模型文本永不拼入 shell。
"""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any

from ..skills.toolbox import calculator, json_extractor

GRID = 3


def task_fingerprint(task: dict[str, Any]) -> str:
    """题面+答案的内容指纹：同一实例跨轮次可识别（重启不重复执行、sealed 不入库都靠它）。"""
    blob = json.dumps({k: task[k] for k in ("family", "question", "answer")},
                       ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def make_task(family: str, rng: random.Random, *, index: int = 0) -> dict[str, Any]:
    """生成一个任务实例：question（给神经/展示）、answer（oracle）、verifier 元数据。

    task_id 含 rng 种子流与序号，保证同一 seed 的循环可复现、不同实例可去重。
    """
    tag = f"{index:02d}-{rng.randint(0, 9999):04d}"
    if family == "arithmetic":
        a, b = rng.randint(2, 19), rng.randint(2, 19)
        q = f"请计算：{a}加{b}等于多少？"
        task = {"task_id": f"arith-{a:02d}-{b:02d}-{tag}",
                 "family": family, "question": q, "answer": str(a + b),
                 "tool": "calculator", "tool_args": {"expression": f"{a}+{b}"}}
    elif family == "json_extract":
        names = ["张三", "李四", "王五", "赵六"]
        cities = ["北京", "上海", "广州", "深圳"]
        doc = {"name": rng.choice(names), "city": rng.choice(cities),
                "age": rng.randint(18, 60)}
        field = rng.choice(["name", "city", "age"])
        # 题面必须自带数据：原来只写"从数据里取出字段 age 的值"，同一句题面配不同答案，
        # 既是矛盾监督（同 prompt 两个 target），又让封存集按题面去重时出现假"未污染"。
        q = f"JSON字段提取：{json.dumps(doc, ensure_ascii=False)} 中字段 {field} 的值是多少？"
        task = {"task_id": f"json-{field}-{tag}",
                 "family": family, "question": q, "payload": json.dumps(doc, ensure_ascii=False),
                 "answer": str(doc[field]), "tool": "json_extractor",
                 "tool_args": {"payload": json.dumps(doc, ensure_ascii=False),
                                "field_path": field}}
    elif family == "grid_rules":
        # 环形网格：size 与方向一起随机，实例空间 2·s³·(s-1)（s=3..5 共 746 个），
        # 避免"18 个实例练完就没得练"导致的假饱和（原固定 3x3 只能造 18 题）。
        size = rng.choice([GRID, 4, 5])
        x, y = rng.randint(0, size - 1), rng.randint(0, size - 1)
        k = rng.randint(1, size - 1)
        direction = rng.choice(["下", "上"])
        ny = (y + k) % size if direction == "下" else (y - k) % size
        q = (f"网格规则：{size}x{size} 网格环形上下移动。"
             f"从坐标 ({x},{y}) 向{direction}走 {k} 步后的纵坐标是多少？只回答数字。")
        task = {"task_id": f"grid-{size}-{x}{y}-{direction}{k}-{tag}",
                 "family": family, "question": q, "answer": str(ny),
                 "tool": None, "tool_args": {},
                 "meta": {"size": size, "x": x, "y": y, "k": k, "direction": direction}}
    else:
        raise ValueError(f"未知任务族 {family!r}")
    task["fingerprint"] = task_fingerprint(task)
    return task


def make_sealed_set(family: str, n: int, *, seed: int) -> list[dict[str, Any]]:
    """封存评测集：独立 RNG 流生成、生成后固定，**永不进入学习流**（任务书 §4/§8）。

    与练习流分开种子的原因：原实现用同一个 rng 一边出题一边"每 4 条抽 1 条封存"，
    封存题与练习题来自同一条随机流且随循环次数漂移，重启后同一实例既可能进训练集
    又可能进封存集——污染评测。这里用独立 seed，并且按内容指纹去重。
    """
    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    guard = 0
    while len(out) < n and guard < n * 200:
        guard += 1
        t = make_task(family, rng, index=9000 + guard)
        key = t["question"]          # 去重口径=题面：同一题面必须有唯一答案，否则评测无意义
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out


def execute_tool(task: dict[str, Any]) -> str:
    """工具执行（白名单实现，独立于神经尝试；贡献单独计量）。"""
    tool = task.get("tool")
    if tool == "calculator":
        return str(calculator(**task["tool_args"]))
    if tool == "json_extractor":
        return str(json_extractor(**task["tool_args"]))
    raise ValueError(f"任务族 {task['family']} 没有注册工具（仅神经可尝试，验证仍由 oracle）")


def verify(task: dict[str, Any], attempt: str) -> str:
    """独立 oracle 判分：verified | rejected | uncertain。"""
    exp = task.get("answer")
    if exp is None:
        return "uncertain"
    a = attempt.strip()
    if a == str(exp):
        return "verified"
    try:
        return "verified" if abs(float(a) - float(exp)) < 1e-9 else "rejected"
    except (TypeError, ValueError):
        return "rejected"
