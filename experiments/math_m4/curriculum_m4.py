"""M4 固定课程定义（零梯度）。

用于声明基础符号、真实关联、状态计算、算法泛化四层课程，
不执行训练、不修改数据、不提供答案查找。

运行：python curriculum_m4.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

STAGES = [
    ("L1_symbolic", ["A1_copy1", "A2_copy2", "A3_copyN", "A5_repeat",
                     "A4_reverse", "A6_index"]),
    ("L2_binding", ["B2_bind", "B4_bind", "B8_bind", "B_rename"]),
    ("L3_state", ["C_history_current", "C_reassign", "C_indirect"]),
    ("L4_generalization", ["D_distance", "D_interference"]),
]

STAGE_OF_TASK = {task: stage for stage, tasks in STAGES for task in tasks}
STAGE_ORDER = {stage: i for i, (stage, _) in enumerate(STAGES)}


def load_split(split: str) -> list[dict]:
    path = DATA / f"m4_{split}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def stage_of(row: dict) -> str:
    return STAGE_OF_TASK.get(row.get("task"), "L_unknown")


def build_curriculum(rows: list[dict], stage: str | None = None) -> list[dict]:
    """按预登记课程顺序返回行引用；不重采样，不复制凑数。"""
    selected = [r for r in rows if stage is None or stage_of(r) == stage]
    return sorted(selected, key=lambda r: (STAGE_ORDER.get(stage_of(r), 99),
                                           r.get("task", ""), r.get("prompt", "")))


def summarize(rows: list[dict]) -> dict:
    by_stage = Counter(stage_of(r) for r in rows)
    by_task = Counter(r.get("task", "UNKNOWN") for r in rows)
    return {"n_rows": len(rows),
            "by_stage": {s: by_stage.get(s, 0) for s, _ in STAGES},
            "by_task": dict(sorted(by_task.items())),
            "unknown_tasks": sorted(t for t in by_task if t not in STAGE_OF_TASK)}


def main() -> None:
    out = {"status": "DEFINITION_ONLY", "gradient_steps": 0,
           "stages": {stage: tasks for stage, tasks in STAGES},
           "splits": {split: summarize(build_curriculum(load_split(split)))
                      for split in ("train", "val", "sealed")}}
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
