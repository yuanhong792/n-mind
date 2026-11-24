"""课程学习编排：符号 → 复制 → 位置 → 变量绑定 → 更新 → 运算 → 逻辑 → 程序。

提供每阶段的任务配额与训练/验证/封存装载；训练后期回放已学任务检查遗忘。
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"

STAGES = [
    {"id": "S0", "name": "symbols", "tasks": ["C0_copy1", "C0_copyN"], "max_steps": 96},
    {"id": "S1", "name": "copy_index", "tasks": ["C0_copyN", "C0_reverse", "C0_index", "C0_next"], "max_steps": 160},
    {"id": "S2", "name": "bind1", "tasks": ["C2_bind1", "C2_indirect"], "max_steps": 128},
    {"id": "S3", "name": "bindN", "tasks": ["C2_bindN", "C2_counter", "C2_reassign"], "max_steps": 384},
    {"id": "S4", "name": "arith", "tasks": ["C3_arith", "C3_var_arith"], "max_steps": 512},
    {"id": "S5", "name": "logic_program", "tasks": ["C4_logic", "C5_program"], "max_steps": 0},
    {"id": "S6", "name": "generate_fix", "tasks": ["C6_generate", "C7_fix"], "max_steps": 0},
    {"id": "S7", "name": "autonomous", "tasks": ["C8_feedback", "C9_autoprogram"], "max_steps": 0},
]


def load(split: str):
    import json
    p = DATA / f"c1_{split}.jsonl"
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def stage_report() -> dict:
    rows = load("train")
    counts = Counter(r["task"] for r in rows)
    out = []
    for s in STAGES:
        out.append({"id": s["id"], "name": s["name"], "tasks": s["tasks"],
                    "available_samples": sum(counts.get(t, 0) for t in s["tasks"]),
                    "allocated_steps": s["max_steps"],
                    "status": "READY" if s["max_steps"] > 0 and sum(counts.get(t, 0) for t in s["tasks"]) > 0
                              else "INTERFACE_ONLY"})
    return {"stages": out, "total_train": len(rows),
            "principle": "不跳过基础能力；后期回放已学任务检查灾难性遗忘"}
