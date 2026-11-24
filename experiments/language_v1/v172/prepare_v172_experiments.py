"""按 V18_TRAINING_PLAN.json 生成各实验的可执行 argv 与配置快照。

只生成计划，不执行任何训练（main_model_optimizer_steps = 0）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from nmind.language.neural_dialogue import LMConfig, NMindLM
from v172_protocol import write_json

OUT = Path(__file__).resolve().parent
PLAN = json.loads((OUT / "V18_TRAINING_PLAN.json").read_text(encoding="utf-8"))


def count_params(cfg_dict: dict) -> int:
    return NMindLM(LMConfig(**cfg_dict)).n_params()


def main():
    prepared = []
    shared = PLAN["shared_protocol"]
    for exp in PLAN["experiments"]:
        cfg = dict(exp.get("config") or exp.get("fixed_choice") or {})
        if not cfg:
            cfg = dict(LMConfig(vocab_size=4096, d_model=32, n_blocks=2,
                                block_hidden=32, workspace_dim=16,
                                workspace_slots=2, top_k=2).to_dict())
        cfg.setdefault("vocab_size", 4096)
        cfg.setdefault("query_readout", False)
        n = count_params(cfg)
        prepared.append({
            "id": exp["id"],
            "name": exp["name"],
            "single_variable": exp["single_variable"],
            "config": cfg,
            "parameter_count_actual": n,
            "checkpoint_compatible_with_parent": exp["checkpoint_compatible_with_parent"],
            "seed": shared["seed"],
            "max_steps": shared["max_steps"],
            "max_wall_seconds": shared["max_wall_seconds"],
            "data": shared["data"],
            "argv": [
                str(Path("experiments/language_v1/v172/train_v172.py")),
                "--execute-train", "--authorized",
                "--experiment", exp["id"],
                "--seed", str(shared["seed"]),
                "--max-steps", str(shared["max_steps"]),
                "--max-wall", str(shared["max_wall_seconds"]),
            ],
            "executed": False,
        })
    write_json(OUT / "V18_EXPERIMENT_ARGV.json", {
        "status": "PREPARED_NOT_EXECUTED",
        "main_model_optimizer_steps": 0,
        "experiments": prepared,
        "note": "参数量按实际配置计算；B/C 的 checkpoint 与父模型不兼容，必须按预登记范围重新初始化。",
    })
    print(json.dumps([{"id": p["id"], "params": p["parameter_count_actual"]} for p in prepared],
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
