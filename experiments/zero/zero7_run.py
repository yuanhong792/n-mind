"""ZERO-7 experiment entry. Default is forward-only inspection; training needs new authorization."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import psutil
import torch

from .budget_ledger import Budget
from .zero7_symbolic import SymbolAgent, SymbolWorld

HERE = Path(__file__).parent
RESULTS = HERE / "results" / "zero7"
ARMS = (("inherited", True, False), ("inherited_replay", True, True),
        ("random_control", False, False))
SEEDS = (101, 202, 303)
UPDATES_PER_ARM = 32
PLAN = len(ARMS) * len(SEEDS) * UPDATES_PER_ARM


def plan() -> dict:
    return {"status": "REQUEST_ONLY_NOT_AUTHORIZED", "cpu_updates_max": PLAN,
            "formula": "3 seeds x 3 paired arms x (24 known-symbol + 8 novel-symbol feedbacks)",
            "arms": [a[0] for a in ARMS], "zero5_remaining_140": "NOT_USED"}


def run(budget: Budget) -> dict:
    if PLAN != 288 or budget.snapshot()["reserved"] != 0:
        raise RuntimeError("ZERO7_BUDGET_PLAN_MISMATCH")
    rows = []
    for seed in SEEDS:
        world = SymbolWorld(710000 + seed)
        order = np.random.default_rng(seed + 9000).permutation(192)[:32]
        frozen = SymbolAgent(seed, True)
        for name, inherited, replay in ARMS:
            wall_start, cpu_start = time.perf_counter(), time.process_time()
            agent = SymbolAgent(seed, inherited, replay)
            old_initial = agent.old_skill_accuracy(seed)
            curves, steps = [], []
            for k in range(33):
                if k in (0, 8, 16, 24, 25, 28, 32):
                    curves.append({"updates": k,
                                   "known_dev": agent.evaluate(world, "dev", tuple(range(9))),
                                   "novel_dev": agent.evaluate(world, "dev", (9, 10, 11)),
                                   "reverse_dev": agent.evaluate_reverse(world, "dev"),
                                   "old_skill_balanced_accuracy": agent.old_skill_accuracy(seed)})
                if k == 32:
                    break
                token = k % 9 if k < 24 else 9 + (k - 24) % 3
                sample = world.feedback(token, int(order[k]))
                update = agent.feedback_update(sample, budget)
                rows_step = {"step": k + 1, "token": token, "object_index": int(order[k]), **update}
                steps.append(rows_step)
            final = {"known": agent.evaluate(world, "sealed", tuple(range(9))),
                     "novel": agent.evaluate(world, "sealed", (9, 10, 11)),
                     "reverse": agent.evaluate_reverse(world, "sealed")}
            RESULTS.mkdir(parents=True, exist_ok=True)
            checkpoint = RESULTS / f"{name}_seed_{seed}.pt"
            fd, tmp = tempfile.mkstemp(dir=RESULTS, suffix=".tmp")
            os.close(fd)
            try:
                torch.save({"core": agent.core.predictor.state_dict(),
                            "explorer": agent.core.explorer.state_dict(),
                            "symbol_head": agent.head.state_dict(),
                            "optimizer": agent.optimizer.state_dict(),
                            "seed": seed, "arm": name, "budget": budget.snapshot()}, tmp)
                os.replace(tmp, checkpoint)
            finally:
                if os.path.exists(tmp): os.unlink(tmp)
            rows.append({"seed": seed, "arm": name, "inherited_checkpoint_sha256": agent.source_checkpoint,
                         "old_initial_balanced_accuracy": old_initial,
                         "old_final_balanced_accuracy": agent.old_skill_accuracy(seed),
                         "old_final_sealed_balanced_accuracy": agent.old_skill_accuracy(seed, sealed=True),
                         "curve": curves, "steps": steps, "final_sealed": final,
                         "feedback_samples": len(steps),
                         "replayed_examples": sum(step["replayed_examples"] for step in steps),
                         "cpu_seconds": time.process_time()-cpu_start,
                         "wall_seconds": time.perf_counter()-wall_start,
                         "peak_working_set_bytes": getattr(psutil.Process().memory_info(), "peak_wset", psutil.Process().memory_info().rss),
                         "replay_storage_bytes": sum(row["x"].numel()*row["x"].element_size()+row["y"].numel()*row["y"].element_size() for row in agent.core.replay_buffer),
                         "checkpoint": str(checkpoint), "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()})
        rows.append({"seed": seed, "arm": "frozen_inherited_control", "updates": 0,
                     "final_sealed": {"known": frozen.evaluate(world, "sealed", tuple(range(9))),
                                      "novel": frozen.evaluate(world, "sealed", (9,10,11)),
                                      "reverse": frozen.evaluate_reverse(world, "sealed")},
                     "old_skill_balanced_accuracy": frozen.old_skill_accuracy(seed, sealed=True)})
        _, _, forward_labels = world.query("sealed", tuple(range(12)))
        _, _, reverse_labels = world.reverse_query("sealed")
        generator = torch.Generator().manual_seed(seed + 9900)
        rows.append({"seed": seed, "arm": "random_prediction_control", "updates": 0,
                     "forward_accuracy": float((torch.randint(2, forward_labels.shape, generator=generator) == forward_labels).float().mean()),
                     "reverse_full_symbol_accuracy": float((torch.randint(9, reverse_labels.shape, generator=generator) == reverse_labels).float().mean())})
    return {"status": "COMPLETE", "actual_updates": budget.snapshot()["completed"], "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--authorization-id")
    args = parser.parse_args()
    if not args.run:
        print(json.dumps(plan(), indent=2))
        return
    if not args.authorization_id or not args.authorization_id.startswith("zero7-"):
        raise SystemExit("ZERO7_NEW_AUTHORIZATION_REQUIRED")
    authorization = HERE / "ZERO7_AUTHORIZATION.json"
    if not authorization.exists():
        raise SystemExit("ZERO7_TRAINING_NOT_AUTHORIZED")
    permit = json.loads(authorization.read_text(encoding="utf-8"))
    if (permit.get("authorization_id") != args.authorization_id or
            permit.get("approved_updates_max") != PLAN or permit.get("approved") is not True):
        raise SystemExit("ZERO7_AUTHORIZATION_MISMATCH")
    prereg = json.loads((HERE / "ZERO7_PREREGISTRATION.json").read_text(encoding="utf-8"))
    for name, digest in prereg["source_sha256"].items():
        if hashlib.sha256((HERE / name).read_bytes()).hexdigest() != digest:
            raise SystemExit("ZERO7_SOURCE_HASH_MISMATCH")
    budget = Budget(args.authorization_id, PLAN, RESULTS / "ZERO7_BUDGET_STATE.json")
    result = run(budget)
    (RESULTS / "ZERO7_RESULTS.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"status": result["status"], "updates": result["actual_updates"]}))


if __name__ == "__main__":
    main()
