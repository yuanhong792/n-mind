"""Run auditable N-Cell architecture ablations without teacher, tools or memory.

Example:
  python experiments/v37_ncell_architecture/run_ablation.py --steps 1 --arms legacy router_ste
  python experiments/v37_ncell_architecture/run_ablation.py --steps 16 --arms legacy router_ste attention router_attention reset all
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import torch

from nmind.evaluation.ncell_ablation import ARMS, run_experiment, write_report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20261123)
    parser.add_argument("--steps", type=int, default=8,
                        help="Candidate optimizer steps per arm, capped at 128")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-samples", type=int, default=8)
    parser.add_argument("--arms", nargs="+", choices=list(ARMS),
                        default=list(ARMS))
    parser.add_argument("--final-sealed", action="store_true",
                        help="One-time final sealed evaluation; never used for development tuning")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--max-seconds-per-arm", type=float, default=240.0)
    parser.add_argument("--output-dir",
                        default="experiments/v37_ncell_architecture/out")
    args = parser.parse_args(argv)
    if args.steps < 0 or args.steps > 128:
        parser.error("--steps must be between 0 and 128")
    if args.threads < 1:
        parser.error("--threads must be >= 1")
    torch.set_num_threads(args.threads)
    result = run_experiment(
        seed=args.seed, steps=args.steps, batch_size=args.batch_size,
        eval_samples=args.eval_samples, arms=tuple(args.arms),
        final_sealed=args.final_sealed,
        wall_seconds_per_arm=args.max_seconds_per_arm)
    result["environment"] = {"python": sys.version.split()[0],
                             "torch": torch.__version__,
                             "platform": platform.platform(),
                             "threads": torch.get_num_threads(),
                             "device": "cpu"}
    paths = write_report(result, Path(args.output_dir))
    compact = {
        "status": "EXPLORATORY_RESULTS_ONLY",
        "report": [str(p) for p in paths],
        "arms": {a: {"params": x["parameters"],
                      "optimizer_steps_executed": x["training"]["optimizer_steps_executed"],
                      "dev_full_answer_exact_count": x["after_dev"]["full_answer_exact_count"],
                      "dev_n": x["after_dev"]["evaluated"],
                      "dev_paired_both_correct": x["after_dev"]["pair_both_correct"],
                      "dev_pairs": x["after_dev"]["paired_contexts"],
                      "loss_last": (round(x["training"]["curve"][-1]["loss_answer_eos"], 5)
                                    if x["training"]["curve"] else None),
                      "grad_router_last": (x["training"]["curve"][-1]["grad"]["router"]
                                           if x["training"]["curve"] else None),
                      "parameter_delta": x["parameter_delta"]}
                 for a, x in result["arms"].items()}}
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
