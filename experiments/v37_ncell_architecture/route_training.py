"""Route N-Cell ablation to free GitHub CPU or prepare an opt-in local GPU job.

No external API, cloud spending, self-hosted runner, or remote control is used.
This planner writes a JSON contract and a Windows PowerShell command file;
it NEVER launches local training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

# These are *conservative project policy bounds*, not GitHub's official runner limits.
CLOUD_MAX_STEPS_PER_ARM = 128
CLOUD_MAX_TOTAL_STEPS = 768
CLOUD_MAX_ESTIMATED_RAM_GB = 10.0
LOCAL_MAX_STEPS_PER_ARM = 2048
ARMS = ("legacy", "router_ste", "attention", "router_attention", "reset", "all")


def decide_route(*, steps: int, arms: list[str], estimated_ram_gb: float,
                 requires_gpu: bool = False, force_local: bool = False) -> dict:
    if not arms or len(set(arms)) != len(arms) or any(a not in ARMS for a in arms):
        raise ValueError("arms must be nonempty, distinct registered candidates")
    if not 0 <= steps <= LOCAL_MAX_STEPS_PER_ARM:
        raise ValueError(f"steps must be in 0..{LOCAL_MAX_STEPS_PER_ARM}")
    if not math.isfinite(estimated_ram_gb) or estimated_ram_gb <= 0:
        raise ValueError("estimated RAM must be finite and positive")
    total = steps * len(arms)
    reasons = []
    if requires_gpu:
        reasons.append("GPU_REQUIRED")
    if force_local:
        reasons.append("USER_FORCED_LOCAL")
    if steps > CLOUD_MAX_STEPS_PER_ARM:
        reasons.append("CLOUD_PER_ARM_STEP_POLICY")
    if total > CLOUD_MAX_TOTAL_STEPS:
        reasons.append("CLOUD_TOTAL_STEP_POLICY")
    if estimated_ram_gb > CLOUD_MAX_ESTIMATED_RAM_GB:
        reasons.append("CLOUD_ESTIMATED_RAM_POLICY")
    if reasons:
        return {"route": "LOCAL_MANUAL", "reasons": reasons, "total_steps_max": total,
                "device": "cuda", "status": "READY_FOR_USER_LAUNCH",
                "automatic_local_execution": False}
    return {"route": "GITHUB_FREE_CPU", "reasons": ["WITHIN_PRESET_CPU_POLICY"],
            "total_steps_max": total, "device": "cpu", "status": "READY_FOR_GITHUB_ACTIONS",
            "automatic_local_execution": False}


def write_plan(directory: str | Path, *, steps: int, arms: list[str],
               estimated_ram_gb: float = 2.0, requires_gpu: bool = False,
               force_local: bool = False, batch_size: int = 4,
               eval_samples: int = 40, seed: int = 20261123) -> tuple[Path, Path]:
    if not 1 <= batch_size <= 120 or not 1 <= eval_samples <= 40:
        raise ValueError("invalid batch size or dev sample count")
    route = decide_route(steps=steps, arms=arms,
                         estimated_ram_gb=estimated_ram_gb,
                         requires_gpu=requires_gpu, force_local=force_local)
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    params = {"seed": seed, "steps": steps, "arms": arms, "batch_size": batch_size,
              "eval_samples": eval_samples, "estimated_ram_gb": estimated_ram_gb,
              "requires_gpu": requires_gpu, "force_local": force_local}
    body = {"schema": "nmind-training-route/1", "project_policy": {
        "cloud_max_steps_per_arm": CLOUD_MAX_STEPS_PER_ARM,
        "cloud_max_total_steps": CLOUD_MAX_TOTAL_STEPS,
        "cloud_max_estimated_ram_gb": CLOUD_MAX_ESTIMATED_RAM_GB,
        "local_max_steps_per_arm": LOCAL_MAX_STEPS_PER_ARM,
        "not_github_official_hardware_limits": True},
        "job": params, **route, "source_revision": "use-pinned-branch-checkout",
        "training_performed": False, "stable_checkpoint_modified": False,
        "sealed_evaluation_requested": False}
    body["plan_sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    j = out / "V37_TRAINING_ROUTE.json"
    ps = out / "V37_RUN_LOCAL.ps1"
    j.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")
    arm_str = " ".join(arms)
    command = ("python experiments/v37_ncell_architecture/run_ablation.py "
               f"--steps {steps} --batch-size {batch_size} --eval-samples {eval_samples} "
               f"--arms {arm_str} --seed {seed} --threads 4 "
               f"--device {route['device']} --max-steps-per-arm "
               f"{LOCAL_MAX_STEPS_PER_ARM if route['device'] == 'cuda' else CLOUD_MAX_STEPS_PER_ARM} "
               "--max-seconds-per-arm 3600 "
               "--save-candidates "
               "--output-dir experiments/v37_ncell_architecture/local_routed_run")
    if route["route"] == "LOCAL_MANUAL":
        lines = [
            "# Generated local training plan. Review before invoking this file.",
            "# This file never changes Stable and does not upload private data.",
            "$ErrorActionPreference = 'Stop'",
            "if (-not (Test-Path 'experiments/v37_ncell_architecture/run_ablation.py')) {",
            "    throw 'Run from the checked-out V3.7 project root'",
            "}",
            "python -c 'import torch; assert torch.cuda.is_available()'",
            "if ($LASTEXITCODE -ne 0) { throw 'CUDA preflight failed' }",
            command,
            "if ($LASTEXITCODE -ne 0) { throw 'N-Cell candidate experiment failed' }",
        ]
    else:
        lines = [
            "# GITHUB_FREE_CPU: this workload fits the preset policy.",
            "# Do not execute locally; run it via GitHub Actions.",
            "# The command below is informational and is not automatically executed.",
            "# " + command,
        ]
    ps.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return j, ps


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--arms", nargs="+", choices=ARMS, required=True)
    p.add_argument("--estimated-ram-gb", type=float, default=2.0)
    p.add_argument("--requires-gpu", action="store_true")
    p.add_argument("--force-local", action="store_true")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--eval-samples", type=int, default=40)
    p.add_argument("--seed", type=int, default=20261123)
    p.add_argument("--output-dir",
                   default="experiments/v37_ncell_architecture/route_out")
    args = p.parse_args()
    files = write_plan(args.output_dir, steps=args.steps, arms=args.arms,
                       estimated_ram_gb=args.estimated_ram_gb,
                       requires_gpu=args.requires_gpu, force_local=args.force_local,
                       batch_size=args.batch_size, eval_samples=args.eval_samples,
                       seed=args.seed)
    print(files[0].read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
