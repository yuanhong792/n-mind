"""ZERO-6 paired screening entry point; default path is status-only."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import hashlib
from pathlib import Path

from .ZERO5_AGENT import ROOT
from .budget_ledger import Budget
from .zero6_continual import minimum_budget_plan, run_paired_screening


HERE = Path(__file__).parent
RESULTS = ROOT / "results" / "zero6"


def verify_frozen_inputs() -> str:
    prereg = json.loads((HERE / "ZERO6_PREREGISTRATION.json").read_text(encoding="utf-8"))
    if prereg.get("status") != "FROZEN_REQUEST_ONLY_NOT_AUTHORIZED":
        raise RuntimeError("ZERO6_PREREGISTRATION_NOT_FROZEN")
    for name, expected in prereg["source_hashes"].items():
        source_path = HERE.parents[1] / "AGENTS.md" if name == "AGENTS.md" else HERE / name
        if hashlib.sha256(source_path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"ZERO6_SOURCE_HASH_MISMATCH:{name}")
    actual_manifest = hashlib.sha256((HERE / "ZERO6_TASK_MANIFEST.json").read_bytes()).hexdigest()
    if actual_manifest != prereg["task_manifest_sha256"]:
        raise RuntimeError("ZERO6_TASK_MANIFEST_HASH_MISMATCH")
    return actual_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="run only with explicit new ZERO-6 authorization")
    parser.add_argument("--authorized", action="store_true", help="explicitly confirm a granted ZERO-6 authorization")
    parser.add_argument("--authorization-id")
    parser.add_argument("--budget-limit", type=int)
    args = parser.parse_args()
    if not args.run:
        zero5 = json.loads((ROOT / "results" / "ZERO5_BUDGET_STATE.json").read_text(encoding="utf-8"))
        print(json.dumps({"status": "NO_TRAINING", "request": minimum_budget_plan(),
                          "zero5_budget_state_read_only": zero5}, indent=2))
        return
    if not args.authorized or not args.authorization_id or args.authorization_id.startswith("zero5-"):
        raise SystemExit("ZERO6_REQUIRES_EXPLICIT_NEW_AUTHORIZATION_ID")
    if args.budget_limit != 360:
        raise SystemExit("ZERO6_BUDGET_LIMIT_MUST_MATCH_FROZEN_REQUEST_360")
    protocol_sha = verify_frozen_inputs()
    state_path = RESULTS / "ZERO6_BUDGET_STATE.json"
    budget = Budget(args.authorization_id, 360, state_path)
    policies = {seed: HERE / "checkpoints" / f"zero4_G2_{seed}.pt"
                for seed in (101, 202, 303)}
    result = run_paired_screening(
        budget, policies, authorization_confirmed=True,
        checkpoint_root=RESULTS / "checkpoints",
        protocol_sha256=protocol_sha)
    RESULTS.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix="zero6-results", suffix=".tmp", dir=RESULTS)
    os.close(fd)
    try:
        Path(temp_path).write_text(json.dumps({"result": result, "budget": budget.snapshot()}, indent=2),
                                   encoding="utf-8")
        os.replace(temp_path, RESULTS / "ZERO6_RESULTS.json")
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    print(json.dumps({"status": result["status"], "actual_updates": result["actual_completed_updates"],
                      "budget": budget.snapshot()}, indent=2))


if __name__ == "__main__":
    main()
