"""Forward-only ZERO-6 self-check; performs no optimizer updates."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import torch

from .ZERO5_AGENT import ROOT
from .zero6_continual import ContinuousAgent, SkillWorld, feature_sha, minimum_budget_plan


HERE = Path(__file__).parent


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    zero5_path = ROOT / "results" / "ZERO5_BUDGET_STATE.json"
    zero5_before = json.loads(zero5_path.read_text(encoding="utf-8")) if zero5_path.exists() else None
    agent = ContinuousAgent(101)
    params = list(agent.predictor.parameters()) + list(agent.explorer.parameters())
    params_before = [p.detach().clone() for p in params]
    manifest = json.loads((HERE / "ZERO6_TASK_MANIFEST.json").read_text(encoding="utf-8"))
    contexts = (-1.0, -0.5, 0.0, 0.5, 1.0)
    world_specs = [(world_seed, context)
                   for seed in (101, 202, 303)
                   for world_seed, context in zip(manifest["world_seeds_by_model_seed"][str(seed)], contexts)]
    worlds = [SkillWorld(seed, context) for seed, context in world_specs]
    audits = [world.audit_hashes() for world in worlds]
    all_hashes = [{name: {feature_sha(row) for row in world._sets[name][0]}
                   for name in world.SETS}
                  for world in worlds]
    cross_world_intersections = {}
    for left in range(len(worlds)):
        for right in range(left + 1, len(worlds)):
            cross_world_intersections[f"{worlds[left].world_seed}:{worlds[right].world_seed}"] = {
                f"{a}:{b}": len(all_hashes[left][a] & all_hashes[right][b])
                for a in worlds[left].SETS for b in worlds[right].SETS}
    cross_world_disjoint = all(count == 0 for pair in cross_world_intersections.values()
                               for count in pair.values())
    rule_signatures = {hashlib.sha256(torch.cat((world._w, world._bias.reshape(1))).numpy().tobytes()).hexdigest()
                       for world in worlds}
    hidden_independent = len(rule_signatures) == len(worlds)
    predictions = [agent.predictor(world.agent.candidates()).argmax(-1)[:8].tolist() for world in worlds]
    label_counts = {str(world.world_seed): {
        name: {str(label): int((world._sets[name][1] == label).sum()) for label in (0, 1)}
        for name in world.SETS} for world in worlds}
    both_classes_present = all(counts["0"] > 0 and counts["1"] > 0
                               for world_counts in label_counts.values()
                               for counts in world_counts.values())
    zero5_after = json.loads(zero5_path.read_text(encoding="utf-8")) if zero5_path.exists() else None
    gates = manifest["frozen_selective_gates"]
    gate_hashes_valid = all(sha256(HERE / "checkpoints" / value["file"]) == value["sha256"]
                            for value in gates.values())
    unchanged = all(torch.equal(before, after) for before, after in
                    zip(params_before, params))
    result = {
        "status": "PASS_NO_TRAINING_SELF_CHECK",
        "training_updates": 0,
        "parameter_count": agent.parameter_count,
        "forward_only_predictions_recorded": len(predictions),
        "world_feature_hash_audits": audits,
        "all_world_splits_disjoint": all(a["all_disjoint"] for a in audits),
        "cross_world_feature_hash_intersections": cross_world_intersections,
        "all_worlds_feature_disjoint": cross_world_disjoint,
        "hidden_rules_independent": hidden_independent,
        "label_counts_by_world_and_split": label_counts,
        "both_classes_present_in_all_splits": both_classes_present,
        "historical_zero4_selective_gate_hashes_valid": gate_hashes_valid,
        "parameter_tensors_unchanged": unchanged,
        "zero5_budget_state_before": zero5_before,
        "zero5_budget_state_after": zero5_after,
        "zero5_budget_unchanged": zero5_before == zero5_after,
        "source_hashes": {name: sha256(path) for name, path in {
            "zero6_continual.py": HERE / "zero6_continual.py",
            "zero6_preflight.py": HERE / "zero6_preflight.py",
            "zero6_run.py": HERE / "zero6_run.py",
            "budget_ledger.py": HERE / "budget_ledger.py",
            "ZERO5_AGENT.py": HERE / "ZERO5_AGENT.py",
            "ZERO5_EXPLORATION.py": HERE / "ZERO5_EXPLORATION.py",
            "zero4.py": HERE / "zero4.py",
            "AGENTS.md": HERE.parents[1] / "AGENTS.md",
        }.items()},
        "manifest_sha256": sha256(HERE / "ZERO6_TASK_MANIFEST.json"),
        "minimum_budget_request": minimum_budget_plan(),
    }
    if not (result["all_world_splits_disjoint"] and cross_world_disjoint and hidden_independent and both_classes_present and unchanged and gate_hashes_valid
            and result["zero5_budget_unchanged"] and agent.parameter_count == 403):
        result["status"] = "FAIL"
        raise SystemExit(json.dumps(result, indent=2))
    (HERE / "ZERO6_NO_TRAIN_PREFLIGHT.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True,
                            text=True).stdout.strip()
    preregistration = {
        "status": "FROZEN_REQUEST_ONLY_NOT_AUTHORIZED",
        "based_on_commit": commit,
        "task_manifest_sha256": result["manifest_sha256"],
        "source_hashes": result["source_hashes"],
        "new_budget_requested_max": 360,
        "zero5_remaining_140": "NOT_USED",
        "arms": ["ordinary_sgd", "experience_replay_fifo32", "frozen_zero4_G2_selective_update"],
        "seeds": [101, 202, 303],
        "worlds": ["A", "B", "C", "D", "E"],
        "primary_metric": "per-skill balanced-accuracy learning curves plus retention after each later world",
        "sealed_gate": "each paired seed ordinary-SGD mean skill balanced-accuracy development gain >= 0.05",
        "training_authorized": False,
    }
    (HERE / "ZERO6_PREREGISTRATION.json").write_text(json.dumps(preregistration, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
