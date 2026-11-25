"""Read-only ZERO-7 admission audit; no optimizer step or feedback use."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from .zero7_run import HERE, PLAN, SEEDS, plan
from .zero7_symbolic import SymbolAgent, SymbolWorld, ZERO6
from .zero6_continual import feature_sha


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    zero6_before = json.loads((ZERO6 / "ZERO6_BUDGET_STATE.json").read_text(encoding="utf-8"))
    zero5_path = HERE / "results" / "ZERO5_BUDGET_STATE.json"
    zero5_before = json.loads(zero5_path.read_text(encoding="utf-8"))
    audits = []
    all_sets = []
    for seed in SEEDS:
        world = SymbolWorld(710000 + seed)
        inherited = SymbolAgent(seed, True)
        fresh = SymbolAgent(seed, False)
        inherited_before = [p.detach().clone() for p in inherited.core.predictor.parameters()]
        head_same = all(torch.equal(a, b) for a, b in zip(inherited.head.parameters(), fresh.head.parameters()))
        known = inherited.evaluate(world, "dev", tuple(range(9)))
        novel = inherited.evaluate(world, "dev", (9, 10, 11))
        reverse = inherited.evaluate_reverse(world, "dev")
        sample = world.feedback(0, 0)
        x, token, label, relation, canonical = sample
        diagnostic_loss = (F.cross_entropy(inherited.logits(x[None], torch.tensor([token])), label[None]) +
                           F.cross_entropy(inherited.head.name(x[None], torch.tensor([relation]), inherited.core.predictor.net[0]),
                                           torch.tensor([canonical])))
        params = list(inherited.core.predictor.parameters()) + list(inherited.head.parameters())
        gradients = torch.autograd.grad(diagnostic_loss, params, allow_unused=True)
        active_gradient_tensors = sum(g is not None and torch.isfinite(g).all().item() and g.abs().sum().item() > 0
                                      for g in gradients)
        checkpoint = ZERO6 / "checkpoints" / f"seed_{seed}" / "experience_replay" / "after_world_4.pt"
        values = world.audit()
        values.update({"model_seed": seed, "inherited_checkpoint_sha256": sha(checkpoint),
                       "vocabulary_unique": len(set(world.vocabulary)) == 12,
                       "checkpoint_match": inherited.source_checkpoint == sha(checkpoint),
                       "paired_new_head_initialization": head_same,
                       "old_skill_dev_balanced_accuracy": inherited.old_skill_accuracy(seed),
                       "known_dev_forward": known, "novel_dev_forward": novel,
                       "reverse_dev_forward": reverse,
                       "neural_gradient_tensors_nonzero": active_gradient_tensors,
                       "symbol_head_parameters": sum(p.numel() for p in inherited.head.parameters()),
                       "predictor_weights_unchanged": all(torch.equal(a,b) for a,b in zip(inherited_before,inherited.core.predictor.parameters()))})
        audits.append(values)
        all_sets.append({name:{feature_sha(x) for x in world._sets[name][0]} for name in world._sets})
    cross = {f"{a}:{b}":sum(len(all_sets[a][left] & all_sets[b][right]) for left in all_sets[a] for right in all_sets[b])
             for a in range(len(SEEDS)) for b in range(a+1,len(SEEDS))}
    zero6_after = json.loads((ZERO6 / "ZERO6_BUDGET_STATE.json").read_text(encoding="utf-8"))
    zero5_after = json.loads(zero5_path.read_text(encoding="utf-8"))
    passed = (PLAN == 288 and zero6_before == zero6_after and zero5_before == zero5_after and
              all(all(v == 0 for v in a["intersections"].values()) and a["checkpoint_match"] and
                  a["paired_new_head_initialization"] and a["predictor_weights_unchanged"] and
                  a["vocabulary_unique"] for a in audits) and
              all(a["neural_gradient_tensors_nonzero"] >= 4 for a in audits) and
              len({a["hidden_rule_sha256"] for a in audits}) == len(audits) and all(x == 0 for x in cross.values()))
    result = {"status":"PASS_NO_TRAINING" if passed else "FAIL", "training_updates":0,
              "audits":audits,"cross_world_feature_hash_intersections":cross,
              "zero6_budget_unchanged":zero6_before==zero6_after,
              "zero5_budget_unchanged":zero5_before==zero5_after,"budget_request":plan()}
    (HERE/"ZERO7_NO_TRAIN_PREFLIGHT.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    if not passed:
        raise SystemExit("ZERO7_PREFLIGHT_FAILED")
    sources={name:sha(HERE/name) for name in ("zero7_symbolic.py","zero7_run.py","zero7_preflight.py")}
    prereg={"status":"FROZEN_REQUEST_ONLY_NOT_AUTHORIZED","zero6_source_commit":"c0d537e07455aa8ccb2f579d84deb22269243bff",
            "source_sha256":sources,"seeds":list(SEEDS),"symbol_world_seed_formula":"710000 + model_seed",
            "arms":["inherited","inherited_replay","random_control"],
            "zero_update_controls":["frozen_inherited_control","random_prediction_control"],
            "support_protocol":"24 feedbacks for tokens 0..8; then 8 feedbacks for novel tokens 9..11; same paired object order",
            "sealed_protocol":"Only final endpoint; not used for parameter selection",
            "primary_metrics":["novel-symbol sealed accuracy","unseen-object combination accuracy","old ZERO-6 skill retention"],
            "budget_max":PLAN,"training_authorized":False}
    (HERE/"ZERO7_PREREGISTRATION.json").write_text(json.dumps(prereg,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":result["status"],"worlds":len(audits),"updates":0,"budget_request":PLAN}))


if __name__ == "__main__":
    main()
