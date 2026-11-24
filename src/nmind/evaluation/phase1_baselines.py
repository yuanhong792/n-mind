"""Shortcut baselines and counterfactual binding probes."""
from __future__ import annotations
import random
from collections import Counter


def shortcut_predictions(tasks, *, seed=1234):
    rng = random.Random(seed)
    answers = [t.target for t in tasks]
    majority = Counter(answers).most_common(1)[0][0] if answers else ""
    out = {"majority": [majority] * len(tasks), "random": [],
           "fixed_position": []}
    for t in tasks:
        out["random"].append(rng.choice(sorted(set(answers))) if answers else "")
        slots = dict(t.slots)
        pos = slots.get("position", slots.get("position_b", "1号位置"))
        ent = slots.get("query_entity", slots.get("entity", slots.get("entity_b", "小红")))
        out["fixed_position"].append(f"{ent}在{pos}。")
    return out


def recent_entity_predictions(tasks):
    out = []
    for t in tasks:
        s = dict(t.slots)
        ent = s.get("entity_b", s.get("entity", s.get("query_entity", "")))
        pos = s.get("position_b", s.get("position", "1号位置"))
        out.append(f"{ent}在{pos}。")
    return out


def accuracy(predictions, tasks):
    return sum(p == t.target for p, t in zip(predictions, tasks)) / max(1, len(tasks))


def counterfactual_follow_rate(pairs, predictions_by_fingerprint):
    if not pairs:
        return {"status": "INSUFFICIENT_N", "n": 0, "follow_rate": None}
    hits = 0
    for pair in pairs:
        ok = True
        for item in pair["tasks"]:
            pred = predictions_by_fingerprint.get(item["fingerprint"], "")
            ok &= pred == item["target"]
        hits += int(ok)
    return {"status": "OK", "n": len(pairs), "follow_rate": hits / len(pairs)}
