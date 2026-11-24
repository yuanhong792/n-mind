"""Shortcut baselines and counterfactual binding probes."""
from __future__ import annotations
import random
import math
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

def wilson_interval(successes: int, n: int, z: float = 1.959963984540054):
    if n <= 0:
        return (float("nan"), float("nan"))
    p = successes / n; den = 1 + z*z/n
    c = (p + z*z/(2*n))/den
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/den
    return (c-h, c+h)

def combinatorial_probe_pairs(n_pairs: int = 200, seed: int = 20262402):
    """Deterministic probe generator with entity x position combinations."""
    rng = random.Random(seed)
    entities = [f"实体{i:02d}" for i in range(10)]
    positions = [f"位置{i:02d}" for i in range(10)]
    rows = []
    while len(rows) < n_pairs:
        a, b = rng.sample(entities, 2); pa, pb = rng.sample(positions, 2)
        q = a if len(rows) % 2 == 0 else b
        rows.append({"id": len(rows), "a": a, "b": b, "pa": pa, "pb": pb,
                     "query": q, "answer_a": pa if q == a else pb,
                     "answer_b": pb if q == a else pa})
    return rows
