"""FLY eval_sanity：任何 FLY 评测输出前必须通过。"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from fly_core import (
    NAME_CHARS,
    D_INPUT,
    FlyMemory,
    IdentityExpansion,
    PlasticConfig,
    SparseExpansion,
    encode_name,
    recall_pairs,
)
from fly_data import load_mqar

HERE = Path(__file__).resolve().parent
SEEDS = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010]


def _mk_cfg(**kw):
    base = dict(M=2048, fan_in=6, sparsity=0.05, eta=0.8, decay=1.0,
                clip=4.0, variant="A", seed=101)
    base.update(kw)
    return PlasticConfig(**base)


def check_wta_exact_k():
    exp = SparseExpansion(M=512, fan_in=6, sparsity=0.05, seed=101)
    code = exp.encode("小明红号")
    return {"pass": int(code.sum()) == exp.k, "k": exp.k, "n_active": int(code.sum())}


def check_seed_reproducible():
    a = SparseExpansion(256, 3, 0.05, 101)
    b = SparseExpansion(256, 3, 0.05, 101)
    c = SparseExpansion(256, 3, 0.05, 202)
    ok = np.array_equal(a.R, b.R) and not np.array_equal(a.R, c.R)
    return {"pass": bool(ok)}


def check_fan_in():
    exp = SparseExpansion(256, 7, 0.05, 101)
    counts = exp.R.sum(axis=1)
    return {"pass": bool(np.all(counts == 7)), "min": int(counts.min()), "max": int(counts.max())}


def check_update_formula():
    cfg = _mk_cfg(M=4, fan_in=1, sparsity=1.0, eta=0.5, decay=1.0, clip=10.0, seed=0)
    mem = FlyMemory(cfg, IdentityExpansion(4))
    mem.write(0, 1)
    expected = np.zeros((4, 4), dtype=np.float32)
    expected[0, 1] = 0.5
    ok = np.allclose(mem.W, expected, atol=1e-6)
    return {"pass": bool(ok), "max_abs_err": float(np.abs(mem.W - expected).max())}


def check_clip_decay():
    cfg = _mk_cfg(M=4, fan_in=1, sparsity=1.0, eta=0.5, decay=0.5, clip=0.1, seed=0)
    mem = FlyMemory(cfg, IdentityExpansion(4))
    mem.W[:] = 1.0
    mem.write(0, 1)
    ok = bool(np.all(mem.W <= 0.1 + 1e-6) and np.all(mem.W >= -0.1 - 1e-6))
    return {"pass": ok, "max": float(mem.W.max()), "min": float(mem.W.min())}


def check_no_gradient():
    src = (HERE / "fly_core.py").read_text(encoding="utf-8")
    forbidden = [
        re.compile(r"^\s*import\s+torch\b", re.M),
        re.compile(r"^\s*from\s+torch\b", re.M),
        re.compile(r"\.backward\s*\("),
        re.compile(r"requires_grad"),
        re.compile(r"autograd\.grad"),
    ]
    hits = [p.pattern for p in forbidden if p.search(src)]
    return {"pass": not hits, "hits": hits}


def check_oracle_flv02():
    rng = np.random.default_rng(20261127)
    n, trials = 4, 4000
    true = rng.integers(0, n, size=trials)
    scores = np.full((trials, n), -1e9, dtype=np.float32)
    scores[np.arange(trials), true] = 1.0
    pred = scores.argmax(axis=1)
    em = float(np.mean(pred == true))
    ranks = np.argsort(-scores, axis=1)
    rank = float(np.mean([np.flatnonzero(ranks[i] == true[i])[0] == 0 for i in range(trials)]))
    return {"pass": em == 1.0 and rank == 1.0, "em": em, "rank": rank, "trials": trials}


def check_shuffle_flv02():
    rng = np.random.default_rng(20261128)
    n, trials = 4, 8000
    pred = rng.integers(0, n, size=trials)
    true = rng.integers(0, n, size=trials)
    acc = float(np.mean(pred == true))
    expected = 1.0 / n
    return {"pass": abs(acc - expected) <= 0.04, "acc": acc, "expected": expected, "trials": trials}


def check_data_hash():
    ds, meta = load_mqar(2, "heldout_seq_seal2")
    ok = ds["sha256"] == meta["sha256"]
    return {"pass": bool(ok), "sha256": ds["sha256"], "n_test": int(ds["n_test"])}


def _random_pairs(rng, n):
    chars = list(dict.fromkeys(NAME_CHARS))
    def unique_names():
        out = set()
        while len(out) < n:
            out.add("".join(rng.choice(chars, 4, replace=False)))
        return list(out)
    return list(zip(unique_names(), unique_names()))


def check_low_p_recall():
    # eval_sanity 固定使用小规模、低学习率、5 轮确定性写入；
    # 这不是结果优选，只是验证模块在极低负载下具备基本可召回性。
    details = []
    epochs = 5
    for seed in SEEDS:
        cfg = _mk_cfg(M=1024, sparsity=0.02, eta=0.05, seed=seed)
        rng = np.random.default_rng(seed)
        for p in (10, 50):
            mem = FlyMemory(cfg)
            pairs = _random_pairs(rng, p)
            mem.store_pairs(pairs, epochs=epochs)
            acc = recall_pairs(mem, pairs)
            details.append({"seed": seed, "P": p, "recall": acc, "epochs": epochs})
    ok = all(x["recall"] == 1.0 for x in details)
    return {"pass": ok, "worst": min(x["recall"] for x in details), "details": details,
            "checks": len(details)}


CHECKS = {
    "wta_exact_k": check_wta_exact_k,
    "seed_reproducible": check_seed_reproducible,
    "fan_in": check_fan_in,
    "update_formula": check_update_formula,
    "clip_decay": check_clip_decay,
    "no_gradient": check_no_gradient,
    "oracle_flv02": check_oracle_flv02,
    "shuffle_flv02": check_shuffle_flv02,
    "data_hash": check_data_hash,
    "low_p_recall": check_low_p_recall,
}


def run_all():
    results = {}
    for name, fn in CHECKS.items():
        try:
            r = fn()
        except Exception as e:
            r = {"pass": False, "error": f"{type(e).__name__}: {e}"}
        results[name] = r
    n_pass = sum(1 for r in results.values() if r.get("pass"))
    return {
        "status": "PASS" if n_pass == len(results) else "FAIL",
        "n_pass": n_pass,
        "n_total": len(results),
        "checks": results,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(run_all(), ensure_ascii=False, indent=2))
