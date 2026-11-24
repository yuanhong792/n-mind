"""Deterministic, auditable summary statistics for Phase 1."""
from __future__ import annotations
import numpy as np


def summarize(values, *, seed: int = 1234, n_boot: int = 10000,
              min_n: int = 3) -> dict:
    arr = np.asarray(list(values), dtype=float)
    if arr.size < min_n:
        return {"status": "INSUFFICIENT_N", "n": int(arr.size), "min_n": min_n}
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for i in range(n_boot):
        means[i] = rng.choice(arr, size=arr.size, replace=True).mean()
    return {"status": "OK", "n": int(arr.size), "mean": float(arr.mean()),
            "std_ddof1": float(arr.std(ddof=1)),
            "bootstrap_ci95": [float(np.percentile(means, 2.5)),
                                float(np.percentile(means, 97.5))],
            "bootstrap_seed": seed, "bootstrap_resamples": n_boot}


def paired_bootstrap(a, b, *, seed=1234, n_boot=10000, min_n=3) -> dict:
    x, y = np.asarray(a, float), np.asarray(b, float)
    if x.shape != y.shape:
        raise ValueError("配对序列长度必须一致")
    if x.size < min_n:
        return {"status": "INSUFFICIENT_N", "n": int(x.size), "min_n": min_n}
    return summarize(x - y, seed=seed, n_boot=n_boot, min_n=min_n)


def permutation_pvalue(a, b, *, seed=1234, n_perm=10000, min_n=3) -> dict:
    x, y = np.asarray(a, float), np.asarray(b, float)
    if x.shape != y.shape:
        raise ValueError("配对序列长度必须一致")
    if x.size < min_n:
        return {"status": "INSUFFICIENT_N", "n": int(x.size), "min_n": min_n}
    obs = abs(float((x - y).mean())); rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=x.size)
        hits += abs(float(((x - y) * signs).mean())) >= obs
    return {"status": "OK", "n": int(x.size), "observed_abs_mean": obs,
            "p_value": (hits + 1) / (n_perm + 1), "permutation_seed": seed,
            "permutations": n_perm}
