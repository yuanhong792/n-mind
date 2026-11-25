"""F0 净室数学验证：独立实现，不 import experiments/fly/fly_core.py。

E1 编码冲突率 / E2 容量×epoch / E3 更新干扰 / E4 decay 消融 / E5 clip 消融 /
E6 数值稳定性 / E7 资源基准。CPU float32，无梯度。
"""
from __future__ import annotations

import json
import math
import hashlib
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE
D_IN = 690
CHARS = "小明红号位置收到好了白"          # 13 唯一字符
UNI = 15
BIG = 15 * 15
ENC_SEED = 20271301
SEEDS = (101, 202, 303, 404, 505)


def encode_name(name: str) -> np.ndarray:
    h = np.zeros(D_IN, dtype=np.float32)
    for i, c in enumerate(name):
        h[CHARS.index(c)] += 1.0
    for pos in range(3):
        a, b = CHARS.index(name[pos]), CHARS.index(name[pos + 1])
        h[UNI + pos * BIG + a * 15 + b] += 1.0
    return np.minimum(h, 1.0)


def rand_names(rng: np.random.Generator, n: int) -> list[str]:
    return ["".join(rng.choice(list(CHARS), 4)) for _ in range(n)]


class FrozenEnc:
    """R∈{0,1}, fan_in, z=R h, 全局 TopK, 并列按索引升序（镜像 Caishen §2.2）。"""

    def __init__(self, M=512, fan_in=6, sparsity=0.02, seed=ENC_SEED, signed=False, mode="frozen"):
        self.M, self.k, self.mode = M, max(1, int(round(sparsity * M))), mode
        if mode in ("frozen", "signed"):
            rng = np.random.default_rng(seed)
            R = np.zeros((M, D_IN), dtype=np.float32)
            for r in range(M):
                cols = rng.choice(D_IN, fan_in, replace=False)
                R[r, cols] = rng.choice([-1.0, 1.0], fan_in) if signed else 1.0
            self.R = R
        elif mode == "random_sparse":
            self.rng = np.random.default_rng(seed)
        elif mode == "dense_pm1":
            self.rng = np.random.default_rng(seed)

    def encode(self, name: str) -> np.ndarray:
        out = np.zeros(self.M, dtype=np.float32)
        if self.mode in ("frozen", "signed"):
            z = self.R @ encode_name(name)
        elif self.mode == "dense_pm1":
            # 名字→固定 ±1 稠密码（sha256 稳定哈希，跨进程可复现）
            h = encode_name(name)
            dig = hashlib.sha256(h.tobytes() + name.encode("utf-8")).digest()
            rng = np.random.default_rng(int.from_bytes(dig[:8], "little") % (2**63))
            return rng.choice([-1.0, 1.0], self.M).astype(np.float32)
        else:
            idx = self.rng.choice(self.M, self.k, replace=False)
            out[idx] = 1.0
            return out
        idx = np.argsort(-z, kind="stable")[: self.k]
        out[idx] = 1.0
        return out


class Mem:
    def __init__(self, M, eta=0.2, decay=1.0, clip=1.0, normalized=False):
        self.W = np.zeros((M, M), dtype=np.float32)
        self.M, self.eta, self.decay, self.clip = M, eta, decay, clip
        self.normalized = normalized

    def write(self, a, b):
        if self.decay != 1.0:
            self.W *= self.decay
        delta = b - a @ self.W
        upd = np.outer(a, delta)
        if self.normalized:
            upd /= (float(a @ a) + 1e-6)
        self.W += self.eta * upd
        np.clip(self.W, -self.clip, self.clip, out=self.W)

    def scores(self, a, C):
        return (a @ self.W) @ C.T


def recall(enc, pairs, eta, decay, clip, epochs, normalized=False, M=None):
    M = M or enc.M
    mem = Mem(M, eta=eta, decay=decay, clip=clip, normalized=normalized)
    A = [enc.encode(k) for k, _ in pairs]
    B = [enc.encode(v) for _, v in pairs]
    for _ in range(epochs):
        for a, b in zip(A, B):
            mem.write(a, b)
    C = np.stack(B)
    hits = sum(int(np.argmax(mem.scores(a, C)) == i) for i, a in enumerate(A))
    return hits / len(pairs)


def e1_collision():
    rng = np.random.default_rng(ENC_SEED)
    names = rand_names(rng, 4000)
    out = {}
    for tag, enc in [("frozen", FrozenEnc(mode="frozen")),
                     ("signed", FrozenEnc(signed=True, mode="signed")),
                     ("random_sparse", FrozenEnc(mode="random_sparse")),
                     ("dense_pm1", FrozenEnc(mode="dense_pm1"))]:
        codes = [enc.encode(n) for n in names[:2000]]
        k = int((codes[0] != 0).sum())
        ov, cos = [], []
        for i in range(0, len(codes) - 1, 2):
            a, b = codes[i], codes[i + 1]
            ov.append(float(np.minimum(a, b).sum()))
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            cos.append(float(a @ b / (na * nb + 1e-12)))
        ov = np.asarray(ov)
        out[tag] = {"k": k, "mean_overlap_units": round(float(ov.mean()), 2),
                    "max_overlap_units": int(ov.max()),
                    "overlap_ratio_of_k": round(float(ov.mean() / k), 3),
                    "theory_independent_overlap": round(k * k / enc.M, 2),
                    "mean_cosine": round(float(np.mean(cos)), 3)}
        print("E1", tag, out[tag], flush=True)
    return out


def e2_capacity():
    rows = []
    for tag, mk in [("frozen", lambda: FrozenEnc(mode="frozen")),
                    ("signed", lambda: FrozenEnc(signed=True, mode="signed")),
                    ("random_sparse", lambda: FrozenEnc(mode="random_sparse"))]:
        for eta in (0.05, 0.2, 0.8):
            for P in (10, 50):
                for ep in (1, 5, 20, 50):
                    vals = []
                    for s in SEEDS[:3]:
                        rng = np.random.default_rng(s)
                        enc = mk()
                        pairs = list(zip(rand_names(rng, P), rand_names(rng, P)))
                        vals.append(recall(enc, pairs, eta, 1.0, 4.0, ep))
                    rows.append({"encoder": tag, "eta": eta, "P": P, "epochs": ep,
                                 "recall_mean": round(float(np.mean(vals)), 4),
                                 "recall_min": round(float(np.min(vals)), 4)})
                    print("E2", rows[-1], flush=True)
    return rows


def e3_interference():
    out = []
    rng = np.random.default_rng(ENC_SEED)
    for tag, mk in [("frozen", lambda: FrozenEnc(mode="frozen")),
                    ("random_sparse", lambda: FrozenEnc(mode="random_sparse"))]:
        for rho in (0.0, 0.3, 0.6, 0.9):
            for normalized in (False, True):
                enc = mk()
                names = rand_names(rng, 4)
                a1, b1, b2 = enc.encode(names[0]), enc.encode(names[1]), enc.encode(names[2])
                supp1 = np.flatnonzero(a1)
                keep = supp1[: int(round(rho * len(supp1)))]
                a2 = np.zeros_like(a1)
                a2[keep] = 1.0
                extra = rng.choice(np.setdiff1d(np.arange(enc.M), np.flatnonzero(a2)),
                                   max(0, len(supp1) - len(keep)), replace=False)
                a2[extra] = 1.0
                mem = Mem(enc.M, eta=0.2, decay=1.0, clip=1e9, normalized=normalized)
                mem.write(a1, b1)
                r_before = a1 @ mem.W
                d2 = b2 - a2 @ mem.W                 # 写入函数内部用的同一个 δ
                mem.write(a2, b2)
                diff = (a1 @ mem.W) - r_before
                predicted = 0.2 * float(a1 @ a2) * d2
                if normalized:
                    predicted = predicted / (float(a2 @ a2) + 1e-6)
                out.append({"encoder": tag, "rho": rho, "normalized": normalized,
                            "overlap_a1a2": int(round(float(a1 @ a2))),
                            "delta_read_norm": round(float(np.linalg.norm(diff)), 6),
                            "predicted_norm": round(float(np.linalg.norm(predicted)), 6),
                            "max_abs_error_vs_formula": round(float(np.max(np.abs(diff - predicted))), 8)})
                print("E3", out[-1], flush=True)
    return out


def e4_decay():
    rows = []
    for tag, mk in [("frozen", lambda: FrozenEnc(mode="frozen")),
                    ("signed", lambda: FrozenEnc(signed=True, mode="signed"))]:
        for decay in (1.0, 0.99, 0.9):
            for ep in (1, 10):
                vals = []
                for s in SEEDS[:3]:
                    rng = np.random.default_rng(s)
                    pairs = list(zip(rand_names(rng, 10), rand_names(rng, 10)))
                    vals.append(recall(mk(), pairs, 0.2, decay, 4.0, ep))
                rows.append({"encoder": tag, "decay": decay, "epochs": ep,
                             "P": 10, "recall_mean": round(float(np.mean(vals)), 4)})
                print("E4", rows[-1], flush=True)
    return rows


def e5_clip():
    rows = []
    for tag, mk in [("frozen", lambda: FrozenEnc(mode="frozen")),
                    ("signed", lambda: FrozenEnc(signed=True, mode="signed"))]:
        for clip in (0.05, 0.1, 0.5, 1.0, 4.0, 1e9):
            vals = []
            for s in SEEDS[:3]:
                rng = np.random.default_rng(s)
                pairs = list(zip(rand_names(rng, 10), rand_names(rng, 10)))
                vals.append(recall(mk(), pairs, 0.2, 1.0, clip, 5))
            rows.append({"encoder": tag, "clip": clip, "P": 10, "epochs": 5,
                         "recall_mean": round(float(np.mean(vals)), 4)})
            print("E5", rows[-1], flush=True)
    return rows


def e6_stability():
    M, k, steps, every = 256, 5, 50000, 5000
    out = {}
    for tag, normalized in [("plain_outer", False), ("normalized_delta", True)]:
        rng = np.random.default_rng(ENC_SEED)
        mem = Mem(M, eta=0.2, decay=1.0, clip=1e9, normalized=normalized)
        supp = np.stack([rng.choice(M, k, replace=False) for _ in range(steps)])
        vals = rng.choice([-1.0, 1.0], (steps, k)).astype(np.float32)
        Bv = rng.choice([-1.0, 1.0], (steps, M)).astype(np.float32) / math.sqrt(M)
        curve = []
        t0 = time.perf_counter()
        for i in range(steps):
            a = np.zeros(M, dtype=np.float32)
            a[supp[i]] = vals[i]
            mem.write(a, Bv[i])
            if (i + 1) % every == 0:
                curve.append({"step": i + 1, "fro_norm": round(float(np.linalg.norm(mem.W)), 3),
                              "max_abs": round(float(np.abs(mem.W).max()), 3)})
        dt = time.perf_counter() - t0
        out[tag] = {"M": M, "k": k, "steps": steps, "cpu_s": round(dt, 2),
                    "writes_per_s": round(steps / dt, 1), "curve": curve,
                    "extrapolated_fro_norm_1e6": round(curve[-1]["fro_norm"] * 20, 1)}
        print("E6", tag, out[tag]["cpu_s"], "s", curve[-1], flush=True)
    return out


def e7_resource():
    rows = []
    for M, k in ((2048, 102), (512, 10)):
        rng = np.random.default_rng(ENC_SEED)
        W = rng.standard_normal((M, M), dtype=np.float32)
        a = np.zeros(M, dtype=np.float32)
        idx = rng.choice(M, k, replace=False)
        a[idx] = 1.0
        reps = 1000
        t0 = time.perf_counter()
        for _ in range(reps):
            _ = a @ W
        dense_ms = (time.perf_counter() - t0) * 1000 / reps
        t0 = time.perf_counter()
        for _ in range(reps):
            _ = W[:, idx].sum(axis=1)
        sparse_ms = (time.perf_counter() - t0) * 1000 / reps
        rows.append({"M": M, "k": k, "dense_read_ms": round(dense_ms, 5),
                     "sparse_gather_read_ms": round(sparse_ms, 5),
                     "speedup": round(dense_ms / max(sparse_ms, 1e-9), 2),
                     "dense_mac": M * M, "ideal_sparse_mac": k * M,
                     "gather_elements": k * M,
                     "matrix_mb_fp32": round(M * M * 4 / 1e6, 2)})
        print("E7", rows[-1], flush=True)
    rows.append({"estimator_correction": {
        "proposal_claim": "Key=128, Value=32 → 32,768 元素 / 128 KiB / 稠密读 32,768 MAC",
        "arithmetic": "128×32=4096 元素=16 KiB；32,768 对应 1024×32",
        "corrected": "按提案下拉参数应为 4,096 元素、16 KiB、稠密读 4,096 MAC"}})
    return rows


def main():
    t0 = time.time()
    # 净室自检：P=1 必须 1.0
    sc = {}
    for tag, mk in [("frozen", lambda: FrozenEnc(mode="frozen")),
                    ("signed", lambda: FrozenEnc(signed=True, mode="signed")),
                    ("random_sparse", lambda: FrozenEnc(mode="random_sparse"))]:
        for normalized in (False, True):
            rng = np.random.default_rng(101)
            pairs = list(zip(rand_names(rng, 1), rand_names(rng, 1)))
            sc[f"{tag}_norm{int(normalized)}"] = recall(mk(), pairs, 0.2, 1.0, 1e9, 1, normalized)
    ok = all(v == 1.0 for v in sc.values())
    print("selfcheck", sc, "ok", ok, flush=True)
    res = {"eval_sanity": "N/A（无模型评测）", "selfcheck_P1": sc, "selfcheck_ok": ok,
           "E1": e1_collision()}
    (OUT / "F0_RESULTS.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    cap = e2_capacity()
    (OUT / "F0_MEMORY_CAPACITY.json").write_text(json.dumps(
        {"eval_sanity": "N/A", "rows": cap}, ensure_ascii=False, indent=2), encoding="utf-8")
    inter = e3_interference()
    res["E3"] = inter
    (OUT / "F0_RESULTS.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "F0_DECAY_ABLATION.json").write_text(json.dumps(
        {"eval_sanity": "N/A", "rows": e4_decay()}, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "F0_CLIP_ABLATION.json").write_text(json.dumps(
        {"eval_sanity": "N/A", "rows": e5_clip()}, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "F0_NUMERICAL_STABILITY.json").write_text(json.dumps(
        {"eval_sanity": "N/A", **e6_stability()}, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "F0_RESOURCE_BENCHMARK.json").write_text(json.dumps(
        {"eval_sanity": "N/A", "rows": e7_resource()}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("total_wall_s", round(time.time() - t0, 1), flush=True)


if __name__ == "__main__":
    main()
