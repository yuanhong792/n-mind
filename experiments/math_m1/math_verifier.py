"""M1 自动数学验证器：候选算子正式训练前必须通过全部工程/数学测试。"""
from __future__ import annotations

import io
import json
import math
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
LAB = HERE / "math_lab"
sys.path.insert(0, str(LAB))

from delta_rule import DeltaMemoryOperator, delta_step, delta_read, l2_normalize
from erase_write import EraseWriteOperator, gated_delta2_step, shared_delta_step
from uncertainty_memory import UncertaintyMemoryOperator, diagonal_kalman_step
from causal_conv import CausalDepthwiseConv

torch.set_num_threads(4)
RESULTS = []


def record(name, target, passed, metrics=None, error=None):
    RESULTS.append({"name": name, "target": target, "pass": bool(passed),
                    "metrics": metrics or {}, "error": error})
    flag = "PASS" if passed else "FAIL"
    print(f"[{flag}] {name}: {metrics if metrics else error}", flush=True)


def finite(x: torch.Tensor) -> bool:
    return bool(torch.isfinite(x).all())


def test_dimensions():
    S = torch.randn(5, 8)
    k = l2_normalize(torch.randn(8))
    v = torch.randn(5)
    out = delta_step(S, k, v, torch.tensor(0.5), torch.tensor(0.9))
    out2 = gated_delta2_step(S, k, v, torch.tensor(0.4), torch.tensor(0.6), torch.tensor(0.9))
    B, dv, dk = 3, 5, 8
    Sb = torch.randn(B, dv, dk)
    kb = l2_normalize(torch.randn(B, dk))
    vb = torch.randn(B, dv)
    p = torch.ones(dk)
    kalman, pn, kap, res, err = diagonal_kalman_step(
        Sb[0], kb[0], vb[0], p, torch.ones(dk), torch.full((dk,), 0.01),
        torch.tensor(1.0))
    conv = CausalDepthwiseConv(4, 4)(torch.randn(2, 7, 4))
    ok = (tuple(out.shape) == (5, 8) and tuple(out2.shape) == (5, 8)
          and tuple(kalman.shape) == (5, 8) and tuple(pn.shape) == (8,)
          and tuple(kap.shape) == (8,) and tuple(res.shape) == (5,)
          and tuple(conv.shape) == (2, 7, 4))
    record("tensor_dimensions", "all", ok,
           {"delta": list(out.shape), "gated_delta2": list(out2.shape),
            "kalman_S": list(kalman.shape), "conv": list(conv.shape)})


def test_causality():
    torch.manual_seed(1)
    x = torch.randn(1, 6, 4)
    y1 = CausalDepthwiseConv(4, 4)(x)
    x2 = x.clone()
    x2[0, -1] += 10.0
    y2 = CausalDepthwiseConv(4, 4)(x2)  # 初始权重不同，改权重会干扰；改为同一实例 eval
    conv = CausalDepthwiseConv(4, 4).eval()
    with torch.no_grad():
        y1 = conv(x)
        y2 = conv(x2)
    diff = float((y1[:, :-1] - y2[:, :-1]).abs().max())
    record("causal_no_future_leak", "causal_conv", diff < 1e-12,
           {"max_earlier_output_delta": diff})


def test_gradcheck():
    torch.manual_seed(2)
    S = torch.randn(4, 3, dtype=torch.double, requires_grad=True)
    k = l2_normalize(torch.randn(3, dtype=torch.double)).detach().requires_grad_(True)
    v = torch.randn(4, dtype=torch.double, requires_grad=True)
    beta = torch.tensor(0.6, dtype=torch.double, requires_grad=True)
    ok = torch.autograd.gradcheck(
        lambda a, b, c, d: delta_step(a, b, c, d, torch.tensor(0.9, dtype=torch.double)),
        (S, k, v, beta), eps=1e-6, atol=1e-4)
    record("gradient_numerical_check", "delta_rule", ok, {"gradcheck": ok})


def test_shared_gate_equivalence():
    torch.manual_seed(3)
    S = torch.randn(4, 6)
    k = l2_normalize(torch.randn(6))
    v = torch.randn(4)
    beta, alpha = torch.tensor(0.7), torch.tensor(0.9)
    a = delta_step(S, k, v, beta, alpha)
    b = shared_delta_step(S, k, v, beta, alpha)
    diff = float((a - b).abs().max())
    record("shared_gate_matches_delta", "erase_write", diff < 1e-6,
           {"max_abs_diff": diff})


def test_finite_extreme():
    torch.manual_seed(4)
    S = torch.randn(6, 10) * 1e4
    k = l2_normalize(torch.randn(10) * 1e4)
    v = torch.randn(6) * 1e4
    out = S
    for _ in range(50):
        out = delta_step(out, k, v, torch.tensor(0.999), torch.tensor(0.999))
    kappa_ok = True
    p = torch.ones(10)
    Sn, pn, kappa, res, err = diagonal_kalman_step(
        S, k, v, p, torch.ones(10), torch.full((10,), 1e-4), torch.tensor(1e-6))
    ok = finite(out) and finite(Sn) and finite(pn) and finite(kappa)
    record("finite_extreme_inputs", "delta+kalman", ok,
           {"state_norm_after_50": round(float(out.norm()), 4),
            "kalman_state_norm": round(float(Sn.norm()), 4),
            "kappa_finite": bool(finite(kappa))})


def test_zero_input():
    S = torch.randn(4, 6)
    k = torch.zeros(6)
    v = torch.zeros(4)
    out = delta_step(S, k, v, torch.tensor(0.5), torch.tensor(0.9))
    diff = float((out - 0.9 * S).abs().max())
    record("zero_input", "delta_rule", diff < 1e-7, {"max_abs_diff": diff})


def test_repeated_write():
    torch.manual_seed(5)
    dk, dv = 16, 8
    k = l2_normalize(torch.randn(dk))
    v1, v2 = torch.randn(dv), torch.randn(dv)
    S = torch.zeros(dv, dk)
    S = delta_step(S, k, v1, torch.tensor(1.0))
    r1 = (delta_read(S, k) - v1).norm().item()
    S = delta_step(S, k, v2, torch.tensor(1.0))
    r2 = (delta_read(S, k) - v2).norm().item()
    record("repeated_key_overwrite", "delta_rule", r2 <= r1 + 1e-6,
           {"first_write_error": round(r1, 5), "second_write_error": round(r2, 5)})


def test_incremental_consistency():
    torch.manual_seed(6)
    dk, dv, T = 12, 7, 9
    keys = l2_normalize(torch.randn(T, dk))
    vals = torch.randn(T, dv)
    S = torch.zeros(dv, dk)
    for t in range(T):
        S = delta_step(S, keys[t], vals[t], torch.tensor(0.6), torch.tensor(0.98))
    # 与显式递归展开的矩阵式结果比较
    S2 = torch.zeros(dv, dk)
    for t in range(T):
        decayed = 0.98 * S2
        pred = decayed @ keys[t]
        S2 = decayed + 0.6 * (vals[t] - pred).unsqueeze(-1) * keys[t].unsqueeze(0)
    diff = float((S - S2).abs().max())
    record("incremental_inference_consistency", "delta_rule", diff < 1e-6,
           {"max_abs_diff": diff})


def test_batch_vs_sample():
    torch.manual_seed(7)
    B, dv, dk = 4, 6, 10
    S = torch.randn(B, dv, dk)
    k = l2_normalize(torch.randn(B, dk))
    v = torch.randn(B, dv)
    b = torch.sigmoid(torch.randn(B, dk))
    w = torch.sigmoid(torch.randn(B, dv))
    a = torch.sigmoid(torch.randn(B))
    batched = gated_delta2_step(S, k, v, b, w, a)
    singles = torch.stack([gated_delta2_step(S[i], k[i], v[i], b[i], w[i], a[i])
                           for i in range(B)])
    diff = float((batched - singles).abs().max())
    record("batch_equals_per_sample", "erase_write", diff < 1e-5,
           {"max_abs_diff": diff})


def test_no_state_leak():
    torch.manual_seed(8)
    B, dv, dk = 2, 6, 10
    S = torch.randn(B, dv, dk)
    k = l2_normalize(torch.randn(B, dk))
    v = torch.randn(B, dv)
    b = torch.sigmoid(torch.randn(B, dk))
    w = torch.sigmoid(torch.randn(B, dv))
    alpha = torch.sigmoid(torch.randn(B))
    out1 = gated_delta2_step(S, k, v, b, w, alpha)
    S2 = S.clone(); k2 = k.clone(); v2 = v.clone(); b2 = b.clone(); w2 = w.clone(); a2 = alpha.clone()
    S2[0] += 100; k2[0] += 3; v2[0] += 5; b2[0] += 0.2; w2[0] -= 0.2; a2[0] = 0.1
    out2 = gated_delta2_step(S2, k2, v2, b2, w2, a2)
    diff = float((out1[1] - out2[1]).abs().max())
    record("no_cross_sample_state_leak", "erase_write", diff < 1e-12,
           {"sample1_max_delta": diff})


def test_checkpoint_roundtrip():
    torch.manual_seed(9)
    out = []
    for cls in (DeltaMemoryOperator, EraseWriteOperator, UncertaintyMemoryOperator):
        op = cls(8, 5)
        buf = io.BytesIO()
        torch.save(op.state_dict(), buf)
        buf.seek(0)
        op2 = cls(8, 5)
        op2.load_state_dict(torch.load(buf, weights_only=True))
        same = all(torch.equal(a, b) for a, b in zip(op.state_dict().values(),
                                                      op2.state_dict().values()))
        out.append((cls.__name__, same))
    record("checkpoint_save_load", "all_operators", all(x[1] for x in out),
           {k: v for k, v in out})


def test_deterministic_properties():
    torch.manual_seed(10)
    dk, dv = 16, 8
    # 正交键存储误差
    Q, _ = torch.linalg.qr(torch.randn(dk, dk))
    keys = l2_normalize(Q[:dv])                         # [dv,dk]
    vals = torch.randn(dv, dv)
    S = torch.zeros(dv, dk)
    for i in range(dv):
        S = delta_step(S, keys[i], vals[i], torch.tensor(1.0))
    err = float(torch.stack([(delta_read(S, keys[i]) - vals[i]).norm() for i in range(dv)]).mean())

    # 非正交干扰：随机键之间余弦越高，误差越大
    torch.manual_seed(11)
    base = l2_normalize(torch.randn(dk))
    S2 = torch.zeros(dv, dk)
    S2 = delta_step(S2, base, torch.randn(dv), torch.tensor(1.0))
    near = l2_normalize(base + 0.35 * torch.randn(dk))
    near_err = float((delta_read(S2, near)).norm())

    # 变量数量曲线
    curve = {}
    for n in (2, 4, 8, 16, 32):
        torch.manual_seed(100 + n)
        K = l2_normalize(torch.randn(n, dk))
        V = torch.randn(n, dv)
        Sm = torch.zeros(dv, dk)
        for i in range(n):
            Sm = delta_step(Sm, K[i], V[i], torch.tensor(0.9))
        e = float(torch.stack([(delta_read(Sm, K[i]) - V[i]).norm() for i in range(n)]).mean())
        curve[str(n)] = round(e, 4)
    ok = err < 1e-4 and curve["2"] < curve["32"]
    record("deterministic_math_properties", "delta_rule", ok,
           {"orthogonal_mean_error": round(err, 6),
            "nonorthogonal_leak": round(near_err, 4),
            "variable_count_error": curve})


def main():
    t0 = time.perf_counter()
    tests = [test_dimensions, test_causality, test_gradcheck, test_shared_gate_equivalence,
             test_finite_extreme, test_zero_input, test_repeated_write,
             test_incremental_consistency, test_batch_vs_sample, test_no_state_leak,
             test_checkpoint_roundtrip, test_deterministic_properties]
    for fn in tests:
        try:
            fn()
        except Exception as exc:
            record(fn.__name__, "exception", False, {}, f"{type(exc).__name__}: {exc}")
    passed = all(r["pass"] for r in RESULTS)
    report = {
        "status": "M1_MATH_TESTS_PASS" if passed else "M1_MATH_TESTS_FAIL",
        "verifier": "math_verifier.py",
        "torch": torch.__version__,
        "seconds": round(time.perf_counter() - t0, 2),
        "tests": RESULTS,
        "elimination_rule": "FAIL 任一测试即淘汰对应候选，不得进入短时训练",
    }
    (HERE / "M1_MATH_TESTS.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "passed": sum(r["pass"] for r in RESULTS),
                      "total": len(RESULTS), "seconds": report["seconds"]}, ensure_ascii=False))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
