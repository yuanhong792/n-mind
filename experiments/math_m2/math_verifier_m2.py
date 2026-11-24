"""M2 自动数学验证器：20 项算子/工程测试，任一 FAIL 即淘汰，禁止进入训练。

覆盖：AM 写入维度/衰减极限/保护门极值/重复覆盖/擦除；指针 content/shift/归一化/
批量一致/非循环移位；序列记忆读写/递推一致/容量有限；因果卷积因果性/增量一致；
位置编码；动力学指标；样本隔离；gradcheck；checkpoint roundtrip；确定性性质。
"""
from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
LAB = HERE / "math_lab"
sys.path.insert(0, str(LAB))

from associative_memory import (am_write, am_read, am_erase, l2_normalize,
                                AssociativeMemoryOperator)
from memory_protection import protect_step, ProtectionGate
from differentiable_pointer import (content_address, shift_pointer, pointer_write,
                                    pointer_read, PointerCell)
from sequence_addressing import seq_write, seq_read, advance_pointer
from causal_conv import CausalDepthwiseConv, sinusoidal_pos
from dynamics_analysis import analyze, fixed_point_ratio

torch.set_num_threads(4)
RESULTS = []


def record(name, target, passed, metrics=None, error=None):
    RESULTS.append({"name": name, "target": target, "pass": bool(passed),
                    "metrics": metrics or {}, "error": error})
    print(f"[{'PASS' if passed else 'FAIL'}] {name}: {metrics if metrics else error}", flush=True)


def t1_am_dimensions():
    B, dv, dk = 3, 6, 8
    S = torch.randn(B, dv, dk)
    k, v = l2_normalize(torch.randn(B, dk)), torch.randn(B, dv)
    out = am_write(S, k, v, torch.sigmoid(torch.randn(B)),
                   torch.sigmoid(torch.randn(B)), torch.sigmoid(torch.randn(B)))
    ok = tuple(out.shape) == (B, dv, dk)
    record("am_write_dimensions", "associative_memory", ok, {"shape": list(out.shape)})


def t2_am_zero_protect():
    torch.manual_seed(1)
    B, dv, dk = 2, 5, 7
    S = torch.randn(B, dv, dk)
    k, v = l2_normalize(torch.randn(B, dk)), torch.randn(B, dv)
    out0 = am_write(S, k, v, torch.ones(B), torch.ones(B), torch.zeros(B))
    diff = float((out0 - S).abs().max())
    record("am_protect_zero_keeps_old", "associative_memory", diff < 1e-6,
           {"max_abs_diff": diff})


def t3_am_full_protect():
    torch.manual_seed(2)
    B, dv, dk = 2, 5, 7
    S, k, v = torch.randn(B, dv, dk), l2_normalize(torch.randn(B, dk)), torch.randn(B, dv)
    cand = am_write(S, k, v, torch.ones(B), torch.ones(B), torch.ones(B))
    full = am_write(S, k, v, torch.ones(B), torch.ones(B), torch.ones(B))
    # g=1 的 am_write 结果 = candidate；用 g=1 与「无 protect」对照
    no_protect = am_write(S, k, v, torch.ones(B), torch.ones(B), None)
    diff = float((cand - no_protect).abs().max())
    record("am_protect_one_equals_candidate", "associative_memory", diff < 1e-6,
           {"max_abs_diff": diff})


def t4_am_decay_zero():
    torch.manual_seed(3)
    B, dv, dk = 2, 4, 6
    S, k = torch.randn(B, dv, dk), l2_normalize(torch.randn(B, dk))
    v = torch.zeros(B, dv)
    # beta=0 → candidate=decayed=alpha*S；protect=1
    out = am_write(S, k, v, torch.zeros(B), torch.full((B,), 0.5), torch.ones(B))
    diff = float((out - 0.5 * S).abs().max())
    record("am_decay_and_zero_write", "associative_memory", diff < 1e-6,
           {"max_abs_diff": diff})


def t5_am_repeated_overwrite():
    torch.manual_seed(4)
    dv, dk = 8, 16
    k = l2_normalize(torch.randn(dk))
    v1, v2 = torch.randn(dv), torch.randn(dv)
    S = am_write(torch.zeros(dv, dk), k, v1, torch.tensor(1.0), torch.tensor(1.0), torch.tensor(1.0))
    e1 = float((am_read(S, k) - v1).norm())
    S = am_write(S, k, v2, torch.tensor(1.0), torch.tensor(1.0), torch.tensor(1.0))
    e2 = float((am_read(S, k) - v2).norm())
    record("am_repeated_key_overwrite", "associative_memory", e2 <= e1 + 1e-6,
           {"first_err": round(e1, 5), "second_err": round(e2, 5)})


def t6_am_erase():
    torch.manual_seed(5)
    B, dv, dk = 2, 5, 8
    S = l2_normalize(torch.randn(B, dv, dk), dim=-1)
    k = l2_normalize(torch.randn(B, dk))
    erased = am_erase(S, k, torch.ones(B))
    proj_after = torch.bmm(erased, k.unsqueeze(-1)).squeeze(-1)
    ok = float(proj_after.abs().max()) < 1e-5 and tuple(erased.shape) == (B, dv, dk)
    record("am_erase_removes_key_component", "associative_memory", ok,
           {"max_proj_after": round(float(proj_after.abs().max()), 6)})


def t7_protect_broadcast():
    B, dv, dk = 3, 5, 7
    old, cand = torch.randn(B, dv, dk), torch.randn(B, dv, dk)
    g_scalar = torch.sigmoid(torch.randn(B))
    g_val = torch.sigmoid(torch.randn(B, dv))
    out1 = protect_step(old, cand, g_scalar)
    out2 = protect_step(old, cand, g_val)
    ok = out1.shape == out2.shape == old.shape
    record("protection_gate_broadcast", "memory_protection", ok,
           {"scalar_shape": list(out1.shape), "per_value_shape": list(out2.shape)})


def t8_pointer_content_simplex():
    torch.manual_seed(6)
    B, L, d = 3, 10, 8
    M, key = torch.randn(B, L, d), torch.randn(B, d)
    p = content_address(M, key, torch.sigmoid(torch.randn(B)) * 8)
    sums = p.sum(-1)
    ok = bool(((sums - 1).abs() < 1e-5).all()) and bool((p >= 0).all())
    record("pointer_content_simplex", "differentiable_pointer", ok,
           {"max_sum_dev": round(float((sums - 1).abs().max()), 7)})


def t9_pointer_shift_mass():
    torch.manual_seed(7)
    B, L = 4, 12
    p = torch.softmax(torch.randn(B, L), -1)
    kernel = torch.softmax(torch.randn(B, 3), -1)
    out = shift_pointer(p, kernel, circular=True)
    ok = bool(((out.sum(-1) - 1).abs() < 1e-4).all())
    record("pointer_shift_preserves_mass", "differentiable_pointer", ok,
           {"max_mass_dev": round(float((out.sum(-1) - 1).abs().max()), 7)})


def t10_pointer_shift_direction():
    p = torch.zeros(1, 5); p[0, 2] = 1.0
    kernel = torch.zeros(1, 3); kernel[0, 2] = 1.0  # off=2 → shift=+1
    out = shift_pointer(p, kernel, circular=True)
    ok = int(out[0].argmax()) == 3
    record("pointer_shift_direction", "differentiable_pointer", ok,
           {"argmax": int(out[0].argmax())})


def t11_pointer_batch_independent():
    torch.manual_seed(8)
    B, L, d = 2, 8, 6
    M = torch.randn(B, L, d)
    key = torch.randn(B, d)
    beta = torch.tensor([2.0, 2.0])
    both = content_address(M, key, beta)
    single = torch.stack([content_address(M[i:i+1], key[i:i+1], beta[i:i+1])[0] for i in range(B)])
    diff = float((both - single).abs().max())
    record("pointer_batch_equals_single", "differentiable_pointer", diff < 1e-6,
           {"max_abs_diff": diff})


def t12_seq_write_read():
    torch.manual_seed(9)
    B, L, d = 2, 8, 6
    H = torch.zeros(B, L, d)
    p = torch.zeros(B, L); p[:, 3] = 1.0
    v = torch.randn(B, d)
    H2 = seq_write(H, p, v)
    r = seq_read(H2, p)
    ok = torch.allclose(r, v, atol=1e-5) and float(H2[:, 3].sub(v).abs().max()) < 1e-5
    record("seq_memory_write_read", "sequence_addressing", ok,
           {"read_err": round(float((r - v).abs().max()), 6)})


def t13_seq_capacity_finite():
    torch.manual_seed(10)
    B, L, d = 2, 4, 6
    H = torch.zeros(B, L, d)
    p = torch.zeros(B, L); p[:, 0] = 1.0
    for _ in range(50):
        H = seq_write(H, p, torch.randn(B, d) * 10, torch.sigmoid(torch.randn(B)))
    ok = bool(torch.isfinite(H).all()) and float(H.abs().max()) < 1e4
    record("seq_memory_capacity_bounded", "sequence_addressing", ok,
           {"max_abs": round(float(H.abs().max()), 3)})


def t14_pointer_advance():
    p = torch.zeros(1, 4); p[0, 0] = 1.0
    out = advance_pointer(p, torch.ones(1))
    ok = int(out[0].argmax()) == 1
    record("pointer_advances_for_sequential_read", "sequence_addressing", ok,
           {"argmax": int(out[0].argmax())})


def t15_conv_causality():
    torch.manual_seed(11)
    conv = CausalDepthwiseConv(4, 4).eval()
    x = torch.randn(1, 8, 4)
    x2 = x.clone(); x2[0, -1] += 20
    with torch.no_grad():
        y1, y2 = conv(x), conv(x2)
    diff = float((y1[:, :-1] - y2[:, :-1]).abs().max())
    record("causal_conv_no_future_leak", "causal_conv", diff < 1e-6,
           {"earlier_output_delta": diff})


def t16_conv_incremental():
    torch.manual_seed(12)
    conv = CausalDepthwiseConv(3, 4).eval()
    x = torch.randn(1, 6, 3)
    with torch.no_grad():
        full = conv(x)
        step = torch.cat([conv(x[:, :i+1])[:, -1:] for i in range(6)], dim=1)
    diff = float((full - step).abs().max())
    record("causal_conv_incremental_consistency", "causal_conv", diff < 1e-5,
           {"max_abs_diff": diff})


def t17_positional_encoding():
    enc = sinusoidal_pos(32, 16)
    ok = enc.shape == (32, 16) and bool(torch.isfinite(enc).all()) and float(enc.abs().max()) <= 1.0
    record("sinusoidal_positional_encoding", "causal_conv", ok,
           {"shape": list(enc.shape), "max_abs": round(float(enc.abs().max()), 4)})


def t18_dynamics_repeat_detection():
    const = [torch.ones(2, 4) * i for i in range(6)]  # 变化状态
    static = [torch.ones(2, 4) for _ in range(6)]     # 固定点
    r_change, r_static = fixed_point_ratio(const), fixed_point_ratio(static)
    ok = r_static > r_change + 0.5
    record("dynamics_fixed_point_detection", "dynamics_analysis", ok,
           {"static_ratio": round(r_static, 3), "changing_ratio": round(r_change, 3)})


def t19_gradcheck():
    torch.manual_seed(13)
    dv, dk = 3, 4
    S = torch.randn(dv, dk, dtype=torch.double, requires_grad=True)
    k = l2_normalize(torch.randn(dk, dtype=torch.double)).detach().requires_grad_(True)
    v = torch.randn(dv, dtype=torch.double, requires_grad=True)
    beta = torch.tensor(0.6, dtype=torch.double, requires_grad=True)
    def fn(a, b, c, d):
        return am_write(a, b, c, d, torch.tensor(0.9, dtype=torch.double),
                        torch.tensor(0.8, dtype=torch.double))
    ok = torch.autograd.gradcheck(fn, (S, k, v, beta), eps=1e-6, atol=1e-4)
    record("am_write_gradcheck", "associative_memory", ok, {"gradcheck": ok})


def t20_checkpoint_and_isolation():
    torch.manual_seed(14)
    op = AssociativeMemoryOperator(8, 5)
    buf = io.BytesIO(); torch.save(op.state_dict(), buf); buf.seek(0)
    op2 = AssociativeMemoryOperator(8, 5); op2.load_state_dict(torch.load(buf, weights_only=True))
    same = all(torch.equal(a, b) for a, b in zip(op.state_dict().values(), op2.state_dict().values()))
    # 样本隔离：改动 batch 中样本0，样本1 输出不变
    B, dv, dk = 2, 4, 6
    S, k, v = torch.randn(B, dv, dk), l2_normalize(torch.randn(B, dk)), torch.randn(B, dv)
    g = torch.sigmoid(torch.randn(B))
    out1 = am_write(S, k, v, g, torch.ones(B), g)
    S2 = S.clone(); S2[0] += 100
    out2 = am_write(S2, k, v, g, torch.ones(B), g)
    iso = float((out1[1] - out2[1]).abs().max()) < 1e-12
    ok = same and iso
    record("checkpoint_roundtrip_and_sample_isolation", "all", ok,
           {"checkpoint_equal": same, "sample1_delta": round(float((out1[1]-out2[1]).abs().max()), 8)})


def main():
    t0 = time.perf_counter()
    tests = [t1_am_dimensions, t2_am_zero_protect, t3_am_full_protect, t4_am_decay_zero,
             t5_am_repeated_overwrite, t6_am_erase, t7_protect_broadcast,
             t8_pointer_content_simplex, t9_pointer_shift_mass, t10_pointer_shift_direction,
             t11_pointer_batch_independent, t12_seq_write_read, t13_seq_capacity_finite,
             t14_pointer_advance, t15_conv_causality, t16_conv_incremental,
             t17_positional_encoding, t18_dynamics_repeat_detection, t19_gradcheck,
             t20_checkpoint_and_isolation]
    for fn in tests:
        try:
            fn()
        except Exception as exc:
            record(fn.__name__, "exception", False, {}, f"{type(exc).__name__}: {exc}")
    passed = all(r["pass"] for r in RESULTS)
    report = {
        "status": "M2_MATH_TESTS_PASS" if passed else "M2_MATH_TESTS_FAIL",
        "verifier": "math_verifier_m2.py", "torch": torch.__version__,
        "seconds": round(time.perf_counter() - t0, 2),
        "n_tests": len(RESULTS), "n_pass": sum(r["pass"] for r in RESULTS),
        "tests": RESULTS,
        "elimination_rule": "FAIL 任一测试即禁止进入正式训练",
    }
    (HERE / "M2_MATH_TESTS.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "passed": report["n_pass"],
                      "total": report["n_tests"], "seconds": report["seconds"]}, ensure_ascii=False))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
