"""M41 M4-A 低秩 KV Cache 独立审计（零梯度）。

逐位置比较完整前向、逐 token 增量、分块 prefill；
显式检查 cache 形状、元素数、FP32 字节数，禁止广播掩盖形状错误。

输出 M41_KV_CACHE_AUDIT.json。
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "models"))
from low_rank_interaction import LowRankInteractionM4  # noqa: E402

TOL = 1e-4


def shape_diff(a, b, label, failures):
    if tuple(a.shape) != tuple(b.shape):
        failures.append({"label": label, "kind": "shape_mismatch",
                         "a": list(a.shape), "b": list(b.shape)})
        return None
    return float((a - b).abs().max().detach().item())


def run_case(model, B, T, seed):
    torch.manual_seed(seed)
    x = torch.randint(4, 100, (B, T))
    valid = torch.ones(B, T, dtype=torch.bool)
    failures = []
    with torch.no_grad():
        full, st_full = model(x, valid_mask=valid)

        state = model.initial_state(B, x.device)
        inc = []
        for t in range(T):
            y, state = model(x[:, t:t + 1], state=state,
                             valid_mask=valid[:, t:t + 1])
            inc.append(y)
        incremental = torch.cat(inc, dim=1)
        inc_diff = shape_diff(full, incremental, f"B{B}_T{T}_incremental", failures)

        chunks, rem = [], T
        for c in (1, 3, 4, 5, 7, 8, 16):
            if rem <= 0:
                break
            n = min(c, rem)
            chunks.append(n)
            rem -= n
        while rem > 0:
            chunks.append(min(16, rem))
            rem -= min(16, rem)
        state = model.initial_state(B, x.device)
        outs, off = [], 0
        for n in chunks:
            y, state = model(x[:, off:off + n], state=state,
                             valid_mask=valid[:, off:off + n])
            outs.append(y)
            off += n
        chunked = torch.cat(outs, dim=1)
        chunk_diff = shape_diff(full, chunked, f"B{B}_T{T}_chunked", failures)

        ck, cv = st_full["ck"], st_full["cv"]
        cache_shapes = {"ck": [list(k.shape) for k in ck],
                        "cv": [list(v.shape) for v in cv]}
        h, rh, dh = model.blocks[0].attn.h, model.blocks[0].attn.rh, model.blocks[0].attn.dh
        expected_ck, expected_cv = [B, h, T, rh], [B, h, T, dh]
        shape_ok = all(list(k.shape) == expected_ck for k in ck) and \
            all(list(v.shape) == expected_cv for v in cv)
        if not shape_ok:
            failures.append({"label": f"B{B}_T{T}_cache_shape", "kind": "unexpected_cache_shape",
                             "observed": cache_shapes,
                             "expected": {"ck": expected_ck, "cv": expected_cv}})
        element_counts = [int(k.numel() + v.numel()) for k, v in zip(ck, cv)]
        byte_counts = [int((k.numel() + v.numel()) * k.element_size()) for k, v in zip(ck, cv)]
        expected_elements = B * h * T * (rh + dh)
        expected_bytes = expected_elements * 4
        if any(n != expected_elements for n in element_counts) or \
                any(n != expected_bytes for n in byte_counts):
            failures.append({"label": f"B{B}_T{T}_cache_size", "kind": "unexpected_cache_size",
                             "elements": element_counts, "bytes": byte_counts,
                             "expected_elements": expected_elements,
                             "expected_bytes": expected_bytes})
        state_ok = state["pos"] == T and len(state["ck"]) == len(model.blocks)
        if not state_ok:
            failures.append({"label": f"B{B}_T{T}_state", "kind": "bad_state",
                             "pos": state["pos"], "n_blocks": len(state["ck"])})

    ok = (not failures and inc_diff is not None and chunk_diff is not None
          and inc_diff <= TOL and chunk_diff <= TOL)
    return {"B": B, "T": T, "seed": seed, "chunks": chunks,
            "max_abs_diff_incremental": inc_diff,
            "max_abs_diff_chunked": chunk_diff,
            "cache_shapes": cache_shapes,
            "expected_cache_shapes": {"ck": expected_ck, "cv": expected_cv},
            "cache_elements_per_block": element_counts,
            "cache_bytes_fp32_per_block": byte_counts,
            "expected_elements_per_block": expected_elements,
            "expected_bytes_per_block": expected_bytes,
            "failures": failures, "pass": ok}


def audit():
    torch.set_num_threads(20)
    model = LowRankInteractionM4(102, d_model=64, nhead=4, layers=2, r=16)
    model.eval()
    cases = [run_case(model, 1, 16, 20261130),
             run_case(model, 2, 17, 20261131),
             run_case(model, 3, 32, 20261132)]
    status = "PASS" if all(c["pass"] for c in cases) else "FAILED"
    out = {
        "status": status,
        "model": "M4-A_low_rank_interaction",
        "comparison_rule": "shape 完全相等后逐位置 max_abs_diff；禁止广播比较",
        "tolerance": TOL,
        "scale_rule": "score=sqrt(reduced_head_dim)，value_aggregation=value_head_dim",
        "cases": cases,
    }
    p = HERE / "M41_KV_CACHE_AUDIT.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[m41-kv] status={status} 写出 {p}")
    return out


if __name__ == "__main__":
    audit()
