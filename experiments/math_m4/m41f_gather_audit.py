"""M41F 选择性交互 Gather 内存/延迟审计（零梯度）。

对比：
- 旧实现：Kt.unsqueeze(2).expand(...).reshape(...) 为每个 query 物化全部历史；
- 新实现：按 (batch,head) 直接高级索引，只复制选中槽位。

记录 Profiler 观察到的实际分配字节、中间元素数、Tensor 复制字节、
Prefill/增量延迟、真实矩阵乘次数、logits 最大误差。输出 M41F_GATHER_AUDIT.json。
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

import sys
MODELS = Path(__file__).resolve().parent / "models"
sys.path.insert(0, str(MODELS))
from selective_interaction import GroupedSelectiveAttention, SelectiveInteractionM4  # noqa: E402

HERE = Path(__file__).resolve().parent
TOL = 1e-4


def old_gather(Kt, Vt, pos):
    B, h, T, S = pos.shape
    L, dh = Kt.shape[2], Kt.shape[-1]
    idx = pos.reshape(B * h * T, S).unsqueeze(-1).expand(-1, -1, dh)
    k_exp = Kt.unsqueeze(2).expand(B, h, T, L, dh).reshape(B * h * T, L, dh)
    v_exp = Vt.unsqueeze(2).expand(B, h, T, L, dh).reshape(B * h * T, L, dh)
    k_flat = k_exp.gather(1, idx)
    v_flat = v_exp.gather(1, idx)
    old_elements = 2 * B * h * T * L * dh
    new_elements = 2 * B * h * T * S * dh
    return (k_flat.reshape(B, h, T, S, dh), v_flat.reshape(B, h, T, S, dh),
            {"old_expand_elements": old_elements, "new_gather_elements": new_elements,
             "saved_elements": old_elements - new_elements,
             "copy_bytes": int(new_elements * Kt.element_size()),
             "slots_per_query": S})


def profiled_call(fn, repeats=5):
    for _ in range(2):
        fn()
    with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof:
        for _ in range(repeats):
            fn()
    rows = []
    total = 0
    for e in prof.key_averages():
        mem = int(getattr(e, "self_cpu_memory_usage", 0) or 0)
        if mem > 0:
            total += mem
            rows.append({"name": e.key, "self_cpu_memory_bytes": mem, "count": e.count})
    return {"allocated_bytes_total": total,
            "allocated_bytes_per_call": round(total / repeats, 2),
            "ops": sorted(rows, key=lambda x: -x["self_cpu_memory_bytes"])[:8]}


def timed(fn, repeats=20, warmup=5):
    for _ in range(warmup):
        fn()
    vals = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        vals.append(time.perf_counter() - t0)
    return {"median_ms": round(statistics.median(vals) * 1000, 4),
            "p95_ms": round(sorted(vals)[int(0.95 * (len(vals) - 1))] * 1000, 4)}


def run_generation(model, x, steps=16):
    with torch.no_grad():
        logits, state = model(x)
        for _ in range(steps):
            logits, state = model(torch.zeros(1, 1, dtype=torch.long), state=state,
                                  valid_mask=torch.ones(1, 1, dtype=torch.bool))
    return logits


def main():
    torch.set_num_threads(20)
    torch.manual_seed(20261124)
    new_fn = GroupedSelectiveAttention._gather
    model = SelectiveInteractionM4(102, d_model=64, nhead=4, layers=2,
                                   group_size=4, top_k=2).eval()

    B, T, L, dh = 2, 17, 32, 16
    x = torch.randint(4, 100, (B, T))
    K = torch.randn(2, 4, L, dh)
    V = torch.randn(2, 4, L, dh)
    pos = torch.randint(0, L, (2, 4, T, 12))

    GroupedSelectiveAttention._gather = staticmethod(new_fn)
    new_k, new_v, new_meta = new_fn(K, V, pos)
    new_prof = profiled_call(lambda: new_fn(K, V, pos))
    GroupedSelectiveAttention._gather = staticmethod(old_gather)
    old_k, old_v, old_meta = old_gather(K, V, pos)
    old_prof = profiled_call(lambda: old_gather(K, V, pos))
    gather_diff = float((new_k - old_k).abs().max().item())
    gather_diff_v = float((new_v - old_v).abs().max().item())

    GroupedSelectiveAttention._gather = staticmethod(new_fn)
    with torch.no_grad():
        new_logits, new_state = model(x)
    new_stats = new_state["stats"]
    GroupedSelectiveAttention._gather = staticmethod(old_gather)
    with torch.no_grad():
        old_logits, old_state = model(x)
    old_stats = old_state["stats"]
    logits_diff = float((new_logits - old_logits).abs().max().item())

    prefill_x = torch.randint(4, 100, (1, 64))

    def prefill_once():
        with torch.no_grad():
            return model(prefill_x)

    GroupedSelectiveAttention._gather = staticmethod(new_fn)
    new_prefill = timed(prefill_once)
    new_gen = timed(lambda: run_generation(model, prefill_x))
    new_stats_prefill = prefill_once()[1]["stats"]
    GroupedSelectiveAttention._gather = staticmethod(old_gather)
    old_prefill = timed(prefill_once)
    old_gen = timed(lambda: run_generation(model, prefill_x))
    old_stats_prefill = prefill_once()[1]["stats"]
    GroupedSelectiveAttention._gather = staticmethod(new_fn)

    mem_saved = old_prof["allocated_bytes_per_call"] - new_prof["allocated_bytes_per_call"]
    short_actual = new_stats["total_executed_macs"]
    short_dense = sum(b["dense_attention_macs"] for b in new_stats["blocks"])
    prefill_actual = new_stats_prefill["total_executed_macs"]
    prefill_dense = sum(b["dense_attention_macs"] for b in new_stats_prefill["blocks"])
    overhead = {
        "short_T17_B2_actual_executed_macs": short_actual,
        "short_T17_B2_dense_attention_macs": short_dense,
        "short_T17_B2_ratio": round(short_actual / max(1, short_dense), 4),
        "short_sequence_controller_overhead_exceeds_dense": short_actual > short_dense,
        "prefill_T64_B1_actual_executed_macs": prefill_actual,
        "prefill_T64_B1_dense_attention_macs": prefill_dense,
        "prefill_T64_B1_ratio": round(prefill_actual / max(1, prefill_dense), 4),
    }
    checks = {
        "GATHER_LOGITS_EQUAL": logits_diff <= TOL and gather_diff <= TOL and gather_diff_v <= TOL,
        "NEW_MEMORY_LE_NOT_OLD": new_prof["allocated_bytes_per_call"] <= old_prof["allocated_bytes_per_call"],
        "NEW_GATHER_ELEMENTS_LE_OLD": new_meta["new_gather_elements"]
        <= old_meta["old_expand_elements"],
        "TRUE_MATMUL_COUNT_EQUAL": new_stats["total_executed_macs"] == old_stats["total_executed_macs"],
    }
    out = {
        "status": "PASS" if all(checks.values()) else "FAILED",
        "checks": checks,
        "note": ("直接索引避免为每个 query 物化全部历史；"
                 "但内存访问仍有成本，是否值得必须结合训练后真实能力/延迟判断。"),
        "gather_shapes": {"B": B, "h": 4, "T": T, "L": L, "S": 12, "dh": dh},
        "elements": {"old_expand_elements": old_meta["old_expand_elements"],
                     "new_gather_elements": new_meta["new_gather_elements"],
                     "saved_elements": new_meta["saved_elements"],
                     "new_copy_bytes": new_meta["copy_bytes"]},
        "profiler_memory": {"new_direct_index": new_prof,
                            "old_expand_reshape": old_prof,
                            "saved_bytes_per_call": round(mem_saved, 2)},
        "logits_max_abs_diff_old_vs_new": logits_diff,
        "gather_output_max_abs_diff": max(gather_diff, gather_diff_v),
        "true_matmul_macs": {"new": new_stats["total_executed_macs"],
                             "old": old_stats["total_executed_macs"],
                             "dense_reference": short_dense},
        "compute_overhead_by_length": overhead,
        "latency": {
            "prefill_T64_new_direct_ms": new_prefill,
            "prefill_T64_old_expand_ms": old_prefill,
            "incremental_prefill_plus_16_new_ms": new_gen,
            "incremental_prefill_plus_16_old_ms": old_gen,
            "prefill_delta_pct": round(
                (new_prefill["median_ms"] - old_prefill["median_ms"]) /
                max(1e-9, old_prefill["median_ms"]) * 100, 2),
            "incremental_delta_pct": round(
                (new_gen["median_ms"] - old_gen["median_ms"]) /
                max(1e-9, old_gen["median_ms"]) * 100, 2),
        },
        "gather_stats_new": {k: new_stats[k] for k in
                             ("gather_old_expand_elements", "gather_new_elements",
                              "gather_copy_bytes")},
        "gather_stats_old": {k: old_stats[k] for k in
                             ("gather_old_expand_elements", "gather_new_elements",
                              "gather_copy_bytes")},
        "prefill_stats": {"new": new_stats_prefill, "old": old_stats_prefill},
    }
    p = HERE / "M41F_GATHER_AUDIT.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[m41f-gather] status={out['status']} mem_saved={mem_saved} "
          f"prefill_delta={out['latency']['prefill_delta_pct']}% 写出 {p}")
    return out


if __name__ == "__main__":
    main()
