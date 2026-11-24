"""P2：R2 微型 Transformer 真实计算成本解剖（零梯度，inference_mode）。

分三种口径，互不替代：
  1. 解析 MAC：按公式计算各模块乘加数，区分因果有效三角区与实际稠密执行量。
  2. Profiler：torch.profiler CPU 热点（含Python调度开销）。
  3. 真实墙钟：预热 + 多次重复，中位数与 P95，model.eval() + torch.inference_mode()。

长度：16/32/64/128/256/512；Batch：1/8/16；FP32。
同时测量 KV Cache 增量生成（T=1，L=前文长度）与 Answer-only Head 可节省量。

运行：python benchmark_runner.py
"""

from __future__ import annotations

import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

HERE = Path(__file__).resolve().parents[0]  # experiments/math_m4
M3 = HERE.parent / "math_m3"
sys.path.insert(0, str(M3))
sys.path.insert(0, str(M3 / "models"))

from transformer_reference import TinyTransformerM3  # noqa: E402

LENGTHS = [16, 32, 64, 128, 256, 512]
BATCHES = [1, 8, 16]
D, H, LAYERS, V = 64, 4, 2, 102


def analytic_mac(B: int, T: int, L: int | None = None) -> dict:
    """解析 MAC。L 为 key 长度；None 表示完整因果前向 L=T。"""
    L = L if L is not None else T
    dh = D // H
    per_layer = {
        "qkv_proj": B * T * D * 3 * D,
        "rope_elementwise": B * T * H * dh * 2,
        "score_qk_dense": B * H * T * L * dh,
        "score_qk_causal_effective": B * H * dh * T * (T + 1) // 2,
        "softmax_non_mac": 0,
        "attn_v_dense": B * H * T * L * dh,
        "attn_v_causal_effective": B * H * dh * T * (T + 1) // 2,
        "out_proj": B * T * D * D,
        "ffn_up": B * T * D * 2 * D,
        "ffn_down": B * T * 2 * D * D,
        "layernorm_elementwise": B * T * D * 4,
    }
    per_layer["attention_block_total_dense"] = sum(
        per_layer[k] for k in ("qkv_proj", "score_qk_dense", "attn_v_dense", "out_proj"))
    per_layer["ffn_total"] = per_layer["ffn_up"] + per_layer["ffn_down"]
    total = sum(v for k, v in per_layer.items() if not k.startswith("score_qk_causal")
                and not k.startswith("attn_v_causal") and not k.endswith("_total_dense")
                and k != "ffn_total")
    return {"per_layer": per_layer, "model_without_head": total,
            "vocab_head_all_positions": B * T * D * V,
            "vocab_head_answer_only_estimate": B * min(T, 4) * D * V,
            "note": "Answer-only 估计按每样本4个监督位置；实际值由数据分布决定",
            "causal_savings_ratio": round(1 - (per_layer["score_qk_causal_effective"] +
                     per_layer["attn_v_causal_effective"]) /
                     max(1, per_layer["score_qk_dense"] + per_layer["attn_v_dense"]), 4)}


def measure_wall(model, B: int, T: int, repeats: int = 30, warmup: int = 5):
    model.eval()
    x = torch.randint(4, V, (B, T))
    with torch.inference_mode():
        for _ in range(warmup):
            model(x)
        ts = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            model(x)
            ts.append(time.perf_counter() - t0)
    return {"median_sec": round(statistics.median(ts), 6),
            "p95_sec": round(sorted(ts)[int(0.95 * (len(ts) - 1))], 6),
            "min_sec": round(min(ts), 6), "repeats": repeats, "warmup": warmup}


def measure_incremental(model, B: int, L: int, repeats: int = 20):
    """KV Cache 增量：prefill L 个 token（不计时），再逐 token 生成 QA 步。"""
    model.eval()
    pre = torch.randint(4, V, (B, L))
    with torch.inference_mode():
        _, cache = model(pre)
        tok = torch.randint(4, V, (B, 1))
        for _ in range(3):  # warmup
            _, cache = model(tok, state=cache)
        ts = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            _, cache = model(tok, state=cache)
            ts.append(time.perf_counter() - t0)
    return {"L": L, "median_sec": round(statistics.median(ts), 6),
            "p95_sec": round(sorted(ts)[int(0.95 * (len(ts) - 1))], 6)}


def profile_once(model, B: int, T: int):
    model.eval()
    x = torch.randint(4, V, (B, T))
    with torch.inference_mode():
        for _ in range(3):
            model(x)
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            for _ in range(5):
                model(x)
    events = sorted(prof.key_averages(), key=lambda e: e.self_cpu_time_total, reverse=True)[:12]
    total = sum(e.self_cpu_time_total for e in prof.key_averages())
    return {"top": [{"name": e.key, "self_cpu_us": round(e.self_cpu_time_total, 1),
                     "share": round(e.self_cpu_time_total / max(1.0, total), 4)} for e in events],
            "total_self_cpu_us": round(total, 1)}


def main():
    torch.set_num_threads(20)
    model = TinyTransformerM3(V, d_model=D, nhead=H, layers=LAYERS)
    out = {
        "status": "COMPLETE", "precision": "FP32",
        "hardware": {"cpu": "Intel64 Family 6 Model 183 Stepping 1", "logical_cores": 28,
                     "torch_threads": torch.get_num_threads(), "torch": torch.__version__},
        "model": {"params": sum(p.numel() for p in model.parameters()),
                  "d_model": D, "nhead": H, "layers": LAYERS, "vocab": V,
                  "qkv": "fused Linear(d,3d)"},
        "method": {"wall": "model.eval()+inference_mode, 预热5次, 重复30次, 中位数/P95",
                   "profiler": "torch.profiler CPU, 5 次前向累积",
                   "analytic": "逐模块 MAC 公式；因果有效三角区与实际稠密执行量分列"},
        "by_length": {}, "incremental": {}, "profiler": {},
    }
    for T in LENGTHS:
        rec = {}
        for B in BATCHES:
            rec[f"B{B}"] = {"analytic": analytic_mac(B, T), "wall": measure_wall(model, B, T)}
        out["by_length"][str(T)] = rec
        print(f"[cost] T={T} B1 median={rec['B1']['wall']['median_sec']*1000:.3f}ms "
              f"B16 median={rec['B16']['wall']['median_sec']*1000:.3f}ms", flush=True)
    for L in (16, 64, 256):
        out["incremental"][str(L)] = measure_incremental(model, 1, L)
    out["profiler"]["B1_T32"] = profile_once(model, 1, 32)
    out["profiler"]["B8_T128"] = profile_once(model, 8, 128)

    # 瓶颈判定（基于实测 + 解析，不凭直觉）
    a32 = analytic_mac(1, 32)["per_layer"]
    denom = a32["attention_block_total_dense"] + a32["ffn_total"] + D * V
    out["bottleneck_short_seq"] = {
        "attention_share": round(a32["attention_block_total_dense"] / denom, 4),
        "ffn_share": round(a32["ffn_total"] / denom, 4),
        "head_share": round(D * V / denom, 4),
        "note": "短序列若 FFN/Head 占比更大，不得盲目优化注意力矩阵",
    }
    a512 = analytic_mac(1, 512)["per_layer"]
    denom512 = a512["attention_block_total_dense"] + a512["ffn_total"] + 512 * D * V
    out["bottleneck_long_seq"] = {
        "attention_share": round(a512["attention_block_total_dense"] / denom512, 4),
        "ffn_share": round(a512["ffn_total"] / denom512, 4),
        "head_share": round(512 * D * V / denom512, 4),
    }
    p = HERE / "M4_TRANSFORMER_COST_ANALYSIS.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[cost] 写出 {p}")


if __name__ == "__main__":
    main()
