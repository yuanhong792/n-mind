"""M3 计算成本核算系统（P10）。

同时报告三类成本：
1. 数学成本：MAC、FLOP、非线性算子数、状态更新次数、动态路径执行次数。
2. 硬件成本：CPU 延迟、Prompt 吞吐、自回归生成吞吐、峰值内存、状态内存、权重体积、线程数。
3. 学习成本：达标所需样本、步数、Token、累计计算量（由训练结果回填）。

统一约定：
- MAC（乘加）计数：Linear(d_in,d_out) 一次 MAC = d_in*d_out；注意力 MAC 按 q·k 与 softmax(·)v。
- FLOP = 2 * MAC（一次乘加记 2 次浮点运算）。
- 非线性算子（tanh/sigmoid/GELU/ReLU）逐元素，不计 MAC 但计数。
- 状态更新次数 = 每个时间步的状态写入次数。
- 动态路径执行次数 = M3-B 各路径实际执行次数（由 collect_stats 采集）。

性能测量：预热、多次重复、中位数、P95；能耗无可信数据时标 NOT_AVAILABLE（禁止 TDP*时间伪造）。
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "models"))

from gru_reference import GRUReferenceM3          # noqa: E402
from transformer_reference import TinyTransformerM3  # noqa: E402
from low_rank_neural import LowRankNeuralM3       # noqa: E402
from adaptive_computation import AdaptiveComputationM3  # noqa: E402

VOCAB = 102
THREADS = 20


# ---------------------------------------------------------------------------
# 数学成本：解析公式
# ---------------------------------------------------------------------------

def model_mac_profile(name: str, V: int, **cfg) -> dict:
    """每个模型单 token 前向 MAC（解析），含动态路径（若适用）。

    返回 dict: per_token_forward_mac, nonlinear_ops_per_token,
              state_updates_per_token, n_layers, d。
    """
    if name == "R1_gru_reference":
        d_model = cfg.get("d_model", 64)
        hidden = cfg.get("hidden", 128)
        # GRUCell: 3 门 * (d_model*hidden + hidden*hidden) 近似
        gru = 3 * (d_model * hidden + hidden * hidden)
        mix = hidden * hidden
        head = hidden * V
        return {
            "per_token_forward_mac": gru + mix + head,
            "nonlinear_ops_per_token": 3 + 1,   # 3 门 sigmoid/tanh + GELU
            "state_updates_per_token": 1,
            "n_layers": 1,
            "state_dim": hidden,
        }
    if name == "R2_tiny_transformer":
        d = cfg.get("d_model", 128)
        n_head = cfg.get("n_head", 4)
        dh = d // n_head
        n_layers = cfg.get("n_layers", 2)
        ff = cfg.get("ff_mult", 2)
        head = d * V
        per_layer_attn = 3 * d * d + 3 * d * d + 2 * (n_head * dh * dh)  # qkv+proj+attn matmul
        # attn 的 q·k 与 ·v 各 dh*L*T，T=1 增量时近似 2*dh
        per_layer_attn += 2 * (dh + dh)  # 单 token 增量
        per_layer_ff = 2 * (d * d * ff)
        per_token = head + n_layers * (per_layer_attn + per_layer_ff)
        return {
            "per_token_forward_mac": per_token,
            "nonlinear_ops_per_token": n_layers * (1 + 2 * ff) + 1,
            "state_updates_per_token": n_layers * 2,
            "n_layers": n_layers,
            "state_dim": n_layers * 2 * d,
        }
    if name == "M3-A_low_rank_neural":
        d = cfg.get("d", 96)
        r = cfg.get("r", 16)
        # 低秩 2*d*r，对角 d，Wx d*d，gate d*d，head d*V
        per_token = 2 * d * r + d + d * d + d * d + d * V
        return {
            "per_token_forward_mac": per_token,
            "nonlinear_ops_per_token": 3,   # tanh + sigmoid(gate) + (head)
            "state_updates_per_token": 1,
            "n_layers": cfg.get("n_layers", 1),
            "state_dim": d,
        }
    if name == "M3-B_adaptive_computation":
        d = cfg.get("d", 96)
        r = cfg.get("r", 16)
        hm = cfg.get("hidden_mult", 2)
        # controller 2d*3；基础对角 d；低秩 2d*r；扩展 ff 2*(d*d*hm)；head d*V
        base = d
        c1 = 2 * d * r
        c2 = 2 * d * d * hm
        per_token_all_paths = 2 * d * 3 + base + c1 + c2 + d * V
        return {
            "per_token_forward_mac": per_token_all_paths,
            "nonlinear_ops_per_token": 4,   # controller softmax + GELU + read sigmoid + head
            "state_updates_per_token": 1,
            "n_layers": 1,
            "state_dim": d,
            "path_macs": {"C0": base, "C1": base + c1, "C2": base + c1 + c2},
        }
    raise ValueError(f"未知模型: {name}")


# ---------------------------------------------------------------------------
# 硬件成本：实测
# ---------------------------------------------------------------------------

def _time_fn(fn, warmup=3, repeats=10):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeats):
        s = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - s)
    ts.sort()
    return {
        "median_sec": statistics.median(ts),
        "p95_sec": ts[int(len(ts) * 0.95)] if len(ts) >= 20 else ts[-1],
        "min_sec": ts[0],
        "max_sec": ts[-1],
    }


def measure_model(name: str, model, tokens_fn, V: int, seq_lens, batch_sizes):
    """测量单模型在 (BS, seq) 下的前向延迟、生成吞吐、峰值内存、权重体积。"""
    model.eval()
    torch.set_num_threads(THREADS)
    cfg = model.cfg if hasattr(model, "cfg") else {}
    prof = model_mac_profile(name, V, **cfg)

    # 权重体积
    nbytes = sum(p.numel() * p.element_size() for p in model.parameters())

    rows = []
    for bs in batch_sizes:
        for seq in seq_lens:
            # 前向延迟（full sequence）
            x = tokens_fn(bs, seq)
            fwd = _time_fn(lambda: model(x, valid_mask=torch.ones(bs, seq, dtype=torch.bool)))
            # 自回归生成吞吐：逐 token
            def gen():
                s = model.initial_state(bs, x.device)
                logits, st = model(x[:, :1], state=s, valid_mask=torch.ones(bs, 1, dtype=torch.bool))
                for _ in range(seq - 1):
                    logits, st = model(x[:, -1:], state=st, valid_mask=torch.ones(bs, 1, dtype=torch.bool))
            gen_t = _time_fn(gen, warmup=2, repeats=5)
            # 峰值内存
            torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
            mem_peak_mb = None
            try:
                import resource
                before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                model(x, valid_mask=torch.ones(bs, seq, dtype=torch.bool))
                after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                mem_peak_mb = round((after - before) / 1024.0, 2)
            except Exception:
                mem_peak_mb = None

            rows.append({
                "batch_size": bs,
                "seq_len": seq,
                "forward_median_sec": round(fwd["median_sec"], 6),
                "forward_p95_sec": round(fwd["p95_sec"], 6),
                "gen_median_sec": round(gen_t["median_sec"], 6),
                "gen_tokens_per_sec": round(bs * seq / gen_t["median_sec"], 1) if gen_t["median_sec"] > 0 else None,
                "weight_bytes": nbytes,
                "weight_mb": round(nbytes / 1048576.0, 4),
                "state_elements": prof["state_dim"],
                "state_bytes": prof["state_dim"] * 4,
                "mem_peak_mb": mem_peak_mb,
            })
    return rows


# ---------------------------------------------------------------------------
# 动态路径统计（仅 M3-B）
# ---------------------------------------------------------------------------

def measure_dynamic_paths(model, tokens_fn, bs=8, seq=32):
    """采集 M3-B 各路径执行次数与稠密 vs 稀疏 MAC。"""
    x = tokens_fn(bs, seq)
    _, stats = model(x, valid_mask=torch.ones(bs, seq, dtype=torch.bool),
                     collect_stats=True, hard_gate=True)
    total_tokens = bs * seq
    counts = stats["path_counts"]
    executed = stats["executed_macs"]
    # 稠密对照（软门全执行 = C2 路径成本，不含 controller，与 executed_macs 口径一致）
    d = model.cfg["d"]
    r = model.cfg["r"]
    hm = model.cfg["hidden_mult"]
    dense_mac = d + 2 * d * r + 2 * d * d * hm   # = C2 path
    dense_total = dense_mac * total_tokens
    return {
        "total_tokens": total_tokens,
        "path_counts": counts,
        "path_utilization": {str(k): round(v / total_tokens, 4) for k, v in counts.items()},
        "executed_macs": executed,
        "dense_macs": dense_total,
        "skipped_mac_ratio": round(1.0 - executed / dense_total, 4) if dense_total > 0 else 0.0,
        "controller_macs": stats["controller_macs"],
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    torch.set_num_threads(THREADS)
    V = VOCAB

    builders = {
        "R1_gru_reference": (GRUReferenceM3, "R1 GRU 参考"),
        "R2_tiny_transformer": (TinyTransformerM3, "R2 微型因果 Transformer"),
        "M3-A_low_rank_neural": (LowRankNeuralM3, "M3-A 低秩神经计算"),
        "M3-B_adaptive_computation": (AdaptiveComputationM3, "M3-B 自适应最小计算"),
    }

    seq_lens = [16, 32, 64, 128]
    batch_sizes = [1, 8, 16]

    results = {"threads": THREADS, "vocab": V, "models": {},
               "energy_note": "NOT_AVAILABLE（无可信硬件功耗测量，禁止 TDP*时间伪造）"}

    for key, (cls, label) in builders.items():
        torch.manual_seed(0)
        model = cls(V)
        cfg = model.cfg if hasattr(model, "cfg") else {}
        prof = model_mac_profile(key, V, **cfg)
        prof["label"] = label
        prof["params"] = sum(p.numel() for p in model.parameters())
        prof["weight_mb"] = round(sum(p.numel() * p.element_size() for p in model.parameters()) / 1048576.0, 4)
        prof["state_elements"] = model.state_elements()

        def tokens_fn(bs, seq, _m=model):
            return torch.randint(4, V, (bs, seq))

        rows = measure_model(key, model, tokens_fn, V, seq_lens, batch_sizes)
        prof["measurements"] = rows

        if key == "M3-B_adaptive_computation":
            prof["dynamic_paths"] = measure_dynamic_paths(model, tokens_fn)

        results["models"][key] = prof
        print(f"[compute_accounting] {key}: params={prof['params']} "
              f"per_token_mac={prof['per_token_forward_mac']} measurements={len(rows)}")

    out = HERE / "M3_COMPUTE_ACCOUNTING.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[compute_accounting] 写出 {out}")


if __name__ == "__main__":
    main()
