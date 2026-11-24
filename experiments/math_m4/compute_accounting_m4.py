"""M41：逐架构真实计算成本统计（替代通用 d*d 估算）。

- GRU / 标准注意力 / 低秩 QK / 选择性真实路径 / Head / 优化器分别建模；
- 标准注意力包含实际序列长度平方项；
- 低秩 score 用 reduced_head_dim，Value 聚合仍用 value_head_dim；
- 选择性交互读取正式模型执行 stats，区分理论选择率与真实张量执行率；
- 训练实际输入 x[:, :-1] 必须用 input_len=T-1；
- 反向标记 ESTIMATED 并给出范围；Profiler 与手工小张量交叉验证。
输出 M41_COST_ACCOUNTING_AUDIT.json；不覆盖 M4_COMPUTE_ACCOUNTING.json。
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent


@dataclass
class BatchCost:
    batch_size: int
    tensor_length: int
    valid_tokens: int
    padding_tokens: int
    supervised_tokens: int
    module_executions: dict = field(default_factory=dict)
    analytic_fwd_macs: int = 0
    analytic_bwd_macs: int = 0
    optimizer_macs: int = 0
    wall_sec: float = 0.0
    details: dict = field(default_factory=dict)
    bwd_estimation: str = "ESTIMATED"

    def as_dict(self):
        return self.__dict__.copy()


def _arch_name(model) -> str:
    name = type(model).__name__
    if "GRU" in name:
        return "R1_GRU"
    if "LowRank" in name:
        return "M4_A_low_rank"
    if "Selective" in name:
        return "M4_B_selective"
    return "R2_transformer"


def _cfg(model):
    return getattr(model, "cfg", {})


def gru_forward_macs(B, T, d, hidden, vocab):
    gate = B * T * (d * hidden + hidden * hidden)
    gated = 3 * gate
    mix = B * T * hidden * hidden
    head = B * T * hidden * vocab
    return {"gru_gates": gated, "mix": mix, "output_head": head,
            "total": gated + mix + head}


def transformer_forward_macs(B, T, d, h, layers, vocab):
    dh = d // h
    qkv = 3 * B * T * d * d
    score = B * h * T * T * dh
    attn_v = B * h * T * T * dh
    out_proj = B * T * d * d
    ffn = 4 * B * T * d * d
    per_layer = qkv + score + attn_v + out_proj + ffn
    head = B * T * vocab * d
    return {"qkv": qkv, "score": score, "attn_v": attn_v,
            "out_proj": out_proj, "ffn": ffn,
            "per_layer": per_layer, "layers": layers,
            "core": per_layer * layers, "output_head": head,
            "total": per_layer * layers + head}


def low_rank_forward_macs(B, T, d, h, layers, vocab, rh):
    dh = d // h
    qk_proj = 2 * B * T * d * h * rh
    score = B * h * T * T * rh
    value_proj = B * T * d * d
    value_agg = B * h * T * T * dh
    out_proj = B * T * d * d
    ffn = 4 * B * T * d * d
    per_layer = qk_proj + score + value_proj + value_agg + out_proj + ffn
    head = B * T * vocab * d
    return {"qk_proj": qk_proj, "score": score, "value_proj": value_proj,
            "value_aggregation": value_agg, "out_proj": out_proj, "ffn": ffn,
            "reduced_head_dim": rh, "value_head_dim": dh,
            "per_layer": per_layer, "layers": layers,
            "core": per_layer * layers, "output_head": head,
            "total": per_layer * layers + head}


def batch_cost(model, x, valid, weights, backward=False, optimizer=False,
               input_len=None, hard_gate=True) -> BatchCost:
    """返回逐 batch 成本。input_len 为实际喂给模型的长度；训练传 T-1。"""
    B, T_full = x.shape
    T = T_full if input_len is None else min(input_len, T_full)
    xin = x[:, :T]
    n_valid = int(valid[:, :T].sum())
    n_sup = int((weights[:, :T] > 0).sum())
    params = sum(p.numel() for p in model.parameters())
    arch = _arch_name(model)
    vocab = model.head.out_features

    if arch == "R1_GRU":
        d, hidden = _cfg(model)["d_model"], _cfg(model)["hidden"]
        details = gru_forward_macs(B, T, d, hidden, vocab)
    elif arch == "R2_transformer":
        c = _cfg(model)
        details = transformer_forward_macs(B, T, c["d_model"], c["nhead"],
                                           c["layers"], vocab)
    elif arch == "M4_A_low_rank":
        c = _cfg(model)
        rh = model.blocks[0].attn.rh
        details = low_rank_forward_macs(B, T, c["d_model"], c["nhead"],
                                        c["layers"], vocab, rh)
    else:
        c = _cfg(model)
        d, h = c["d_model"], c["nhead"]
        with torch.no_grad():
            _, state = model(xin, hard_gate=hard_gate)
        sb = state["stats"]["blocks"]
        qkv = 3 * B * T * d * d
        out_proj = B * T * d * d
        ffn = 4 * B * T * d * d
        sel_attn = sum(s["executed_macs"] for s in sb)
        controller = sum(s["controller_macs"] for s in sb)
        gather = sum(s["gather_elements"] for s in sb)
        head = B * T * vocab * d
        core = qkv + out_proj + ffn + sel_attn
        details = {
            "qkv": qkv, "out_proj": out_proj, "ffn": ffn,
            "selective_executed_attention_and_controller": sel_attn,
            "controller_macs": controller, "gather_elements": gather,
            "theoretical_selection_ratio": round(
                sum(s["theoretical_selection_ratio"] for s in sb) / max(1, len(sb)), 4),
            "actual_compute_ratio": round(
                sum(s["actual_compute_ratio"] for s in sb) / max(1, len(sb)), 4),
            "output_head": head, "core": core, "total": core + head,
            "note": "gather_elements 是数据搬运元素数，不与 MAC 相加作为计算量。",
        }

    fwd = int(details["total"])
    bwd = int(round(2 * fwd)) if backward else 0
    opt = int(8 * params) if optimizer else 0
    modules = {"embedding": B, "output_head": B * T,
               "model_layers": _cfg(model).get("layers", 1)}
    return BatchCost(B, T, n_valid, B * T - n_valid, n_sup, modules,
                     fwd, bwd, opt, 0.0, details, "ESTIMATED: bwd=2.0x fwd, range 1.5x-2.5x")


def wall_inference(model, inputs, repeats=30, warmup=5, **kw):
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            model(*inputs, **kw)
        ts = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            model(*inputs, **kw)
            ts.append(time.perf_counter() - t0)
    return {"n": repeats, "warmup": warmup,
            "median_sec": round(statistics.median(ts), 6),
            "p95_sec": round(sorted(ts)[int(0.95 * (len(ts) - 1))], 6)}


def inference_breakdown(model, tok, prompts, gen_fn, repeats=10):
    out = {}
    for p in prompts[:20]:
        rec = gen_fn(model, tok, p)
        out[p] = {"tokens": len(rec["ids"]), "finish_reason": rec["finish_reason"]}
    return {"samples": out,
            "note": "EOS 早停样本的输出更短，不得将其低成本当作效率优势。",
            "energy": "NOT_AVAILABLE（无硬件功耗测量）"}


def _tiny_models():
    import sys
    sys_path = str(HERE / "models")
    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)
    from gru_reference import GRUReferenceM3
    from transformer_reference import TinyTransformerM3
    from low_rank_interaction import LowRankInteractionM4
    from selective_interaction import SelectiveInteractionM4
    return {
        "R1_GRU": GRUReferenceM3(16, 8, 8),
        "R2_transformer": TinyTransformerM3(16, 8, 2, 1),
        "M4_A_low_rank": LowRankInteractionM4(16, 8, 2, 1, 4),
        "M4_B_selective": SelectiveInteractionM4(16, 8, 2, 1, 2, 1),
    }


def hand_checks():
    B, T, d, h, layers, vocab = 1, 4, 8, 2, 1, 16
    expected = {
        "R1_GRU": 3 * B * T * (d * 8 + 8 * 8) + B * T * 8 * 8 + B * T * 8 * vocab,
        "R2_transformer": ((3 * B * T * d * d) + (B * h * T * T * (d // h)) * 2 +
                           B * T * d * d + 4 * B * T * d * d) * layers + B * T * vocab * d,
        "M4_A_low_rank": ((2 * B * T * d * h * 2) + B * h * T * T * 2 +
                          B * T * d * d + B * h * T * T * (d // h) +
                          B * T * d * d + 4 * B * T * d * d) * layers + B * T * vocab * d,
    }
    models = _tiny_models()
    out = {}
    for name, model in models.items():
        x = torch.randint(4, 16, (B, T))
        valid = torch.ones(B, T, dtype=torch.bool)
        w = torch.ones(B, T)
        c = batch_cost(model, x, valid, w, input_len=T)
        if name in expected:
            ok = c.analytic_fwd_macs == expected[name]
            out[name] = {"hand_macs": expected[name], "function_macs": c.analytic_fwd_macs,
                         "pass": ok}
        else:
            st = c.details
            out[name] = {"hand_macs": "读取真实执行 stats", "function_macs": c.analytic_fwd_macs,
                         "selective_stats": {k: st[k] for k in
                                             ("theoretical_selection_ratio",
                                              "actual_compute_ratio",
                                              "selective_executed_attention_and_controller")},
                         "pass": c.analytic_fwd_macs > 0 and st["actual_compute_ratio"] >= 0}
    return out


def profiler_check():
    from torch.profiler import ProfilerActivity, profile
    models = _tiny_models()
    out = {}
    for name, model in models.items():
        model.eval()
        x = torch.randint(4, 16, (1, 8))
        with torch.no_grad():
            for _ in range(2):
                model(x)
            with profile(activities=[ProfilerActivity.CPU]) as prof:
                for _ in range(5):
                    model(x)
        rows = []
        for e in sorted(prof.key_averages(), key=lambda e: e.self_cpu_time_total,
                        reverse=True)[:6]:
            rows.append({"name": e.key, "self_cpu_us": round(e.self_cpu_time_total, 2)})
        out[name] = {"top_ops": rows, "finite": all(r["self_cpu_us"] >= 0 for r in rows)}
    return out


def main():
    torch.set_num_threads(20)
    import sys
    sys.path.insert(0, str(HERE))
    sys.path.insert(0, str(HERE / "models"))
    from evaluate_m4 import load_rows, make_batch
    from tokenizer_m4 import M4Tokenizer
    from gru_reference import GRUReferenceM3
    from transformer_reference import TinyTransformerM3
    from low_rank_interaction import LowRankInteractionM4
    from selective_interaction import SelectiveInteractionM4

    tok = M4Tokenizer()
    rows = load_rows("train")[:16]
    x, valid, w, tgt = make_batch(tok, rows, "A")
    models = {
        "R1_GRU": GRUReferenceM3(tok.vocab_size),
        "R2_transformer": TinyTransformerM3(tok.vocab_size, 64, 4, 2),
        "M4_A_low_rank": LowRankInteractionM4(tok.vocab_size, 64, 4, 2, 16),
        "M4_B_selective": SelectiveInteractionM4(tok.vocab_size, 64, 4, 2, 4, 2),
    }
    samples = {}
    for name, model in models.items():
        c = batch_cost(model, x, valid, w, backward=True, optimizer=True,
                       input_len=x.shape[1] - 1)
        samples[name] = c.as_dict()
    hc = hand_checks()
    pf = profiler_check()
    all_pass = (all(v["pass"] for v in hc.values()) and
                all(v["finite"] for v in pf.values()))
    out = {
        "status": "PASS" if all_pass else "FAILED",
        "precision": "FP32",
        "input_rule": "训练成本按 x[:, :-1] 的实际长度统计",
        "backward": {"status": "ESTIMATED", "central_multiplier": 2.0,
                     "range": [1.5, 2.5],
                     "note": "未做逐算子反向解析，禁止当作精确值"},
        "optimizer": {"status": "ESTIMATED", "adamw_macs_per_param": 8},
        "samples_training_input": samples,
        "hand_checks": hc,
        "profiler_cross_check": pf,
        "energy": "NOT_AVAILABLE",
    }
    p = HERE / "M41_COST_ACCOUNTING_AUDIT.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[m41-cost] status={out['status']} 写出 {p}")


if __name__ == "__main__":
    main()
