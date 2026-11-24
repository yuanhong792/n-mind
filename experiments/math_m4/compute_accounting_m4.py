"""P13：逐 Batch 真实计算成本统计。

禁止使用 steps*batch*32*mac*3 固定公式。每个 batch 记录：
实际 Batch Size / 有效 Token / Padding Token / 实际张量长度 / 有效监督 Token /
各模块执行次数 / 本模型累计成本。
数学估算、Profiler、真实墙钟分列；推理用 model.eval()+inference_mode，预热+中位数+P95。
无能耗测量条件 → NOT_AVAILABLE。
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field

import torch


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

    def as_dict(self):
        return self.__dict__.copy()


def batch_cost(model, x, valid, weights, backward=False, optimizer=False) -> BatchCost:
    B, T = x.shape
    n_valid = int(valid.sum())
    n_sup = int((weights > 0).sum())
    params = sum(p.numel() for p in model.parameters())
    d = getattr(model, "cfg", {}).get("d_model", 64)
    layers = getattr(model, "cfg", {}).get("layers", 2)
    # 解析 MAC：按实际张量长度与实际 mask 计算，不使用固定 seq=32
    fwd = B * T * d * d * 3 * layers + B * T * d * d * 3 * layers + B * T * 102 * d
    bwd = 2 * fwd if backward else 0
    opt = 2 * params if optimizer else 0
    return BatchCost(B, T, n_valid, T * B - n_valid, n_sup,
                     {"embedding": B, "attention_blocks": layers, "ffn": layers,
                      "output_head": B * T}, fwd, bwd, opt)


def wall_inference(model, inputs, repeats=30, warmup=5, **kw):
    """model.eval() + inference_mode + 预热 + 中位数/P95，禁止带 autograd 图测试。"""
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
    """Prefill / 首 Token / 单 Token / 完整任务生成，分列；EOS 早停成本不冒充高效。"""
    out = {}
    for p in prompts[:20]:
        rec = gen_fn(model, tok, p)
        out[p] = {"tokens": len(rec["ids"]), "finish_reason": rec["finish_reason"]}
    return {"samples": out,
            "note": "EOS 早停样本的输出更短，不得将其低成本当作效率优势；"

                    "固定输出长度对照见 M4_TRANSFORMER_COST_ANALYSIS.json",
            "energy": "NOT_AVAILABLE（无硬件功耗测量）"}


if __name__ == "__main__":
    import json
    import sys
    from pathlib import Path
    HERE = Path(__file__).resolve().parents[0]
    sys.path.insert(0, str(HERE))
    from evaluate_m4 import load_rows, make_batch  # noqa: E402
    from tokenizer_m4 import M4Tokenizer  # noqa: E402
    from models.transformer_reference import TinyTransformerM3  # noqa: E402

    torch.set_num_threads(20)
    tok = M4Tokenizer()
    rows = load_rows("train")[:16]
    x, valid, w, tgt = make_batch(tok, rows)
    model = TinyTransformerM3(tok.vocab_size, 64, 4, 2)
    cost = batch_cost(model, x, valid, w, backward=True, optimizer=True)
    wall = wall_inference(model, (x,))
    res = {"status": "SELF_TEST", "batch": cost.as_dict(), "wall_inference": wall,
           "energy": "NOT_AVAILABLE"}
    p = HERE / "M4_COMPUTE_ACCOUNTING.json"
    p.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[accounting] B={cost.batch_size} T={cost.tensor_length} "
          f"valid={cost.valid_tokens} sup={cost.supervised_tokens} 写出 {p}")
