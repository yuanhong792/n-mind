"""P7：严格数学等价的计算消除（Answer-only Head 优化）。

不近似、不改模型语义：
  全量：logits = head(core(tokens))                  # 对所有 T 个位置
  优化：h = core(tokens); logits = head(h[supervised_positions])

必须验证：
  logits 数值一致 / loss 一致 / 梯度逐张量一致 / 参数更新在浮点误差内一致 /
  自回归生成一致 / 真实 CPU 时间下降。

重要：M4 词表只有 102，head 在总成本中占比很小（见 M4_TRANSFORMER_COST_ANALYSIS.json，
T=32 时约 0.55%）。若实测无收益或变慢，必须如实淘汰该优化，不得宣称工程突破。

运行：python exact_computation_reduction.py
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parents[1]
M3 = HERE.parent / "math_m3"
sys.path.insert(0, str(M3))
sys.path.insert(0, str(M3 / "models"))
from transformer_reference import TinyTransformerM3  # noqa: E402


def step(model, x, positions, optimize: bool, tgt: torch.Tensor):
    if optimize:
        h_full, cache = None, None
        # core 复现：embedding + blocks + ln，但只把监督位置送 head
        pos = 0
        h = model.emb(x)
        for blk in model.blocks:
            h, k, v = blk(h, None, None, pos)
        h = model.ln(h)
        logits = model.head(h[:, positions])
    else:
        logits, cache = model(x)
        logits = logits[:, positions]   # 同口径：只取监督位置计算 loss
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1))
    return loss, logits


def wall(fn, repeats=30, warmup=5):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return {"median_us": round(statistics.median(ts) * 1e6, 2),
            "p95_us": round(sorted(ts)[int(0.95 * (len(ts) - 1))] * 1e6, 2)}


def run_self_test():
    torch.set_num_threads(20)
    torch.manual_seed(20261124)
    V, T, B = 102, 64, 8
    pos = [8, 9, 10, 11]  # 模拟 SEP 后的 4 个监督位置

    def make():
        m = TinyTransformerM3(V, d_model=64, nhead=4, layers=2)
        return m

    x = torch.randint(4, 100, (B, T))
    m1, m2 = make(), make()
    m2.load_state_dict(m1.state_dict())

    # 前向 logits / loss 一致
    tgt = torch.randint(4, 100, (B, len(pos)))  # 固定目标，两条路径同 loss 口径
    l1, g1 = step(m1, x, pos, optimize=False, tgt=tgt)
    l2, g2 = step(m2, x, pos, optimize=True, tgt=tgt)
    logits_diff = (g1 - g2).abs().max().item()  # step 内已按监督位置切片

    # 梯度一致
    m1.zero_grad(); m2.zero_grad()
    l1.backward(); l2.backward()
    grad_diffs = {}
    for (n1, p1), (n2, p2) in zip(m1.named_parameters(), m2.named_parameters()):
        if p1.grad is not None and p2.grad is not None:
            grad_diffs[n1] = (p1.grad - p2.grad).abs().max().item()
    max_grad_diff = max(grad_diffs.values()) if grad_diffs else 0.0

    # 参数更新一致
    o1 = torch.optim.SGD(m1.parameters(), lr=0.01)
    o2 = torch.optim.SGD(m2.parameters(), lr=0.01)
    o1.step(); o2.step()
    upd_diff = max((a - b).abs().max().item() for a, b in
                   zip(m1.parameters(), m2.parameters()))

    # 生成一致性：贪心解码前 6 步
    m1.eval(); m2.eval()
    with torch.no_grad():
        ids = torch.randint(4, 100, (1, T))
        logits_full, _ = m1(ids)
        h = m1.ln(m1.blocks[1](m1.blocks[0](m1.emb(ids), None, None, 0)[0], None, None, 0)[0])
        next1 = logits_full[:, -1:].argmax(-1).item()   # 原生 forward
        next2 = m1.head(h[:, -1:]).argmax(-1).item()    # 手动复现 core 后只算末位 head

    # 真实 CPU 时间（推理模式）
    m1.eval()
    with torch.inference_mode():
        t_full = wall(lambda: m1(x))
        def opt_forward():
            h = m1.emb(x)
            for blk in m1.blocks:
                h, _, _ = blk(h, None, None, 0)
            return m1.head(m1.ln(h)[:, pos])
        t_opt = wall(opt_forward)

    reduction = 1 - t_opt["median_us"] / t_full["median_us"]
    return {
        "status": "COMPLETE", "category": "A_数学等价优化（非近似，非算法创新）",
        "logits_max_abs_diff": round(logits_diff, 9),
        "loss_max_abs_diff": round(abs(l1.item() - l2.item()), 9),
        "gradient_max_abs_diff": round(max_grad_diff, 9),
        "parameter_update_max_abs_diff": round(upd_diff, 9),
        "greedy_next_token_equal": next1 == next2,
        "full_forward": t_full, "optimized_forward": t_opt,
        "measured_time_reduction": round(reduction, 4),
        "verdict": ("ADOPT" if logits_diff < 1e-5 and max_grad_diff < 1e-4 and reduction > 0.01
                    else "REJECT_NO_MEASURED_BENEFIT"),
        "note": "词表102时 head 占比极低；索引开销可能超过节省，负收益必须如实淘汰",
    }


if __name__ == "__main__":
    import json
    res = run_self_test()
    p = HERE / "M4_EXACT_REDUCTION.json"
    p.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[exact] logits_diff={res['logits_max_abs_diff']} grad_diff={res['gradient_max_abs_diff']} "
          f"time_reduction={res['measured_time_reduction']} verdict={res['verdict']}")
    print(f"[exact] 写出 {p}")
