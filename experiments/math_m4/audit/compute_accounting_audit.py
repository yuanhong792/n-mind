"""M3.1 P0：计算成本统计审计（零梯度）。

任务书禁止继续使用：
    total_mac = steps * batch_size * 32 * per_token_mac * 3
冒充实际累计计算量。

本脚本读取 M3 原始台账与训练结果，量化：
  - 原始记账按全局累计步数导致的虚增倍数。
  - 固定 seq=32 与实际 batch 内有效 Token 数的差异。
  - padding 比例（M3 已有 valid_mask，但 MAC 公式未扣 padding）。
  - 数学估算 / profiler / 真实墙钟三类口径的分隔。

运行：python compute_accounting_audit.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
M3 = HERE.parent / "math_m3"
sys.path.insert(0, str(M3))
from evaluate_m3 import load_rows, make_batch  # noqa: E402
from tokenizer_m3 import M3Tokenizer  # noqa: E402

BATCH_SIZE = 32
SEQ = 32


def main():
    tok = M3Tokenizer()
    led = [json.loads(l) for l in (M3 / "M3_BUDGET_LEDGER.jsonl")
           .read_text(encoding="utf-8").splitlines() if l.strip()]
    res = json.loads((M3 / "M3_TRAINING_RESULTS.json").read_text(encoding="utf-8"))
    acc = json.loads((M3 / "M3_COMPUTE_ACCOUNTING.json").read_text(encoding="utf-8"))

    step_events = [e for e in led if e.get("type") == "step"]
    global_steps = max((e.get("global_step", 0) for e in step_events), default=0)
    # 每架构每 seed 各 512 步，全局步数含此前架构；虚增倍数 = 全局步数 / 单模型步数
    inflation = {name: round(v["curve"][-1]["step"] / 512, 4)
                 for name, v in res.get("models", {}).items() if v.get("curve")}

    # 实际 batch 的 Token 统计（抽样 32 个真实 batch）
    rows = load_rows("train")
    import random
    rng = random.Random(20261124)
    stats = []
    for _ in range(32):
        b = rng.sample(rows, min(BATCH_SIZE, len(rows)))
        x, valid, w, _ = make_batch(tok, b)
        n_valid = int(valid.sum())
        n_pad = int(x.numel() - n_valid)
        n_supervised = int((w[:, 1:] > 0).sum())
        stats.append({"T": x.shape[1], "n_valid": n_valid, "n_pad": n_pad,
                      "n_supervised": n_supervised, "valid_ratio": round(n_valid / x.numel(), 4)})
    avg = {k: round(sum(s[k] for s in stats) / len(stats), 2) for k in ("T", "n_valid", "n_pad", "n_supervised", "valid_ratio")}

    out = {
        "status": "COMPLETE", "forbidden_formula": "steps * batch * 32 * per_token_mac * 3",
        "ledger": {"step_events": len(step_events), "max_global_step": global_steps,
                   "run_end_events": sum(1 for e in led if e.get("type") == "run_end")},
        "inflation_by_global_step": inflation,
        "note_inflation": "M3 原始 curve 用全局累计步数乘成本；此处量化相对单模型 512 步的虚增倍数",
        "actual_batch_sample": {"n_batches": len(stats), "avg": avg,
                                "max_T": max(s["T"] for s in stats),
                                "min_valid_ratio": min(s["valid_ratio"] for s in stats)},
        "accounting_separation": {
            "analytic_mac": "解析 MAC：来自 M3_COMPUTE_ACCOUNTING.json，固定 seq 参考点",
            "profiler": "Profiler 统计：M3_RESOURCE_BENCHMARK.json，实测算子热点",
            "wall_clock": "真实墙钟：M3_BUDGET_LEDGER.jsonl，含调度与 I/O",
            "rule": "三者不得互相替代；M4 必须逐 batch 记录实际有效 Token 与监督 Token",
        },
        "m4_requirement": ["实际 Batch Size", "有效 Token 数", "Padding Token 数",
                           "实际张量计算长度", "有效监督 Token 数", "各模块执行次数",
                           "本模型累计成本", "数学估算/Profiler/墙钟分列"],
    }
    p = HERE / "M31_COMPUTE_ACCOUNTING_AUDIT.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[cost] ledger steps={len(step_events)} global_max={global_steps}")
    print(f"[cost] inflation={inflation}")
    print(f"[cost] avg batch T={avg['T']} valid_ratio={avg['valid_ratio']} supervised={avg['n_supervised']}")
    print(f"[cost] 写出 {p}")


if __name__ == "__main__":
    main()
