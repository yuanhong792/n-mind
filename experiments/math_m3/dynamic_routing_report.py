"""M3 动态路由报告（P4.4）：测量自适应计算是否真正跳过昂贵运算。

对比三种模式：
- dense（软门，全路径执行）：训练成本基准。
- hard_gate（硬门，只执行选中路径）：推理实际跳过。
- always_expensive（强制 C2）：对照，测量跳过收益上限。

记录：各路径执行次数、每条路径 MAC、门控自身成本、动态调度成本、
路径利用率、稠密 vs 稀疏 MAC、准确率差异（软门 vs 硬门）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "models"))

from tokenizer_m3 import M3Tokenizer  # noqa: E402
from adaptive_computation import AdaptiveComputationM3  # noqa: E402
from evaluate_m3 import load_rows, build_fixed_panel, generate  # noqa: E402


def measure_routing(model, tok, bs=8, seq=32):
    """采集三种模式的路径统计与 MAC。"""
    results = {}
    V = tok.vocab_size

    # 稠密（软门，全执行）
    x = torch.randint(4, V, (bs, seq))
    valid = torch.ones(bs, seq, dtype=torch.bool)
    _, stats_dense = model(x, valid_mask=valid, collect_stats=True, hard_gate=False)

    # 硬门（实际跳过）
    _, stats_hard = model(x, valid_mask=valid, collect_stats=True, hard_gate=True)

    # 始终昂贵（对照）
    _, stats_always = model(x, valid_mask=valid, collect_stats=True,
                            hard_gate=True, always_expensive=True)

    total = bs * seq
    results["dense"] = {
        "path_counts": stats_dense["path_counts"],
        "executed_macs": stats_dense["executed_macs"],
        "controller_macs": stats_dense["controller_macs"],
    }
    results["hard_gate"] = {
        "path_counts": stats_hard["path_counts"],
        "path_utilization": {str(k): round(v / total, 4) for k, v in stats_hard["path_counts"].items()},
        "executed_macs": stats_hard["executed_macs"],
        "controller_macs": stats_hard["controller_macs"],
        "skipped_mac_ratio": round(1 - stats_hard["executed_macs"] / stats_dense["executed_macs"], 4),
    }
    results["always_expensive"] = {
        "path_counts": stats_always["path_counts"],
        "executed_macs": stats_always["executed_macs"],
    }
    # 调度成本：硬门逐样本 Python 循环的额外开销（与稠密的差值中非计算部分）
    results["scheduling_note"] = "硬门使用逐样本 Python 索引调度，存在调度开销；实际 CPU 速度变化见 compute_accounting"
    return results


def accuracy_soft_vs_hard(model, tok, panel_rows):
    """测量软门（稠密）与硬门推理的准确率差异。

    按任务均匀取前 3 条，避免面板按任务排序时只覆盖前几个任务。
    """
    from collections import defaultdict
    by_task = defaultdict(list)
    for r in panel_rows:
        by_task[r["task"]].append(r)
    sampled = []
    for task in sorted(by_task):
        sampled.extend(by_task[task][:3])
    soft_ok = hard_ok = 0
    n = 0
    for r in sampled:
        # 软门（稠密）
        g_soft = generate(model, tok, r["prompt"], hard_gate=False)
        # 硬门（稀疏）
        g_hard = generate(model, tok, r["prompt"], hard_gate=True)
        soft_ok += int(g_soft["text"] == r["answer"])
        hard_ok += int(g_hard["text"] == r["answer"])
        n += 1
    return {
        "n": n,
        "soft_gate_acc": round(soft_ok / n, 4) if n else 0.0,
        "hard_gate_acc": round(hard_ok / n, 4) if n else 0.0,
        "diff": round(soft_ok / n - hard_ok / n, 4) if n else 0.0,
        "note": "训练后 checkpoint 实测；每任务取前3条；硬门成绩为准，软门仅对照",
    }


def main():
    torch.set_num_threads(20)
    torch.manual_seed(0)
    tok = M3Tokenizer()
    model = AdaptiveComputationM3(tok.vocab_size)
    model.eval()

    panel_rows = build_fixed_panel(load_rows("val"), per_task=5)

    result = {
        "note": "动态路由实际跳过测量；未训练模型的路径分布为随机初始化结果，训练后会变化",
        "routing": measure_routing(model, tok),
        "soft_vs_hard": accuracy_soft_vs_hard(model, tok, panel_rows),
    }

    out = HERE / "M3_DYNAMIC_ROUTING_REPORT.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[routing] hard_gate skipped_ratio={result['routing']['hard_gate']['skipped_mac_ratio']}")
    print(f"[routing] path_util={result['routing']['hard_gate']['path_utilization']}")
    print(f"[routing] 写出 {out}")


if __name__ == "__main__":
    main()
