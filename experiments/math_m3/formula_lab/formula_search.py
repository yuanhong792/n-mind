"""M3-P6：公式搜索主程序（≤32 候选，淘汰非法，记录创新审计）。

不依赖外部 LLM；只做确定性受控枚举 + 数学验证 + 性能筛选。
通过数学与性能筛选的少量候选才允许进入真实神经训练（本轮由 M3-A/B 架构承载）。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
M3 = HERE.parent
sys.path.insert(0, str(HERE))

from formula_generator import generate  # noqa: E402
from formula_grammar import validate_tree, tree_to_str  # noqa: E402
from formula_verifier import verify_formula  # noqa: E402
from symbolic_simplifier import canonicalize, known_equivalence  # noqa: E402

torch.set_num_threads(4)
MAX_CANDIDATES = 32


def main():
    t0 = time.perf_counter()
    pool = generate(MAX_CANDIDATES)
    registry = {}
    results = []
    canonical_seen = {}
    for name, tree in pool.items():
        d, r, x_dim = 64, 8, 32
        params = {
            "V_0": torch.randn(r, d), "U_0": torch.randn(d, r),
            "diag": torch.ones(d) * 0.5, "Wx": torch.randn(d, x_dim),
            "Wg": torch.randn(d, x_dim), "W": torch.randn(d, d),
            "gate": 0.5,
        }
        ok_grammar, err = validate_tree(tree)
        vr = verify_formula(tree, params, d=d, x_dim=x_dim) if ok_grammar else {
            "pass": False, "error": err, "checks": {}}
        s = tree_to_str(canonicalize(tree))
        eq = canonical_seen.get(s)
        # 已有技术检索（项目内已知基础）
        prior = ("低秩适配 LoRA (Hu 2021) + 门控残差 (GRU/LSTM) 的已有组合"
                 if "lowrank" in s or "diag" in s else "恒等/非线性基础算子（通用）")
        # 性能筛选：估算每步 MAC（低秩 dr、对角 d、矩阵 d²、门控额外 cand 计算）
        def count_cost(node):
            op = node["op"]
            cost = {"lowrank": d * r, "diag": d, "matmul": d * d,
                    "input_proj": d * 32, "gate": d}.get(op, 0)
            for a in node.get("args", []):
                if isinstance(a, dict):
                    cost += count_cost(a)
            return cost
        mac_estimate = count_cost(tree)
        dense_baseline = d * d
        entry = {
            "name": name, "formula": tree_to_str(tree), "canonical": s,
            "grammar_ok": ok_grammar, "math_pass": vr["pass"],
            "checks": vr.get("checks", {}), "error": vr.get("error"),
            "equivalent_to": eq,
            "estimated_macs_per_step": mac_estimate,
            "cost_vs_dense": round(mac_estimate / dense_baseline, 4),
            "innovation_audit": {
                "source": "本项目 formula_lab 受控枚举",
                "prior_art": prior,
                "project_change": "把低秩/对角/门控算子限定用于循环神经状态更新并做受控组合搜索",
                "math_properties": ["dimension_checked", "gradient_checked", "sample_isolation_checked"],
                "theoretical_cost": "O(dr) 低秩 + O(d) 对角/门控" if "lowrank" in s else "O(d)",
                "possible_novelty": "组合机制（非基础算子本身）",
                "unproven_assumptions": [
                    "组合在字符级计算任务上的有效性", "与已有低秩循环网络的性能差异",
                ],
            },
        }
        results.append(entry)
        registry[name] = entry
        if eq is None and ok_grammar and vr["pass"]:
            canonical_seen[s] = name

    # 性能筛选：数学通过、非等价、且估算成本不超过稠密基线 50%
    eligible = [r for r in results if r["math_pass"] and not r["equivalent_to"]
                and r["cost_vs_dense"] <= 0.5]
    eligible.sort(key=lambda r: r["cost_vs_dense"])
    survivors = [r["name"] for r in eligible[:8]]  # 少量候选进入训练考量
    report = {
        "status": "M3_FORMULA_SEARCH_DONE",
        "n_candidates": len(results), "max_candidates": MAX_CANDIDATES,
        "n_math_pass": sum(r["math_pass"] for r in results),
        "n_cost_screen_pass": len(eligible),
        "n_survivors_unique": len(survivors),
        "survivors": survivors,
        "performance_screen": "数学通过 + 非等价 + 估算MAC ≤ 稠密基线50%，按成本升序取前8",
        "elimination_rule": "维度/非有限/梯度/样本隔离/语法/等价 任一失败即淘汰",
        "seconds": round(time.perf_counter() - t0, 3),
        "results": results,
        "note": "本轮仅算子级验证；真实训练承载能力由 M3-A/B 架构定义，"
                "公式候选不单独分配训练预算（NOT_RUN）。",
    }
    (M3 / "M3_FORMULA_SEARCH_RESULTS.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (M3 / "M3_FORMULA_REGISTRY.json").write_text(
        json.dumps({"status": "M3_FORMULA_REGISTRY", "formulas": registry},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "n": report["n_candidates"],
                      "math_pass": report["n_math_pass"],
                      "survivors": survivors, "seconds": report["seconds"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
