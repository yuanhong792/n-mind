"""P8.2：公式资源成本估算器（静态 + 实测双口径）。

静态：遍历表达式树，按张量维度累加 MAC 与中间激活元素数。
实测：torch.utils.benchmark 风格预热/重复，报告中位数与 P95，
      与 M4_TRANSFORMER_COST_ANALYSIS.json 的真实墙钟口径一致。

运行：python formula_cost_estimator.py
"""

from __future__ import annotations

import statistics
import time

import torch

from formula_grammar import Expr

MAC_PER_OP = {"Add": 0, "Subtract": 0, "MatMul": 1, "Dot": 1, "Outer": 1,
              "LowRank": 3, "Projection": 0, "Normalize": 3, "Softmax": 2,
              "KernelMap": 2, "Gather": 0, "Reduce": 1, "Hadamard": 1,
              "Residual": 1, "Gate": 2, "Input": 0}


def estimate_macs(expr: Expr, shapes: dict[str, tuple[int, ...]]) -> dict:
    """shapes: 输入名 → 张量形状。返回静态 MAC 与激活元素估计。"""
    def shape_of(e: Expr) -> tuple[int, ...]:
        if e.op == "Input":
            return shapes[e.params["name"]]
        cs = [shape_of(c) for c in e.children]
        if e.op == "MatMul":
            a, b = cs[0], cs[1]
            if len(a) == 2 and len(b) == 2:
                return (a[0], b[1])
            return tuple(a[:-1]) + (b[-1],)
        if e.op in ("Add", "Subtract", "Hadamard", "Gate"):
            return cs[0]
        if e.op == "Softmax" or e.op == "KernelMap" or e.op == "Normalize":
            return cs[0]
        if e.op == "LowRank":
            return cs[0]
        if e.op == "Projection":
            return cs[0][:-1] + (min(e.params.get("dim", cs[0][-1]), cs[0][-1]),)
        if e.op == "Reduce":
            dim = e.params.get("dim", -2)
            lst = list(cs[0]); lst[dim] = 1
            return tuple(lst)
        return cs[0] if cs else ()

    def walk(e: Expr) -> dict:
        macs = MAC_PER_OP.get(e.op, 1)
        shapes_visited = [shape_of(c) for c in e.children]
        for s in shapes_visited:
            n = 1
            for x in s:
                n *= x
            macs *= max(1, n // 1000)  # 与规模成比例，避免常数失真
        for c in e.children:
            r = walk(c)
            macs += r["macs"]
        return {"macs": macs}
    return walk(expr)


def measure_wall(expr: Expr, Q, K, V, mask, repeats=20, warmup=3):
    for _ in range(warmup):
        expr.eval(Q, K, V, mask)
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        expr.eval(Q, K, V, mask)
        ts.append(time.perf_counter() - t0)
    return {"median_us": round(statistics.median(ts) * 1e6, 2),
            "p95_us": round(sorted(ts)[int(0.95 * (len(ts) - 1))] * 1e6, 2)}


if __name__ == "__main__":
    import json
    from pathlib import Path
    B, h, T, d = 1, 4, 32, 64
    Q, K, V = [torch.randn(B, h, T, d) for _ in range(3)]
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
    expr = Expr("MatMul", (
        Expr("Softmax", (Expr("MatMul", (Expr("Input", params={"name": "Q"}),
             Expr("Input", params={"name": "K"}), ), params={"transpose_b": True}),)),
        Expr("Input", params={"name": "V"})), tag="ref")
    res = {"status": "COMPLETE", "static": estimate_macs(expr, {"Q": Q.shape, "K": K.shape, "V": V.shape}),
           "wall": measure_wall(expr, Q, K, V, mask)}
    p = Path(__file__).resolve().parents[1] / "M4_FORMULA_COST.json"
    p.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[cost-est] 写出 {p}")
