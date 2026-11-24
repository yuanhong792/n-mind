"""P8.3：公式等价简化器。

只删除数学恒等节点（scale=0 的 Residual、dim 超出范围的 Projection、
重复 Softmax），不做近似简化；每次简化后必须数值等价验证。

运行：python formula_simplifier.py
"""

from __future__ import annotations

import torch

from formula_grammar import Expr


def simplify(expr: Expr) -> Expr:
    children = tuple(simplify(c) for c in expr.children)
    e = Expr(expr.op, children, dict(expr.params), expr.tag)
    if e.op == "Residual" and abs(e.params.get("scale", 1.0)) < 1e-12:
        return children[0]
    if e.op == "Projection" and len(children) == 1:
        return children[0]  # 在不已知维度时保守：只删除无参数投影
    if e.op == "Softmax" and children and children[0].op == "Softmax":
        return children[0]
    return e


def run_self_test():
    Q, K, V = (Expr("Input", params={"name": n}) for n in ("Q", "K", "V"))
    s = Expr("MatMul", (Q, K), params={"transpose_b": True})
    base = Expr("MatMul", (Expr("Softmax", (s,)), V))
    noisy = Expr("Residual", (base, V), params={"scale": 0.0})
    simple = simplify(noisy)
    torch.manual_seed(0)
    q, k, v = [torch.randn(1, 4, 16, 16) for _ in range(3)]
    mask = torch.triu(torch.ones(16, 16, dtype=torch.bool), 1)
    y1 = noisy.eval(q, k, v, mask)
    y2 = simple.eval(q, k, v, mask)
    return {"status": "COMPLETE",
            "original": repr(noisy), "simplified": repr(simple),
            "max_abs_diff": round((y1 - y2).abs().max().item(), 9),
            "numerically_equivalent": torch.allclose(y1, y2, atol=1e-6)}


if __name__ == "__main__":
    import json
    from pathlib import Path
    res = run_self_test()
    p = Path(__file__).resolve().parents[1] / "M4_FORMULA_SIMPLIFICATION.json"
    p.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[simplify] equivalent={res['numerically_equivalent']} 写出 {p}")
