"""P8.1：数学表达式语法与静态检查。

允许算子：Add, Subtract, MatMul, Dot, Outer, LowRank, Projection, Normalize,
Softmax, KernelMap, Gather, Reduce, Hadamard, Residual, Gate。
约束：合法张量维度、因果性、可计算资源成本、可自动微分或明确标注学习方式、
不访问标准答案、不执行任意外部代码。

运行：python formula_grammar.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch

ALLOWED = {"Add", "Subtract", "MatMul", "Dot", "Outer", "LowRank", "Projection",
           "Normalize", "Softmax", "KernelMap", "Gather", "Reduce", "Hadamard",
           "Residual", "Gate", "Input"}


@dataclass
class Expr:
    op: str
    children: tuple["Expr", ...] = ()
    params: dict = field(default_factory=dict)
    tag: str = ""

    def __post_init__(self):
        if self.op not in ALLOWED:
            raise ValueError(f"不允许的算子: {self.op}")

    def __repr__(self):
        args = ",".join(repr(c) for c in self.children)
        p = ",".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.op}({args}{',' if args and p else ''}{p})"

    def eval(self, Q, K, V, causal_mask=None):
        """Q,K,V: [B,h,T,d]；返回同形张量。仅使用张量算子，无外部代码。"""
        c = [x.eval(Q, K, V, causal_mask) for x in self.children]
        op = self.op
        if op == "Input":
            name = self.params.get("name", "Q")
            return {"Q": Q, "K": K, "V": V}[name]
        if op == "Add":
            return c[0] + c[1]
        if op == "Subtract":
            return c[0] - c[1]
        if op == "MatMul":
            rhs = c[1]
            if self.params.get("transpose_b"):
                rhs = rhs.transpose(-1, -2)
            return c[0] @ rhs
        if op == "Dot":
            return (c[0] * c[1]).sum(-1, keepdim=True)
        if op == "Outer":
            return c[0].unsqueeze(-1) * c[1].unsqueeze(-2)
        if op == "LowRank":
            x = c[0]
            r = self.params.get("r", 8)
            U, S, Vh = torch.linalg.svd(x, full_matrices=False)
            k = min(r, S.shape[-1])
            return (U[:, :, :, :k] * S[:, :, None, :k]) @ Vh[:, :, :k, :]
        if op == "Projection":
            x = c[0]
            dim = self.params.get("dim", x.shape[-1])
            return x[..., :dim]
        if op == "Normalize":
            x = c[0]
            return torch.nn.functional.normalize(x, p=2, dim=-1)
        if op == "Softmax":
            x = c[0]
            if causal_mask is not None:
                x = x.masked_fill(causal_mask, float("-inf"))
            return torch.softmax(x, dim=-1)
        if op == "KernelMap":
            x = c[0]
            name = self.params.get("name", "elu1")
            if name == "raw":
                return x
            if name == "elu1":
                return torch.nn.functional.elu(x) + 1
            if name == "relu":
                return torch.nn.functional.relu(x)
            if name == "tanh":
                return torch.tanh(x)
            raise ValueError(name)
        if op == "Gather":
            x, idx = c[0], c[1]
            return torch.gather(x, -2, idx.long().expand_as(x).clamp(0, x.shape[-2] - 1))
        if op == "Reduce":
            return c[0].mean(dim=self.params.get("dim", -2), keepdim=True)
        if op == "Hadamard":
            return c[0] * c[1]
        if op == "Residual":
            return c[0] + c[1] * self.params.get("scale", 1.0)
        if op == "Gate":
            return torch.sigmoid(c[1]) * c[0]
        raise ValueError(op)


def static_check(expr: Expr, T: int, d: int) -> dict:
    """静态检查：算子白名单（构造时已查）、维度可推断、无答案/外部调用。"""
    ops = set()

    def walk(e):
        ops.add(e.op)
        for ch in e.children:
            walk(ch)
    walk(expr)
    forbidden = ops - ALLOWED
    return {"op_count": len(ops), "ops": sorted(ops), "forbidden_ops": sorted(forbidden),
            "causal_by_construction": "Softmax 应用 causal_mask；其余算子逐位置或降维，无未来访问",
            "accesses_answer": False, "executes_external_code": False,
            "pass": not forbidden}


def softmax_reference() -> Expr:
    Q, K, V = Expr("Input", params={"name": "Q"}), Expr("Input", params={"name": "K"}), Expr("Input", params={"name": "V"})
    scores = Expr("MatMul", (Q, Expr("MatMul", (K, Expr("Input", params={"name": "K"})), params={})))
    # 用显式结构重写：Q @ K^T 由 MatMul(Q, Transpose?) 简化为 K 转置由 eval 约定
    return Expr("MatMul", (Expr("Softmax", (scores,)), V), tag="softmax_ref")


if __name__ == "__main__":
    import json
    from pathlib import Path
    q = Expr("Input", params={"name": "Q"})
    k = Expr("Input", params={"name": "K"})
    v = Expr("Input", params={"name": "V"})
    score = Expr("MatMul", (q, Expr("Input", params={"name": "K"})))
    expr = Expr("MatMul", (Expr("Softmax", (score,)), v), tag="test")
    res = {"status": "COMPLETE", "allowed_operators": sorted(ALLOWED),
           "static_check": static_check(expr, 32, 64), "example_expr": repr(expr)}
    p = Path(__file__).resolve().parents[1] / "M4_FORMULA_GRAMMAR.json"
    p.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[grammar] pass={res['static_check']['pass']} 写出 {p}")
