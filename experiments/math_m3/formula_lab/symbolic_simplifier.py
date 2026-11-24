"""M3-P6：数学表达式简化与等价检查。

目标：判定候选公式是否与已有候选数学等价（避免「表达形式不同」冒充创新）。
方法：
1. 结构规范化（交换 add/mul 参数顺序、折叠 identity、合并常数）；
2. 数值等价抽样测试（随机 h/x，比较输出，容差 1e-5）；
3. 已知等价规则表（如 gate(h,cand,1)=cand；gate(h,cand,0)=h；
   residual(h,0)=h；diag(h;a=1)=h；lowrank(U=0)=0）。
不宣称形式化证明，只做「工程等价检查」。
"""
from __future__ import annotations

import torch

from formula_grammar import tree_to_str


def canonicalize(node: dict) -> dict:
    """结构规范化：排序可交换算子参数、折叠显式 identity。"""
    op = node["op"]
    args = [canonicalize(a) if isinstance(a, dict) else a for a in node.get("args", [])]
    params = dict(node.get("params", {}))
    if op in ("add", "mul"):
        # 按字符串规范排序，消除 a+b 与 b+a 的差异
        args = sorted(args, key=lambda a: tree_to_str(a) if isinstance(a, dict) else repr(a))
    if op == "residual" and params.get("value") == 0:
        return args[0] if isinstance(args[0], dict) else {"op": "identity", "args": args[:1]}
    return {"op": op, "args": args, "params": params}


def known_equivalence(a: dict, b: dict) -> str | None:
    """返回已知等价规则名或 None。"""
    sa, sb = tree_to_str(canonicalize(a)), tree_to_str(canonicalize(b))
    if sa == sb:
        return "structural_canonical_equality"
    # 显式规则：gate with constant gate 0/1
    rules = [
        ("gate_1_is_candidate", lambda s: s.startswith("gate(") and "g=1" in s),
        ("gate_0_is_h", lambda s: s.startswith("gate(") and "g=0" in s),
    ]
    for name, pred in rules:
        if pred(sa) or pred(sb):
            return name
    return None


def numeric_equivalence(eval_a, eval_b, d=32, n=5, tol=1e-5) -> dict:
    """对两个可调用公式做随机输入数值等价测试。"""
    torch.manual_seed(0)
    max_diff = 0.0
    for _ in range(n):
        h = torch.randn(d)
        x = torch.randn(d)
        ya = eval_a(h, x); yb = eval_b(h, x)
        if ya.shape != yb.shape:
            return {"equivalent": False, "reason": "shape mismatch"}
        max_diff = max(max_diff, float((ya - yb).abs().max()))
    return {"equivalent": max_diff < tol, "max_diff": round(max_diff, 8),
            "tol": tol, "note": "工程数值等价抽样，非形式化证明"}
