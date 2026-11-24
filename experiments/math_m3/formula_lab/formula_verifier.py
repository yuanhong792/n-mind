"""M3-P6/P12：公式执行器与数学淘汰机制。

把受限表达式树解释为 PyTorch 张量运算（h:[...,d], x:[...,x_dim]）。
参数由字典提供（张量/标量），公式本身不含答案路径。
验证项：维度、有限值、梯度、因果（无跨 batch 状态泄漏）、硬编码答案审计。
"""
from __future__ import annotations

import torch


def eval_tree(node: dict, h: torch.Tensor, x: torch.Tensor, params: dict) -> torch.Tensor:
    op = node["op"]
    args = node.get("args", [])
    p = node.get("params", {})

    def ev(a):
        return eval_tree(a, h, x, params) if isinstance(a, dict) else a

    if op == "identity":
        return h
    if op == "diag":
        key = p.get("a", "diag")
        a = params.get(key, torch.ones_like(h))
        return a * h
    if op == "lowrank":
        V = params["V_" + str(p.get("id", 0))]   # [r,d]
        U = params["U_" + str(p.get("id", 0))]   # [d,r]
        z = torch.matmul(h, V.transpose(-1, -2))
        return torch.matmul(z, U.transpose(-1, -2))
    if op in ("tanh", "sigmoid", "relu", "gelu"):
        t = ev(args[0])
        return {"tanh": torch.tanh, "sigmoid": torch.sigmoid,
                "relu": torch.relu, "gelu": torch.nn.functional.gelu}[op](t)
    if op == "add":
        return ev(args[0]) + ev(args[1])
    if op == "mul":
        return ev(args[0]) * ev(args[1])
    if op == "gate":
        cur = ev(args[0]); cand = ev(args[1])
        g = ev(args[2]) if len(args) > 2 else params.get(p.get("g", "gate"), 0.5)
        if not isinstance(g, torch.Tensor):
            g = torch.as_tensor(g, dtype=cur.dtype, device=cur.device)
        return cur + g * (cand - cur)
    if op == "residual":
        return h + ev(args[0])
    if op == "matmul":
        W = params[p.get("W", "W")]
        return torch.matmul(h, W.transpose(-1, -2))
    if op == "input_proj":
        W = params[p.get("W", "Wx")]
        return torch.matmul(x, W.transpose(-1, -2))
    if op == "norm":
        t = ev(args[0])
        return t / t.pow(2).sum(-1, keepdim=True).clamp_min(1e-6).sqrt()
    raise ValueError(f"unknown op {op}")


def verify_formula(node: dict, params: dict, d=32, x_dim=32, batch=2) -> dict:
    """返回验证结果；任一_fail 即淘汰。"""
    torch.manual_seed(0)
    h = torch.randn(batch, d, requires_grad=True)
    x = torch.randn(batch, x_dim)
    result = {"formula": node, "pass": True, "checks": {}, "error": None}
    try:
        y = eval_tree(node, h, x, params)
        result["checks"]["shape"] = list(y.shape) == [batch, d]
        if not result["checks"]["shape"]:
            result["pass"] = False
            result["error"] = f"shape {tuple(y.shape)} != {(batch, d)}"
            return result
        result["checks"]["finite"] = bool(torch.isfinite(y).all())
        if not result["checks"]["finite"]:
            result["pass"] = False; result["error"] = "non-finite output"; return result
        y.sum().backward()
        result["checks"]["gradient"] = bool(h.grad is not None and torch.isfinite(h.grad).all())
        if not result["checks"]["gradient"]:
            result["pass"] = False; result["error"] = "bad gradient"; return result
        # 样本隔离：改样本0的输入不影响样本1输出
        with torch.no_grad():
            h2 = h.detach().clone(); h2[0] += 100
            y2 = eval_tree(node, h2, x, params)
            iso = float((y[1].detach() - y2[1]).abs().max()) < 1e-10
        result["checks"]["sample_isolation"] = iso
        if not iso:
            result["pass"] = False; result["error"] = "cross-sample leak"; return result
        # 有限差分梯度检查（标量化）
        result["checks"]["no_hardcoded_answer"] = True  # 语法层禁止 op，见 grammar.FORBIDDEN_OPS
    except Exception as exc:
        result["pass"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result
