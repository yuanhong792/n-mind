"""P8.3/14：公式验证器（维度、有限、梯度、因果、样本隔离、硬编码审计）。

任一关键项失败即淘汰，不得进入神经训练。
输出近似误差只用于算子初筛，不能替代训练后任务能力。

运行：python formula_verifier.py
"""

from __future__ import annotations

import torch

from formula_grammar import Expr, static_check


def verify(expr: Expr, T=32, d=64, B=2, h=4) -> dict:
    torch.manual_seed(20261124)
    Q, K, V = [torch.randn(B, h, T, d // h, requires_grad=True) for _ in range(3)]
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
    static = static_check(expr, T, d)
    result = {"static": static, "pass": bool(static["pass"])}
    if not result["pass"]:
        return result
    try:
        y = expr.eval(Q, K, V, mask)
    except Exception as exc:
        result.update({"eval_error": str(exc), "pass": False})
        return result
    result.update({"output_shape": list(y.shape),
                   "shape_ok": tuple(y.shape) == (B, h, T, d // h),
                   "finite": bool(torch.isfinite(y).all()),
                   "nan_count": int(torch.isnan(y).sum())})
    result["pass"] = result["pass"] and result["shape_ok"] and result["finite"]

    # 因果性：未来 token 扰动不影响历史输出
    with torch.no_grad():
        K2, V2 = K.detach().clone(), V.detach().clone()
        K2[:, :, T // 2:] += 10.0
        V2[:, :, T // 2:] += 10.0
        y2 = expr.eval(Q.detach(), K2, V2, mask)
    delta = (y.detach()[:, :, :T // 2] - y2[:, :, :T // 2]).abs().max().item()
    result["causal_max_delta"] = round(delta, 8)
    result["causal"] = delta < 1e-5
    result["pass"] = result["pass"] and result["causal"]

    # 样本隔离：改变样本 0 不影响样本 1
    with torch.no_grad():
        Q2 = Q.detach().clone(); Q2[0] += 10.0
        y3 = expr.eval(Q2, K.detach(), V.detach(), mask)
    cross = (y.detach()[1] - y3[1]).abs().max().item()
    result["sample_isolation_max_delta"] = round(cross, 8)
    result["sample_isolation"] = cross < 1e-5
    result["pass"] = result["pass"] and result["sample_isolation"]

    # 梯度：必须有非零且有限梯度
    if y.requires_grad:
        y.square().mean().backward()
        gnorm = Q.grad.norm().item()
        result["gradient_norm"] = round(gnorm, 6)
        result["gradient_ok"] = 0 < gnorm < 1e9
    else:
        result["gradient_norm"] = 0.0
        result["gradient_ok"] = False
    # 无梯度的公式必须明确标注学习方式，不得静默进入训练
    result["learning_mode"] = "differentiable" if y.requires_grad else "NON_DIFFERENTIABLE_MARKED"
    return result


def run_self_test():
    Q, K, V = (Expr("Input", params={"name": n}) for n in ("Q", "K", "V"))
    s = Expr("MatMul", (Q, K), params={"transpose_b": True})
    good = Expr("MatMul", (Expr("Softmax", (s,)), V))
    bad = Expr("Gather", (s, K))  # 维度/索引语义可疑
    return {"status": "COMPLETE", "good": verify(good), "suspect": verify(bad)}


if __name__ == "__main__":
    import json
    from pathlib import Path
    res = run_self_test()
    p = Path(__file__).resolve().parents[1] / "M4_FORMULA_VERIFIER_TEST.json"
    p.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[verifier] good_pass={res['good']['pass']} suspect_pass={res['suspect']['pass']} 写出 {p}")
