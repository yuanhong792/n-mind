"""P8.2：受控公式变异生成器（最多 64 候选，确定性，无 LLM）。

变异类型：降低矩阵交互维度、合并重复计算、替换归一化/核函数、
改变交互顺序、全局替换为分组、低秩残差修正、删除冗余节点。
所有候选只由白名单算子组成；不访问标准答案。

运行：python formula_generator.py
"""

from __future__ import annotations

from formula_grammar import Expr

MAX_CANDIDATES = 64


def inputs():
    return (Expr("Input", params={"name": "Q"}), Expr("Input", params={"name": "K"}),
            Expr("Input", params={"name": "V"}))


def score_expr(qmode="raw", kmode="raw"):
    Q, K, _ = inputs()
    q = Expr("KernelMap", (Q,), params={"name": qmode}) if qmode else Q
    k = Expr("KernelMap", (K,), params={"name": kmode}) if kmode else K
    return Expr("MatMul", (q, k), params={"transpose_b": True})


def base_attention(score, vmode="raw", r=None):
    _, _, V = inputs()
    v = Expr("LowRank", (V,), params={"r": r}) if r else V
    return Expr("MatMul", (Expr("Softmax", (score,)), v))


def generate() -> list[dict]:
    """返回 ≤64 个候选，每个含 id/tag/expr/description。"""
    out = []
    def add(tag, expr, desc):
        if len(out) < MAX_CANDIDATES:
            out.append({"id": f"F{len(out)+1:02d}", "tag": tag, "expr": expr, "description": desc})

    Q, K, V = inputs()
    s = score_expr()
    add("softmax_ref", base_attention(s), "标准 softmax 注意力（参考）")

    # 1) 降低 score 有效秩（后处理低秩，不省 QK）
    for r in (4, 8, 16, 32):
        add(f"lowrank_score_r{r}", base_attention(Expr("LowRank", (s,), params={"r": r})),
            f"对 score 矩阵做 rank-{r} SVD；注意不省 QK 乘法")

    # 2) 降低 Value 表达维度
    for r in (4, 8, 16, 32):
        add(f"lowrank_value_r{r}", base_attention(s, r=r), f"Value 降至 rank-{r}")

    # 3) 核函数替换 Q/K
    for qm in ("relu", "elu1", "tanh"):
        for km in ("", "relu", "elu1"):
            add(f"kernel_q{qm or 'raw'}_k{km or 'raw'}",
                base_attention(score_expr(qm, km)), f"Q 核={qm or 'raw'}, K 核={km or 'raw'}")

    # 4) 归一化与门控变体
    add("normalize_q", base_attention(Expr("MatMul", (Expr("Normalize", (Q,)), K), params={"transpose_b": True})),
        "Q 做 L2 归一化")
    add("normalize_k", base_attention(Expr("MatMul", (Q, Expr("Normalize", (K,))), params={"transpose_b": True})),
        "K 做 L2 归一化")
    add("gate_value", Expr("Gate", (base_attention(s), V)), "输出用 V 做 sigmoid 门控")
    add("residual_v", Expr("Residual", (base_attention(s), V), params={"scale": 0.1}),
        "加 V 的低权重残差")

    # 5) 分组 Reduce 粗交互（全局→分组）
    reduced_q = Expr("Reduce", (Q,), params={"dim": -2})
    reduced_k = Expr("Reduce", (K,), params={"dim": -2})
    add("group_mean_interaction",
        Expr("MatMul", (Expr("Softmax", (Expr("MatMul", (reduced_q, reduced_k), params={"transpose_b": True}),)), V)),
        "Q/K 按序列维求均值后交互，测试分组摘要极限")

    # 6) 交互顺序变体
    add("v_then_score", Expr("MatMul", (Expr("Softmax", (s,)), Expr("Normalize", (V,)))),
        "Value 先归一化再做聚合")
    add("hadamard_gate", Expr("Hadamard", (base_attention(s), Expr("Gate", (base_attention(s), V)))),
        "输出与门控结果逐元素乘")

    # 7) 更多确定性组合直到上限
    for r in (8, 16):
        for qm in ("relu", "elu1"):
            add(f"kernel_q{qm}_score_lr{r}",
                base_attention(Expr("LowRank", (score_expr(qm, ""),), params={"r": r})),
                f"核映射 + score rank-{r} 组合")

    return out[:MAX_CANDIDATES]


if __name__ == "__main__":
    import json
    from pathlib import Path
    cands = generate()
    registry = {"status": "COMPLETE", "max_candidates": MAX_CANDIDATES,
                "n_generated": len(cands),
                "candidates": [{"id": c["id"], "tag": c["tag"], "description": c["description"],
                                "expr": repr(c["expr"])} for c in cands]}
    p = Path(__file__).resolve().parents[1] / "M4_FORMULA_REGISTRY.json"
    p.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[generator] n={len(cands)} 写出 {p}")
