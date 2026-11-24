"""P8.4：带成本约束与能力初筛的公式搜索主流程。

流程：生成 ≤64 候选 → 静态/数值/因果/隔离验证 → 无梯度误差与 CPU 基准 →
Pareto 非支配保留。近似误差只用于初筛，不能替代训练后任务能力。

运行：python formula_search.py
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from formula_generator import generate
from formula_simplifier import simplify
from formula_verifier import verify
from formula_cost_estimator import estimate_macs, measure_wall

HERE = Path(__file__).resolve().parents[1]


def pareto_front(rows: list[dict]) -> list[str]:
    front = []
    for i, a in enumerate(rows):
        dominated = False
        for j, b in enumerate(rows):
            if i == j:
                continue
            if (b["relative_l2"] <= a["relative_l2"] and
                    b["median_us"] <= a["median_us"] and
                    (b["relative_l2"] < a["relative_l2"] or b["median_us"] < a["median_us"])):
                dominated = True
                break
        if not dominated:
            front.append(a["id"])
    return front


def main():
    torch.set_num_threads(20)
    B, h, T, d = 2, 4, 32, 64
    Q, K, V = [torch.randn(B, h, T, d // h) for _ in range(3)]
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
    shapes = {"Q": Q.shape, "K": K.shape, "V": V.shape}
    cands = generate()
    ref = cands[0]["expr"]
    with torch.no_grad():
        y_ref = ref.eval(Q, K, V, mask)
    rows = []
    for c in cands:
        e = simplify(c["expr"])
        v = verify(e, T=T, d=d, B=B, h=h)
        row = {"id": c["id"], "tag": c["tag"], "description": c["description"],
               "expr": repr(e), "verify_pass": v["pass"], "verify": v}
        if not v["pass"]:
            row["eliminated"] = "FAILED_STATIC_OR_NUMERIC_CHECK"
            rows.append(row)
            continue
        with torch.no_grad():
            y = e.eval(Q, K, V, mask)
        row["relative_l2"] = round(float(((y - y_ref).norm() / y_ref.norm()).item()), 6)
        row["static_macs"] = estimate_macs(e, shapes)["macs"]
        row.update(measure_wall(e, Q, K, V, mask))
        row["eliminated"] = None
        rows.append(row)

    survived = [r for r in rows if not r["eliminated"]]
    front = pareto_front(survived) if survived else []
    out = {
        "status": "COMPLETE", "n_candidates": len(cands), "n_survived": len(survived),
        "note": "误差/CPU 仅为算子初筛；进入正式训练需预登记预算，当前 NOT_RUN",
        "pareto_front_operator_stage": front,
        "candidates": rows,
    }
    p = HERE / "M4_FORMULA_SEARCH_RESULTS.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[search] n={len(cands)} survived={len(survived)} pareto={front}")
    print(f"[search] 写出 {p}")


if __name__ == "__main__":
    main()
