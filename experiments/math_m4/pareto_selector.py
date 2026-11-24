"""从公式搜索结果中复算算子阶段 Pareto 前沿。

目标：同时最小化相对误差 relative_l2 与实测中位耗时 median_us。
输出 M4_PARETO_RESULTS.json。误差/耗时只用于算子初筛，不代表训练后能力。
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def dominates(a: dict, b: dict, error_key: str, cost_key: str) -> bool:
    no_worse = (a[error_key] <= b[error_key] and a[cost_key] <= b[cost_key])
    strictly_better = (a[error_key] < b[error_key] or a[cost_key] < b[cost_key])
    return no_worse and strictly_better


def select(rows: list[dict], error_key: str = "relative_l2",
           cost_key: str = "median_us") -> list[dict]:
    front = []
    for row in rows:
        dominators = [other["id"] for other in rows
                      if other is not row and dominates(other, row, error_key, cost_key)]
        if not dominators:
            front.append({"id": row["id"], "tag": row.get("tag"),
                          "relative_l2": row[error_key],
                          "median_us": row[cost_key],
                          "static_macs": row.get("static_macs")})
    return front


def main() -> None:
    src = HERE / "M4_FORMULA_SEARCH_RESULTS.json"
    data = json.loads(src.read_text(encoding="utf-8"))
    survivors = [r for r in data.get("candidates", [])
                 if r.get("verify_pass") and not r.get("eliminated")
                 and r.get("relative_l2") is not None and r.get("median_us") is not None]
    front = select(survivors)
    out = {
        "status": "COMPLETE",
        "source": "M4_FORMULA_SEARCH_RESULTS.json",
        "objectives": ["minimize_relative_l2", "minimize_median_us"],
        "n_survivors": len(survivors),
        "pareto_front_operator_stage": [r["id"] for r in front],
        "pareto_records": front,
        "note": "算子阶段 Pareto；不得替代训练后任务能力或 E1-E5 效率等级",
    }
    dst = HERE / "M4_PARETO_RESULTS.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[pareto] n={len(survivors)} front={out['pareto_front_operator_stage']} 写出 {dst}")


if __name__ == "__main__":
    main()
