"""M3 Pareto 评价系统（P11）。

多目标：能力↑、训练计算↓、推理计算↓、状态容量↓、CPU延迟↓。
候选模型只有在能力不下降、且至少一项资源成本改善时，才算对应维度的有效优化。

效率等级（仅适用于明确验证的任务范围）：
E1 >=20% 计算减少，E2 >=50%，E3 >=90%，E4 >=99%，E5 >=99.9%。

禁止用准确率/参数量作为唯一综合指标。
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def efficiency_grade(reduction_ratio: float) -> str:
    """根据计算减少比例给出效率等级。"""
    r = reduction_ratio
    if r >= 0.999:
        return "E5"
    if r >= 0.99:
        return "E4"
    if r >= 0.90:
        return "E3"
    if r >= 0.50:
        return "E2"
    if r >= 0.20:
        return "E1"
    return "BELOW_E1"


def pareto_front(models: dict) -> list[str]:
    """models: {name: {capability, cost_train, cost_infer, state_elems, latency}}
    返回非支配（Pareto 最优）模型名列表。
    目标：能力最大化，其余成本最小化。
    """
    names = list(models.keys())
    front = []
    for i, a in enumerate(names):
        dominated = False
        for j, b in enumerate(names):
            if i == j:
                continue
            ca, cb = models[a], models[b]
            # b 支配 a：能力>=且所有成本<=，且至少一项严格
            cap_ok = cb["capability"] >= ca["capability"]
            cost_ok = (cb["cost_train"] <= ca["cost_train"] and
                       cb["cost_infer"] <= ca["cost_infer"] and
                       cb["state_elems"] <= ca["state_elems"] and
                       cb["latency"] <= ca["latency"])
            strictly_better = (cb["capability"] > ca["capability"] or
                               cb["cost_train"] < ca["cost_train"] or
                               cb["cost_infer"] < ca["cost_infer"] or
                               cb["state_elems"] < ca["state_elems"] or
                               cb["latency"] < ca["latency"])
            if cap_ok and cost_ok and strictly_better:
                dominated = True
                break
        if not dominated:
            front.append(a)
    return front


def capability_efficiency(reference: dict, candidate: dict) -> dict:
    """同等能力下的效率比较。
    reference/candidate: {capability, compute_to_target, infer_cost}
    只有双方在相同任务上达到约定能力标准时才计算。
    """
    result = {"reference": reference.get("name"), "candidate": candidate.get("name")}
    if reference.get("capability", 0) < candidate.get("min_required", reference.get("capability")):
        result["status"] = "CANDIDATE_BELOW_REQUIREMENT"
        return result
    # 训练效率
    ref_c = reference.get("compute_to_target")
    cand_c = candidate.get("compute_to_target")
    if ref_c and cand_c and cand_c > 0:
        result["efficiency_train"] = round(ref_c / cand_c, 4)
        result["training_compute_reduction"] = round(1 - cand_c / ref_c, 4)
        result["efficiency_grade_train"] = efficiency_grade(1 - cand_c / ref_c)
    else:
        result["efficiency_train"] = None
        result["efficiency_grade_train"] = "TARGET_NOT_REACHED"
    # 推理效率
    ref_i = reference.get("infer_cost")
    cand_i = candidate.get("infer_cost")
    if ref_i and cand_i and cand_i > 0:
        result["efficiency_inference"] = round(ref_i / cand_i, 4)
        result["inference_compute_reduction"] = round(1 - cand_i / ref_i, 4)
        result["efficiency_grade_inference"] = efficiency_grade(1 - cand_i / ref_i)
    else:
        result["efficiency_inference"] = None
        result["efficiency_grade_inference"] = "TARGET_NOT_REACHED"
    return result


def main():
    # 用 compute_accounting 的真实数据构建示例 Pareto（未训练能力，仅资源维度）
    acc = json.loads((HERE / "M3_COMPUTE_ACCOUNTING.json").read_text(encoding="utf-8"))
    models = {}
    for key, v in acc["models"].items():
        # 取 BS=8, seq=32 的前向延迟作为 latency 代理
        lat = None
        for row in v["measurements"]:
            if row["batch_size"] == 8 and row["seq_len"] == 32:
                lat = row["forward_median_sec"]
        models[key] = {
            "capability": 0.0,   # 未训练，能力待训练后回填
            "cost_train": v["per_token_forward_mac"],
            "cost_infer": v["per_token_forward_mac"],
            "state_elems": v["state_elements"],
            "latency": lat or 0.0,
            "params": v["params"],
        }
    front = pareto_front(models)
    result = {
        "note": "未训练状态的资源维度 Pareto；能力维度待训练后回填重算",
        "models": models,
        "pareto_front_resource_only": front,
    }
    out = HERE / "M3_PARETO_RESULTS.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[pareto] 资源维度 Pareto front: {front}")
    print(f"[pareto] 写出 {out}")


if __name__ == "__main__":
    main()
