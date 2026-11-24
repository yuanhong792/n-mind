"""多目标 Pareto 选择：只做非支配排序，不把训练Loss当架构优劣依据。"""
from __future__ import annotations


def dominates(a, b, maximize, minimize):
    not_worse = True
    better = False
    for k in maximize:
        if a.get(k, float("-inf")) < b.get(k, float("-inf")):
            not_worse = False
            break
        if a.get(k, float("-inf")) > b.get(k, float("-inf")):
            better = True
    if not_worse:
        for k in minimize:
            if a.get(k, float("inf")) > b.get(k, float("inf")):
                not_worse = False
                break
            if a.get(k, float("inf")) < b.get(k, float("inf")):
                better = True
    return not_worse and better


def pareto_front(records: dict, maximize=("accuracy", "counterfactual", "retention"),
                 minimize=("trainable_params", "state_elements", "cpu_latency_ms", "train_time")):
    items = list(records.items())
    front = []
    dominated = {}
    for name, rec in items:
        dominated[name] = any(
            other != name and dominates(recs, rec, maximize, minimize)
            for other, recs in items)
    front = [name for name, _ in items if not dominated[name]]
    return {"front": front, "dominated": dominated,
            "maximize": list(maximize), "minimize": list(minimize)}
