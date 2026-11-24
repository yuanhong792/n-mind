"""M3-P6：受控公式生成器（表达式树 + 受控变异 + 组合）。不依赖外部 LLM。

从种子公式出发，用允许的算子替换/插入有限步变异，生成 ≤32 个候选。
所有候选在构造后经 formula_verifier 淘汰。
"""
from __future__ import annotations

import copy
import random

from formula_grammar import validate_tree, tree_to_str

SEED = 20261201


def base_candidates(d=64, r=8):
    """任务书示例 F1/F2/F3 + 历史等价公式，作为种子。"""
    idn = {"op": "identity", "args": []}
    low = {"op": "lowrank", "args": [], "params": {"id": 0}}
    g = {"op": "sigmoid", "args": [{"op": "input_proj", "args": [], "params": {"W": "Wg"}}]}
    cand = {"op": "tanh", "args": [{"op": "add", "args": [
        {"op": "input_proj", "args": [], "params": {"W": "Wx"}},
        {"op": "add", "args": [{"op": "diag", "args": [], "params": {"a": "diag"}}, low]},
    ]}]}
    return {
        "F1_h_plus_lowrank": {"op": "residual", "args": [low]},
        "F2_gated_lowrank": {"op": "gate", "args": [idn, {"op": "tanh", "args": [low]}, g]},
        "F3_gate_diag_low_input": {"op": "gate", "args": [idn, cand, g]},
    }


def mutate(node: dict, rng: random.Random) -> dict:
    """受控变异：随机选一个子树，用等价类算子替换或包裹非线性。"""
    ops_1 = ["tanh", "sigmoid", "relu", "gelu", "norm"]
    if isinstance(node, dict) and node.get("args"):
        idx = rng.randrange(len(node["args"]))
        child = node["args"][idx]
        if isinstance(child, dict) and rng.random() < 0.5:
            node["args"][idx] = {"op": rng.choice(ops_1), "args": [child]}
        elif isinstance(child, dict) and child.get("op") in ops_1:
            node["args"][idx] = child["args"][0]
    return node


def generate(max_candidates=32, seed=SEED):
    rng = random.Random(seed)
    seeds = base_candidates()
    pool = {}
    for name, tree in seeds.items():
        ok, err = validate_tree(tree)
        if ok:
            pool[name] = tree
    # 受控变异扩展
    while len(pool) < max_candidates:
        parent_name = rng.choice(list(pool))
        child = mutate(copy.deepcopy(pool[parent_name]), rng)
        ok, err = validate_tree(child)
        if not ok:
            continue
        s = tree_to_str(child)
        name = f"M{len(pool):02d}"
        if s not in {tree_to_str(t) for t in pool.values()}:
            pool[name] = child
        if len(pool) >= max_candidates:
            break
    return pool
