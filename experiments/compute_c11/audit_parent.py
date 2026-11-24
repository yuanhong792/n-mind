"""P0：严格审计 C1.0 三个 checkpoint，统一验证面板选 C1.1 父模型。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
C1 = HERE.parent / "compute_c1"
sys.path.insert(0, str(C1))
sys.path.insert(0, str(HERE))

from tokenizer_compute import ComputeTokenizer
from model_compute import ComputeLM, load_compute
from evaluate_c11 import build_fixed_panel, evaluate_panel

VAL = HERE / "data" / "c11_val.jsonl"


def main():
    torch.set_num_threads(20)
    tok = ComputeTokenizer()
    cfg_model = ComputeLM  # alias
    val = [json.loads(l) for l in VAL.read_text(encoding="utf-8").splitlines() if l.strip()]
    panel = build_fixed_panel(val, per_task=50)
    paths = {n: C1 / f"candidate_compute_{n}.pt" for n in ("best", "final", "last")}

    # 逐张量加载 + 比较
    models = {}
    metas = {}
    for name, p in paths.items():
        m, blob = load_compute(p, tok)
        models[name] = m
        metas[name] = {"path": str(p), "steps": blob.get("optimizer_steps_executed"),
                       "role": blob.get("extra", {}).get("role"),
                       "saved_at": blob.get("saved_at"),
                       "tokenizer_sha": blob.get("tokenizer_sha256", "")[:16],
                       "config": blob.get("config")}

    def tensor_equal(a, b):
        return all(torch.equal(models[a].state_dict()[k], models[b].state_dict()[k])
                   for k in models[a].state_dict())

    # 随机初始化基线（同 seed）
    torch.manual_seed(20261124)
    rand = ComputeLM(models["best"].cfg)
    rand_diff = {n: round(float(sum((models[n].state_dict()[k].detach().float()
                                     - rand.state_dict()[k].detach().float()).abs().sum()
                                    for k in models[n].state_dict())), 3)
                 for n in models}

    # Router 负载缓冲比较
    router_load = {n: [round(float(x), 4) for x in models[n].block_load_count.tolist()]
                   for n in models}

    # 统一面板评测（best vs final；last==final 已知，也评以核实）
    evals = {}
    for n in ("best", "final"):
        evals[n] = evaluate_panel(models[n], tok, panel, collect_details=False)
        print(json.dumps({"phase": "parent_eval", "ckpt": n,
                          "macro": evals[n]["macro_accuracy"]}, ensure_ascii=False), flush=True)
    last_equal_final = tensor_equal("last", "final")
    best_equal_final = tensor_equal("best", "final")

    # 预登记选择指标：固定面板 macro_accuracy（并列时取更晚 step，防止早停偶然高点）
    selection = "fixed_panel_macro_accuracy; tie -> later optimizer step"
    cand = [("best", metas["best"]["steps"], evals["best"]["macro_accuracy"]),
            ("final", metas["final"]["steps"], evals["final"]["macro_accuracy"])]
    winner = sorted(cand, key=lambda x: (x[2], x[1]), reverse=True)[0][0]

    audit = {
        "status": "PARENT_AUDIT_DONE",
        "checkpoints": metas,
        "tensor_equality": {"best_equals_final": best_equal_final,
                            "last_equals_final": last_equal_final},
        "param_delta_vs_random_init_l1": rand_diff,
        "router_load_buffer": router_load,
        "fixed_panel": {"size": len(panel), "tasks": sorted({r["task"] for r in panel}),
                        "per_task_n": 50},
        "evaluation": evals,
        "selection_rule": selection,
        "selected_parent": winner,
        "note": "统一面板保证 Best/Final 同口径；不按文件 SHA 推断权重相同，逐张量核验。",
    }
    (HERE / "C11_PARENT_AUDIT.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"winner": winner, "best_macro": evals["best"]["macro_accuracy"],
                      "final_macro": evals["final"]["macro_accuracy"],
                      "last_eq_final": last_equal_final}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
