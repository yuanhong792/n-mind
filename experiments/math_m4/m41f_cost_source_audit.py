"""M41F 训练成本来源审计：证明成本统计复用真实训练前向，不再额外跑一次模型。"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from compute_accounting_m4 import batch_cost
from models.selective_interaction import SelectiveInteractionM4

HERE = Path(__file__).resolve().parent


def audit():
    src = (HERE / "train_m4.py").read_text(encoding="utf-8")
    static_ok = ("fwd_state = model(" in src
                 and 'fwd_stats = fwd_state.get("stats")' in src
                 and "forward_stats=fwd_stats" in src
                 and "for _ in range(1):" not in src)
    model = SelectiveInteractionM4(102, d_model=64, nhead=4, layers=2,
                                   group_size=4, top_k=2)
    model.train()
    x = torch.randint(4, 101, (2, 17))
    valid = torch.ones(2, 17, dtype=torch.bool)
    w = torch.ones(2, 17)
    logits, state = model(x[:, :-1], valid_mask=valid[:, :-1])
    stats = state["stats"]
    calls = {"n": 0}
    original_forward = model.forward

    def guarded_forward(*args, **kwargs):
        calls["n"] += 1
        return original_forward(*args, **kwargs)

    model.forward = guarded_forward
    c = batch_cost(model, x, valid, w, backward=True, optimizer=True,
                   input_len=x.shape[1] - 1, hard_gate=True, forward_stats=stats)
    runtime_ok = calls["n"] == 0 and c.details["cost_source"] == "actual_training_forward"
    out = {
        "status": "PASS" if static_ok and runtime_ok else "FAILED",
        "checks": {
            "train_entry_uses_actual_forward": static_ok,
            "batch_cost_does_not_forward_again": runtime_ok,
        },
        "observed": {"second_model_forward_calls": calls["n"],
                     "cost_source": c.details["cost_source"],
                     "gather_elements": c.details.get("gather_elements"),
                     "gather_copy_bytes": c.details.get("gather_copy_bytes")},
        "note": "审计桩在 batch_cost 前替换了 forward；若被多调用一次会立即暴露。",
    }
    p = HERE / "M41F_COST_SOURCE_AUDIT.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[m41f-cost-source] status={out['status']} 写出 {p}")
    return out


if __name__ == "__main__":
    audit()
