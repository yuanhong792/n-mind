"""M41：正式模型实例的强制因果性与增量一致性审计。

不测试 math_lab 简化算子；只测试 R1/R2/M4-A/M4-B 正式类。
所有比较先检查 shape 完全相同，再计算逐位置最大绝对误差，禁止广播掩盖形状错误。
输出 M41F_CAUSALITY_AUDIT.json；历史 M41_CAUSALITY_AUDIT.json 与 M4_MATH_TESTS.json 不覆盖。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "models"))

from tokenizer_m4 import PAD  # noqa: E402
from gru_reference import GRUReferenceM3  # noqa: E402
from transformer_reference import TinyTransformerM3  # noqa: E402
from low_rank_interaction import LowRankInteractionM4  # noqa: E402
from selective_interaction import SelectiveInteractionM4  # noqa: E402

LENGTHS = [1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 32, 64, 128]
TOL = 1e-4


def builders():
    return {
        "R1_gru_reference": lambda: GRUReferenceM3(102, 64, 128),
        "R2_tiny_transformer": lambda: TinyTransformerM3(102, 64, 4, 2),
        "M4-A_low_rank_interaction": lambda: LowRankInteractionM4(102, 64, 4, 2, 16),
        "M4-B_selective_interaction": lambda: SelectiveInteractionM4(102, 64, 4, 2, 4, 2),
    }


def kw_for(name, hard=True):
    return {"hard_gate": hard} if name.startswith("M4-B") else {}


def max_diff(a, b, label):
    if tuple(a.shape) != tuple(b.shape):
        raise AssertionError(f"{label} shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
    return float((a - b).abs().max().detach())


def make_x(name, B, T, seed):
    torch.manual_seed(seed)
    return torch.randint(4, 100, (B, T))


def forward(model, x, name, hard=True, valid=None):
    return model(x, valid_mask=valid, **kw_for(name, hard))[0]


def incremental(model, x, name, hard=True, chunk=None):
    B, T = x.shape
    state = model.initial_state(B, x.device)
    outs = []
    i = 0
    ci = 0
    while i < T:
        n = min(T - i, chunk if chunk else 1)
        y, state = model(x[:, i:i + n], state=state, **kw_for(name, hard))
        outs.append(y)
        i += n
        ci += 1
    return torch.cat(outs, dim=1), state


def audit_model(name, builder):
    model = builder().eval()
    rec = {"model": name, "config": getattr(model, "cfg", {}),
           "n_params": model.n_params(), "status": "PASS", "failures": [],
           "causality": {}, "full_vs_incremental": {}, "chunked_prefill": {},
           "prefill_then_generate": {}, "batch_consistency": {},
           "padding_isolation": {}, "train_deploy": {}, "routing_gradient": None}

    # 1) 因果性：每个长度、每个可修改的未来位置都实测；逐位置切片形状一致。
    for T in LENGTHS:
        x = make_x(name, 1, T, 20261124 + T)
        worst = 0.0
        bad = []
        with torch.no_grad():
            y0 = forward(model, x, name, hard=True)
            for p in range(1, T):
                x2 = x.clone()
                x2[:, p:] = (x2[:, p:] + 1 + (p % 5)) % 96 + 4
                y1 = forward(model, x2, name, hard=True)
                d = max_diff(y0[:, :p], y1[:, :p], f"causal T={T} p={p}")
                worst = max(worst, d)
                if d >= TOL:
                    bad.append({"future_position": p, "max_abs_diff": d})
        rec["causality"][str(T)] = {
            "n_positions_tested": max(0, T - 1),
            "input_seed": 20261124 + T,
            "input_first_16": x[0, :min(16, T)].tolist(),
            "max_abs_diff": worst,
            "failures": bad,
            "pass": worst < TOL,
        }
        if bad:
            rec["failures"].append(f"causality_T{T}")

    # 2) 完整前向 vs 逐 token；3) 分块 prefill；4) prefill 后连续生成。
    for T in LENGTHS:
        x = make_x(name, 1, T, 3000000 + T)
        with torch.no_grad():
            yf = forward(model, x, name, hard=True)
            yi, _ = incremental(model, x, name, hard=True, chunk=1)
            chunks = [1, 3, 4, 5, 7, 8, 16]
            yc, _ = incremental(model, x, name, hard=True, chunk=None)
            outs, state = [], None
            i = 0; ci = 0
            while i < T:
                n = min(chunks[ci % len(chunks)], T - i)
                if state is None:
                    y, state = model(x[:, i:i + n], **kw_for(name, True))
                else:
                    y, state = model(x[:, i:i + n], state=state, **kw_for(name, True))
                outs.append(y); i += n; ci += 1
            yc = torch.cat(outs, dim=1)
            p = max(1, T // 2)
            yp, state = model(x[:, :p], **kw_for(name, True))
            rest = [yp]
            for t in range(p, T):
                y, state = model(x[:, t:t + 1], state=state, **kw_for(name, True))
                rest.append(y)
            ypg = torch.cat(rest, dim=1)
            d1 = max_diff(yf, yi, f"inc T={T}")
            d2 = max_diff(yf, yc, f"chunk T={T}")
            d3 = max_diff(yf, ypg, f"prefill T={T}")
        rec["full_vs_incremental"][str(T)] = {"max_abs_diff": d1, "pass": d1 < TOL}
        rec["chunked_prefill"][str(T)] = {"chunk_sizes": chunks, "max_abs_diff": d2, "pass": d2 < TOL}
        rec["prefill_then_generate"][str(T)] = {"prefill_tokens": max(1, T // 2),
                                                "max_abs_diff": d3, "pass": d3 < TOL}
        if d1 >= TOL or d2 >= TOL or d3 >= TOL:
            rec["failures"].append(f"parity_T{T}")

    # 5) 不同 batch size：batch 与逐样本 shape 必须一致。
    for B in (1, 2, 4):
        T = 17
        x = make_x(name, B, T, 4000000 + B)
        with torch.no_grad():
            yb = forward(model, x, name, hard=True)
            ys = torch.cat([forward(model, x[i:i + 1], name, hard=True)
                            for i in range(B)], dim=0)
            d = max_diff(yb, ys, f"batch B={B}")
        rec["batch_consistency"][str(B)] = {"T": T, "max_abs_diff": d, "pass": d < TOL}
        if d >= TOL:
            rec["failures"].append(f"batch_B{B}")

    # 6) Padding：有效位置 logits 必须与无 padding 完全一致。
    for L in (7, 17, 32):
        for pad_n in (0, 1, 3, 8):
            x = make_x(name, 1, L, 5000000 + L * 100 + pad_n)
            xp = torch.cat([x, torch.full((1, pad_n), PAD)], dim=1)
            valid = torch.ones(1, L + pad_n, dtype=torch.bool)
            valid[:, L:] = False
            with torch.no_grad():
                ya = forward(model, x, name, hard=True, valid=None)
                yb = forward(model, xp, name, hard=True, valid=valid)
                d = max_diff(ya, yb[:, :L], f"padding L={L} pad={pad_n}")
            rec["padding_isolation"][f"L{L}_pad{pad_n}"] = {
                "max_abs_diff": d, "pass": d < TOL}
            if d >= TOL:
                rec["failures"].append(f"padding_L{L}_pad{pad_n}")

    # 7) 训练/部署一致性与路由梯度。
    if name.startswith("M4-B"):
        x = make_x(name, 1, 17, 6000017)
        model.train()
        y_train_soft, st_train = model(x, hard_gate=False)
        y_train_hard, _ = model(x, hard_gate=True)
        model.eval()
        y_eval_hard, st_eval = model(x, hard_gate=True)
        d_th = max_diff(y_train_hard, y_eval_hard, "train_hard vs eval_hard")
        d_soft = max_diff(y_train_soft, y_eval_hard, "train_ste vs eval_hard")
        rec["train_deploy"] = {
            "train_hard_vs_eval_hard": d_th,
            "train_ste_forward_vs_eval_hard": d_soft,
            "pass": d_th < TOL,
            "note": "训练实际执行硬路径；STE 只改变梯度，不改变前向路径。",
        }
        if d_th >= TOL:
            rec["failures"].append("train_deploy")

        model.train()
        model.zero_grad(set_to_none=True)
        y, st = model(x, hard_gate=False)
        loss = y.float().pow(2).mean()
        grads = torch.autograd.grad(loss, list(model.parameters()), allow_unused=True)
        gs = [g for g in grads if g is not None]
        finite = all(torch.isfinite(g).all() for g in gs)
        norm = float(sum(g.norm().item() for g in gs))
        params = list(model.parameters())
        route_norms = []
        for i in range(len(model.blocks)):
            rp = model.blocks[i].attn.summary_proj.weight
            idx = next(j for j, p in enumerate(params) if p is rp)
            g = grads[idx]
            route_norms.append(None if g is None else float(g.norm().detach().item()))
        rec["routing_gradient"] = {
            "finite": bool(finite), "total_grad_norm": norm,
            "route_param_grad_norms": route_norms,
            "pass": bool(finite and norm > 0 and any(v and v > 0 for v in route_norms)),
        }
        if not rec["routing_gradient"]["pass"]:
            rec["failures"].append("routing_gradient")
    else:
        rec["train_deploy"] = {"pass": True, "note": "无离散路由；训练/部署前向路径相同。"}
        rec["routing_gradient"] = {"pass": True, "note": "无路由参数。"}

    if rec["failures"]:
        rec["status"] = "FAILED"
    return rec


def main():
    torch.set_num_threads(20)
    torch.manual_seed(20261124)
    records = {name: audit_model(name, builder) for name, builder in builders().items()}
    all_pass = all(r["status"] == "PASS" for r in records.values())
    out = {
        "status": "PASS" if all_pass else "FAILED",
        "test_scope": "formal_model_instances_only",
        "lengths": LENGTHS,
        "tolerance": TOL,
        "comparison_rule": "shape 完全相等后逐位置 max_abs_diff；禁止广播比较",
        "models": records,
        "decision": "任一模型 FAILED 即禁止进入正式训练" if not all_pass
                    else "全部正式模型通过强制因果/增量审计",
    }
    p = HERE / "M41F_CAUSALITY_AUDIT.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v["status"] for k, v in records.items()}, ensure_ascii=False))
    print(f"[m41] 写出 {p}")


if __name__ == "__main__":
    main()
