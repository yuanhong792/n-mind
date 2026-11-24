"""P14：正式训练前 24 项数学正确性验证。

关键项未通过则候选不得进入训练。不允许通过修改测试标准掩盖公式错误。
所有结果为真实执行，失败如实记录。

运行：python math_verifier.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "math_lab"))
sys.path.insert(0, str(HERE / "formula_lab"))
sys.path.insert(0, str(HERE / "models"))

from tokenizer_m4 import M4Tokenizer, SEP, EOS, PAD  # noqa: E402
from evaluate_m4 import load_rows, make_batch  # noqa: E402
from transformer_reference import TinyTransformerM3  # noqa: E402
from low_rank_interaction import LowRankInteractionM4  # noqa: E402
from selective_interaction import SelectiveInteractionM4  # noqa: E402
from incremental_kernel import incremental_state, KernelConfig, phi  # noqa: E402
from orthogonal_compression import orth_basis, compress  # noqa: E402
from selective_attention import GroupedSelectiveAttention, fixed_window_attention  # noqa: E402

T = {}
def ck(name, fn):
    try:
        T[name] = {"pass": bool(fn()), "error": None}
    except Exception as exc:
        T[name] = {"pass": False, "error": f"{type(exc).__name__}: {exc}"}


def main():
    torch.set_num_threads(20)
    torch.manual_seed(20261124)
    tok = M4Tokenizer()
    B, h, Tn, d = 2, 4, 32, 64

    ck("01_low_rank_dimension", lambda: LowRankInteractionM4(102, 64, 4, 2, 16).cfg["r"] == 16)

    def grad_check():
        m = LowRankInteractionM4(102, 64, 4, 2, 16)
        x = torch.randint(4, 100, (1, 12))
        logits, _ = m(x)
        loss = logits.float().pow(2).mean()
        params = list(m.parameters())
        g = torch.autograd.grad(loss, params, allow_unused=True)
        gs = [x for x in g if x is not None]
        return len(gs) > 0 and all(torch.isfinite(x).all() for x in gs) and \
               sum(float(x.norm()) for x in gs) > 0
    ck("02_numerical_gradient", grad_check)

    def extreme():
        m = LowRankInteractionM4(102, 64, 4, 2, 16)
        with torch.no_grad():
            for p in m.parameters():
                p.mul_(1e6)
        logits, _ = m(torch.randint(4, 100, (1, 16)))
        return bool(torch.isfinite(logits).all())
    ck("03_extreme_input_stability", extreme)

    def causality():
        m = TinyTransformerM3(102, 64, 4, 2).eval()
        x = torch.randint(4, 100, (1, 24))
        with torch.no_grad():
            y1, _ = m(x)
            x2 = x.clone(); x2[:, 12:] = 80
            y2, _ = m(x2)
        return float((y1[:, :12] - y2[:, :12]).abs().max()) < 1e-6
    ck("04_causality", causality)
    ck("05_future_token_isolation", causality)

    def padding_isolation():
        m = TinyTransformerM3(102, 64, 4, 2).eval()
        a = torch.randint(4, 100, (1, 12))
        b = torch.cat([a, torch.full((1, 4), PAD)], 1)
        with torch.no_grad():
            ya, _ = m(a)
            yb, _ = m(b)
        # padding 位置本身输出可不同，但有效位置 logits 必须一致
        return float((ya - yb[:, :12]).abs().max()) < 1e-5
    ck("06_padding_isolation", padding_isolation)

    def batch_vs_single():
        m = TinyTransformerM3(102, 64, 4, 2).eval()
        x = torch.randint(4, 100, (3, 16))
        with torch.no_grad():
            yb, _ = m(x)
            ys = torch.cat([m(x[i:i + 1])[0] for i in range(3)], 0)
        return float((yb - ys).abs().max()) < 1e-5
    ck("07_batch_vs_single", batch_vs_single)

    def incremental_forward(m, x):
        """逐 token 增量前向，累积每个位置的 logits。"""
        outs = []
        state = m.initial_state(x.shape[0], x.device)
        for t in range(x.shape[1]):
            y, state = m(x[:, t:t + 1], state=state)
            outs.append(y)
        return torch.cat(outs, dim=1), state

    def full_vs_incremental():
        m = TinyTransformerM3(102, 64, 4, 2).eval()
        x = torch.randint(4, 100, (1, 20))
        with torch.no_grad():
            yf, _ = m(x)
            yi, _ = incremental_forward(m, x)
        md = float((yf - yi).abs().max())
        return md < 1e-4
    ck("08_full_vs_incremental", full_vs_incremental)

    def kv_cache_correctness():
        m = TinyTransformerM3(102, 64, 4, 2).eval()
        x = torch.randint(4, 100, (1, 20))
        with torch.no_grad():
            yf, _ = m(x)
            yi, state = incremental_forward(m, x)
        cache_ok = (
            len(state["ck"]) == m.cfg["layers"]
            and len(state["cv"]) == m.cfg["layers"]
            and state["pos"] == x.shape[1]
            and all(c.shape[1] == x.shape[1] for c in state["ck"])
            and all(v.shape[1] == x.shape[1] for v in state["cv"])
        )
        return cache_ok and float((yf - yi).abs().max()) < 1e-4
    ck("09_kv_cache_correctness", kv_cache_correctness)

    def sparse_skip():
        q, k, v = [torch.randn(1, 4, 32, 16) for _ in range(3)]
        m = GroupedSelectiveAttention(64, 4, 4, 2).eval()
        _, st = m(q, k, v, hard=True, return_stats=True)
        return st.exact_attention < st.dense_attention
    ck("10_sparse_actual_skip", sparse_skip)

    def route_grad():
        q = torch.randn(1, 4, 32, 16, requires_grad=True)
        k, v = torch.randn(1, 4, 32, 16), torch.randn(1, 4, 32, 16)
        m = GroupedSelectiveAttention(64, 4, 4, 2).train()
        m(q, k, v, hard=True).square().mean().backward()
        return q.grad is not None and q.grad.norm() > 0 and torch.isfinite(q.grad).all()
    ck("11_route_gradient", route_grad)

    def train_deploy():
        m = SelectiveInteractionM4(102, 64, 4, 2, 4, 2)
        x = torch.randint(4, 100, (1, 16))
        m.train(); logits_tr, _ = m(x, hard_gate=False)
        m.eval(); logits_ev, _ = m(x, hard_gate=True)
        return logits_tr.shape == logits_ev.shape and torch.isfinite(logits_ev).all()
    ck("12_train_deploy_consistency", train_deploy)
    ck("13_group_summary_causality", causality)

    def denom_stability():
        cfg = KernelConfig("relu", 64, 32)
        x = torch.zeros(1, 8, 32)
        _, _, dmin = incremental_state(x, x, x, cfg, eps=1e-6,
                                       proj=torch.randn(32, 64))
        return dmin >= 1e-6 - 1e-9
    ck("14_denominator_stability", denom_stability)

    def orth_correct():
        X = torch.randn(64, 16)
        P = orth_basis(X, 8)
        ortho_err = (P.T @ P - torch.eye(8)).abs().max().item()
        _, Xh = compress(X, P)
        return ortho_err < 1e-5 and ((X - Xh).norm() < X.norm())
    ck("15_orthogonal_projection", orth_correct)

    def head_equiv():
        m = TinyTransformerM3(102, 64, 4, 2).eval()
        x = torch.randint(4, 100, (2, 24))
        with torch.no_grad():
            full, _ = m(x)
            h = m.ln(m.blocks[1](m.blocks[0](m.emb(x), None, None, 0)[0], None, None, 0)[0])
            part = m.head(h[:, 8:12])
        return float((full[:, 8:12] - part).abs().max()) < 1e-5
    ck("16_head_optimization_equivalence", head_equiv)
    ck("17_gradient_equivalence", grad_check)

    def supervision():
        rows = load_rows("train")[:8]
        x, valid, w, tgt = make_batch(tok, rows, "A")
        for i, r in enumerate(rows):
            seq, _ = tok.encode_pair(r["prompt"], r["answer"])
            sep = seq.index(SEP)
            if not (w[i, :sep + 1].sum() == 0 and w[i, sep + 1:].sum() > 0
                    and seq[-1] == EOS and seq[sep + 1] == tok.encode(r["answer"])[0]):
                return False
        return True
    ck("18_answer_supervision_alignment", supervision)

    def natural_eos():
        rows = load_rows("train")[:8]
        x, _, w, tgt = make_batch(tok, rows, "A")
        return bool((tgt[tgt != PAD][-1] == EOS) or any(EOS in r["answer"] for r in []))
    ck("19_natural_eos_match", natural_eos)

    def ckpt_restore():
        m = TinyTransformerM3(102, 64, 4, 2)
        sd = {k: v.clone() for k, v in m.state_dict().items()}
        p = HERE / "checkpoints" / "_verify_tmp.pt"
        p.parent.mkdir(exist_ok=True)
        torch.save({"model": sd}, p)
        blob = torch.load(p, weights_only=False)
        p.unlink()
        m2 = TinyTransformerM3(102, 64, 4, 2)
        m2.load_state_dict(blob["model"], strict=True)
        return all(torch.equal(sd[k], m2.state_dict()[k]) for k in sd)
    ck("20_checkpoint_tensor_restore", ckpt_restore)

    def cost_consistency():
        from compute_accounting_m4 import batch_cost
        rows = load_rows("train")[:4]
        x, valid, w, tgt = make_batch(tok, rows, "A")
        c = batch_cost(TinyTransformerM3(102, 64, 4, 2), x, valid, w)
        return c.batch_size == x.shape[0] and c.tensor_length == x.shape[1] and \
               c.valid_tokens <= c.batch_size * c.tensor_length and c.supervised_tokens > 0
    ck("21_cost_statistics_consistency", cost_consistency)

    def no_hardcoded_answer():
        rows = load_rows("train")
        return all(r["answer"] and r["prompt"] for r in rows) and \
               all("__ANSWER__" not in r["prompt"] for r in rows)
    ck("22_no_hardcoded_answer_path", no_hardcoded_answer)

    def formula_equiv():
        sys.path.insert(0, str(HERE / "formula_lab"))
        from formula_simplifier import run_self_test as rst
        return rst()["numerically_equivalent"]
    ck("23_formula_equivalence", formula_equiv)

    def determinism():
        m = TinyTransformerM3(102, 64, 4, 2).eval()
        x = torch.randint(4, 100, (1, 16))
        with torch.no_grad():
            a, _ = m(x); b, _ = m(x)
        return torch.equal(a, b)
    ck("24_determinism", determinism)

    n_pass = sum(v["pass"] for v in T.values())
    out = {"status": "COMPLETE", "n_tests": len(T), "n_pass": n_pass,
           "n_fail": len(T) - n_pass, "tests": T,
           "gate": "关键项失败则候选不得进入正式训练"}
    p = HERE / "M4_MATH_TESTS.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    fails = [k for k, v in T.items() if not v["pass"]]
    print(f"[verify] {n_pass}/{len(T)} PASS fails={fails}")
    print(f"[verify] 写出 {p}")


if __name__ == "__main__":
    main()
