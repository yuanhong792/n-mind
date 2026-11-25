"""阶段 2.7 A：EM/排名分离失败分类 + margin + C1 N-MIND 计时。只用 dev，不读封存集。"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

import mqar_task as T  # noqa: E402
from mqar_run import (DEV, TOK, _model, candidate_scores, free_gen,  # noqa: E402
                      rank_eval, wilson)

ARMS = HERE / "mqar_20261125" / "arms"
OUT = HERE / "mqar_20261125" / "phase27_analysis_a.json"
LEVELS = (1000, 3000, 10000, 30000)


def edit_distance(a: str, b: str) -> int:
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, dp[0] = dp[0], i
        for j, cb in enumerate(b, 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (ca != cb))
            prev = cur
    return dp[-1]


def classify(gen_ids, gen_str, true_str, ctx_vals):
    has_eos = bool(gen_ids.size and (gen_ids == TOK.vocab["<eos>"]).any())
    if gen_str == true_str:
        return "em", None, has_eos
    if len(gen_str) != 4 or not has_eos:
        kind = "c_format"
        if not has_eos and len(gen_str) >= 8:
            sub = "no_eos_extra_chars"
        elif len(gen_str) < 4:
            sub = "early_stop"
        else:
            sub = "len_mismatch"
        return kind, sub, has_eos
    if gen_str in ctx_vals:
        return "a_wrong_binding", None, has_eos
    if gen_str not in ctx_vals:
        d = min((edit_distance(gen_str, v), v) for v in ctx_vals)
        return "b_not_in_ctx", {"min_edit_distance": int(d[0]), "nearest": d[1]}, has_eos
    return "d_other", None, has_eos


@torch.no_grad()
def analyze_arm(arm_id: str, N: int):
    ds, _ = T.load_dataset(N, "heldout_seq")
    plen = int(ds["prompt_len"])
    true_ids = ds["dev_cand"][np.arange(len(ds["dev"])), ds["dev_true"]]
    true_str = [TOK.decode(list(r)) for r in true_ids]
    ctx_vals_all = []
    for r in ds["dev"]:
        s = T.parse_seq(TOK, r)
        ctx_vals_all.append([v for _, v in s["pairs"]])
    rows = []
    for step in LEVELS:
        ck = torch.load(ARMS / f"{arm_id}_step{step}.pt", weights_only=True)
        model = _model("transformer")
        model.load_state_dict(ck["model"])
        gen = free_gen(model, ds["dev"][:, :plen])
        gen_str = [TOK.decode(list(r)) for r in gen]
        em = np.asarray([int(g == t) for g, t in zip(gen_str, true_str)])
        rank = rank_eval(model, ds["dev"], ds["dev_cand"], ds["dev_true"], plen)
        # margin
        margins = []
        with torch.no_grad():
            for i in range(0, len(ds["dev"]), 256):
                x = torch.from_numpy(ds["dev"][i:i + 256]).long().to(DEV)
                c = torch.from_numpy(ds["dev_cand"][i:i + 256]).long().to(DEV)
                lp = torch.log_softmax(model(x), -1)
                sc = candidate_scores(lp, c, plen, T.NAME_LEN)
                tv = sc.gather(1, torch.from_numpy(ds["dev_true"][i:i + 256]).to(DEV)[:, None])
                wrong = sc.scatter(1, torch.from_numpy(ds["dev_true"][i:i + 256]).to(DEV)[:, None], -1e9)
                margins.append((tv - wrong.max(1, keepdim=True).values).cpu())
        margin = torch.cat(margins).numpy()
        # 分类
        cats = {"em": 0, "a_wrong_binding": 0, "b_not_in_ctx": 0, "c_format": 0, "d_other": 0}
        subs: dict = {}
        edit_dists = []
        no_eos = 0
        for i in range(len(ds["dev"])):
            k, sub, has_eos = classify(gen[i], gen_str[i], true_str[i], ctx_vals_all[i])
            cats[k] += 1
            key = sub if isinstance(sub, str) else (f"{k}:{sub['min_edit_distance']}" if isinstance(sub, dict) else str(sub))
            subs[key] = subs.get(key, 0) + 1
            if isinstance(sub, dict) and "min_edit_distance" in sub:
                edit_dists.append(sub["min_edit_distance"])
            if not has_eos:
                no_eos += 1
        lo = wilson(int(rank.sum()), len(rank))[0]
        row = {"arm": arm_id, "step": step, "N": N,
               "em": round(float(em.mean()), 4),
               "rank": round(float(rank.mean()), 4),
               "rank_wilson_lo": lo,
               "gate_at_this_step": lo >= 0.90,
               "margin_mean": round(float(margin.mean()), 4),
               "margin_p10": round(float(np.percentile(margin, 10)), 4),
               "margin_p50": round(float(np.percentile(margin, 50)), 4),
               "margin_p90": round(float(np.percentile(margin, 90)), 4),
               "margin_le_0": int((margin <= 0).sum()),
               "fail_n": int((em == 0).sum()),
               "cats": cats,
               "cats_ratio": {k: round(v / max(1, (em == 0).sum()), 4) for k, v in cats.items() if k != "em"},
               "c_sub": subs,
               "edit_dist_mean": round(float(np.mean(edit_dists)), 3) if edit_dists else None,
               "no_eos": no_eos}
        rows.append(row)
        print(row["arm"], step, "em", row["em"], "rank", row["rank"], "cats", cats)
        del model
        torch.cuda.empty_cache()
    return rows


def c1_timing():
    """C1 timing_only：N-MIND 参考配置在 MQAR N=2 实际长度、batch 32、GPU。"""
    from nmind.language.neural_dialogue import NMindLM, LMConfig

    class NMindAdapter(nn.Module):
        """仅包装 tuple 输出，不改 N-MIND 模型代码。"""

        def __init__(self):
            super().__init__()
            self.lm = NMindLM(LMConfig(vocab_size=TOK.vocab_size))

        def forward(self, x):
            lg, _ = self.lm(x)
            return lg

    m = NMindAdapter().to(DEV)
    n_par = m.lm.n_params()
    ds, _ = T.load_dataset(2, "heldout_seq")
    x = torch.from_numpy(ds["train"][:32]).long().to(DEV)
    L = int(ds["L"])
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    ce = nn.functional.cross_entropy
    V = TOK.vocab_size

    def step():
        opt.zero_grad()
        loss = ce(m(x[:, :-1]).reshape(-1, V), x[:, 1:].reshape(-1))
        loss.backward()
        opt.step()
        return float(loss)

    loss_last = None
    for _ in range(5):
        loss_last = step()
    torch.cuda.synchronize()
    samples = []
    for _ in range(5):
        t = time.perf_counter()
        for _ in range(20):
            loss_last = step()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t) * 1000 / 20)
    ms = float(np.median(samples))
    out = {"params": n_par, "reference": 434553, "within_5pct": abs(n_par - 434553) / 434553 <= 0.05,
           "L": L, "batch": 32, "step_ms_median": round(ms, 3),
           "step_ms_all": [round(s, 3) for s in samples],
           "est_30k_min": round(ms * 30000 / 1000 / 60, 2),
           "c2_allowed": ms * 30000 / 1000 / 60 <= 45,
           "loss_last": round(loss_last or 0.0, 4),
           "device": str(DEV), "gpu": torch.cuda.get_device_name(0)}
    print("C1", out)
    del m, opt
    torch.cuda.empty_cache()
    return out


def main():
    all_rows = []
    for N in (2, 4):
        for seed in (20270101, 20270102, 20270103):
            arm = f"transformer_N{N}_heldout_seq_s{seed}"
            all_rows += analyze_arm(arm, N)
    # 首次达标步数
    first_gate = {}
    for arm in sorted({r["arm"] for r in all_rows}):
        rs = [r for r in all_rows if r["arm"] == arm and r["gate_at_this_step"]]
        first_gate[arm] = rs[0]["step"] if rs else None
    OUT.write_text(json.dumps({"rows": all_rows, "first_gate_step": first_gate},
                              ensure_ascii=False), encoding="utf-8")  # 先落盘 A，C1 崩也不丢
    print("first_gate", first_gate)
    # 自由生成实现核对
    impl_check = {
        "prompt_excludes_answer": True,
        "prompt_start": "dev[:, :plen]",
        "max_new": 8,
        "stop": "eos",
        "em": "decode(gen, skip_special) == 4字值（不显式要求 eos 在串内；多生成字符会使 EM=0）",
        "perfect_model_em": "tests/test_mqar_eval_sanity.py::test_perfect_model_em_rank_follow_all_one = 1.0",
        "eval_sanity": "PASS 8/8 @ 54af4dc",
    }
    c1 = c1_timing()
    out = {"eval_sanity": "PASS 8/8 @ 54af4dc", "impl_check": impl_check,
           "rows": all_rows, "first_gate_step": first_gate, "c1": c1,
           "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved", OUT)


if __name__ == "__main__":
    import mqar_run  # noqa: F401  确保 prereg 断言通过
    main()
