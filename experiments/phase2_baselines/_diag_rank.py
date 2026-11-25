"""A 项诊断：N=2 seed101 在 dev 集上贪心对但排名错的样本（不改 rank_eval）。"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

import mqar_task as T  # noqa: E402
from mqar_run import _model, free_gen, DEV, TOK, T as _  # noqa: E402

NAME_LEN = 4


def buggy_rank(lp, cand, plen):
    """复现现行 rank_eval 的 gather。"""
    seg = lp[:, plen - 1:plen - 1 + NAME_LEN, :]
    sc = seg.gather(2, cand).sum(-1)
    return sc


def fixed_rank(lp, cand, plen):
    """sc[b,j] = sum_t lp[b, plen-1+t, cand[b,j,t]]。"""
    seg = lp[:, plen - 1:plen - 1 + NAME_LEN, :]
    sc = 0
    for t in range(NAME_LEN):
        sc = sc + seg[:, t, :].gather(1, cand[:, :, t])
    return sc


@torch.no_grad()
def main():
    ck = torch.load(HERE / "mqar_20261125/arms/transformer_N2_heldout_seq_s20270101_step30000.pt",
                    weights_only=True)
    model = _model("transformer")
    model.load_state_dict(ck["model"])
    ds, meta = T.load_dataset(2, "heldout_seq")
    dev, plen = ds["dev"], int(ds["prompt_len"])
    cand = torch.from_numpy(ds["dev_cand"]).long().to(DEV)
    true = torch.from_numpy(ds["dev_true"]).to(DEV)

    gen = free_gen(model, dev[:, :plen])
    true_ids = ds["dev_cand"][np.arange(len(dev)), ds["dev_true"]]
    true_str = [TOK.decode(list(r)) for r in true_ids]
    gen_str = [TOK.decode(list(r)) for r in gen]
    em = np.asarray([int(g == t) for g, t in zip(gen_str, true_str)])

    ids_all = torch.from_numpy(dev).long().to(DEV)
    lps, bug_scores, fix_scores = [], [], []
    with torch.no_grad():
        for i in range(0, len(dev), 256):
            lp = torch.log_softmax(model(ids_all[i:i + 256]), -1)
            lps.append(lp)
    lp = torch.cat(lps)
    sc_bug = buggy_rank(lp, cand, plen)
    sc_fix = fixed_rank(lp, cand, plen)
    rank_bug = (sc_bug.argmax(-1) == true).cpu().numpy()
    rank_fix = (sc_fix.argmax(-1) == true).cpu().numpy()

    n = len(dev)
    print(f"dev n={n}")
    print(f"EM_greedy={em.mean():.4f}  rank_BUGGY={rank_bug.mean():.4f}  rank_FIXED={rank_fix.mean():.4f}")
    print(f"greedy==true 且 buggy rank 错: {int((em * (1 - rank_bug)).sum())} ({(em*(1-rank_bug)).mean():.4f})")
    print(f"greedy==true 且 fixed rank 错: {int((em * (1 - rank_fix)).sum())} ({(em*(1-rank_fix)).mean():.4f})")
    print(f"greedy!=true 但 buggy rank 对: {int(((1-em) * rank_bug).sum())}")
    print(f"greedy!=true 但 fixed rank 对: {int(((1-em) * rank_fix).sum())}")

    # 候选构造检查
    ok_in = ok_uniq = ok_tok = 0
    for i in range(n):
        s = T.parse_seq(TOK, dev[i])
        vals = [v for _, v in s["pairs"]]
        cset = [TOK.decode(list(r)) for r in ds["dev_cand"][i]]
        ok_in += int(set(cset) == set(vals) and len(cset) == 2)
        ok_uniq += int(len(set(cset)) == 2)
        ans_slice = TOK.decode(list(dev[i][plen:plen + 4]))
        ok_tok += int(TOK.decode(list(cand[i, ds["dev_true"][i]])) == ans_slice == true_str[i])
    print(f"候选=上下文值集合: {ok_in}/{n}  候选去重: {ok_uniq}/{n}  候选分词==答案切片: {ok_tok}/{n}")

    # 20 个贪心对但 buggy 排名错的例子
    idx = np.where((em == 1) & (rank_bug == 0))[0][:20]
    print(f"\n=== 20 个 greedy=正确 但 buggy 排名错 的 dev 样本 ===")
    for i in idx:
        j_true = int(true[i])
        j_wrong = 1 - j_true
        toks_t = cand[i, j_true].tolist()
        toks_w = cand[i, j_wrong].tolist()
        # 正确打分逐 token
        fix_t = [round(float(sc_fix[i, j_true]), 4)]
        fix_w = [round(float(sc_fix[i, j_wrong]), 4)]
        bug_t = float(sc_bug[i, j_true])
        bug_w = float(sc_bug[i, j_wrong])
        # 逐 token 正确 logp
        per_t = [round(float(lp[i, plen - 1 + t, toks_t[t]]), 3) for t in range(NAME_LEN)]
        per_w = [round(float(lp[i, plen - 1 + t, toks_w[t]]), 3) for t in range(NAME_LEN)]
        print(f"i={int(i)} q={T.parse_seq(TOK, dev[i])['query_key']} true={true_str[i]} gen={gen_str[i]}")
        print(f"  正确候选逐token logp(真上下文)={per_t} sum={fix_t[0]:.4f}")
        print(f"  错误候选逐token logp(真上下文)={per_w} sum={fix_w[0]:.4f}")
        print(f"  buggy分: 正确={bug_t:.4f} 错误={bug_w:.4f} → 判{'对' if rank_bug[i] else '错'}"
              f" | fixed分: 正确={fix_t[0]:.4f} 错误={fix_w[0]:.4f} → 判{'对' if rank_fix[i] else '错'}")
        print(f"  正确候选token={toks_t} 错误候选token={toks_w} 平局={int(sc_fix[i,j_true]==sc_fix[i,j_wrong])}")


if __name__ == "__main__":
    main()
