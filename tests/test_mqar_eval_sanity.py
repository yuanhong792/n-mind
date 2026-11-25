"""阶段 2.6 B/A.3：MQAR 评测自检套件。

修复 rank gather bug 前运行必须失败（暴露问题）；修复后全过。
任何训练结果解读前本套件必须通过。
"""
from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments" / "phase2_baselines"))
sys.path.insert(0, str(ROOT / "src"))

import mqar_run as R  # noqa: E402
import mqar_task as T  # noqa: E402
from mqar_run import DEV, TOK, free_gen, rank_eval, sealed_eval  # noqa: E402

NAME_LEN = 4


class PerfectMQAR(nn.Module):
    """对给定 prompt 确定性输出正确答案（其他候选概率极低）。"""

    def __init__(self, pairs, plen: int, V: int):
        super().__init__()
        # pairs: list of (prompt_ids, answer_ids)  answer 不含 eos
        self.map = {tuple(p): list(a) for p, a in pairs}
        self.plen = plen
        self.V = V

    def forward(self, x):
        B, L = x.shape
        out = torch.full((B, L, self.V), -12.0, device=x.device)
        for i in range(B):
            row = x[i]
            key = tuple(row[: self.plen].tolist())
            ans = self.map.get(key)
            if ans is None:
                # 未登记的 prompt：均匀
                out[i] = 0.0
                continue
            # rank 用：答案槽位逐 token
            for k, tok in enumerate(ans):
                pos = self.plen - 1 + k
                if pos < L:
                    out[i, pos, int(tok)] = 12.0
            # free-gen 用：最后一位预测下一个答案 token / eos
            t = L - self.plen
            if t < len(ans):
                out[i, -1, int(ans[t])] = 12.0
            else:
                out[i, -1, TOK.vocab["<eos>"]] = 12.0
        return out


class UniformModel(nn.Module):
    def forward(self, x):
        return torch.zeros(x.shape[0], x.shape[1], TOK.vocab_size, device=x.device)


def _subset(ds, n=200):
    return {k: (v[:n] if isinstance(v, np.ndarray) and v.shape[0] == ds["n_test"] else v)
            for k, v in ds.items()}


def test_handwritten_scorer_uses_each_answer_position():
    """A.3.a 手写 scorer：候选必须在各自 4 个答案槽位上取 logp。

    构造：位置 3 上正确候选 token 的 logp 远高于其他；buggy gather
    （候选 0 只用位置 0）会漏掉位置 3 的证据而判错。
    """
    B, N, V, plen = 1, 2, 31, 5
    lp = torch.full((B, plen + NAME_LEN, V), -5.0)
    # 正确候选 [1,2,3,4]，错误候选 [5,6,7,8]
    cand = torch.tensor([[[1, 2, 3, 4], [5, 6, 7, 8]]])
    true = torch.tensor([0])
    # 每个位置对正确 token 给高概率；位置 3 差距最大
    for t, tok in enumerate([1, 2, 3, 4]):
        lp[0, plen - 1 + t, tok] = 2.0
    lp[0, plen - 1 + 3, 4] = 8.0  # 决定性证据在位置 3
    for t, tok in enumerate([5, 6, 7, 8]):
        lp[0, plen - 1 + t, tok] = -1.0
    sc = R.candidate_scores(lp, cand, plen)
    assert sc.shape == (B, N)
    assert sc[0, 0] > sc[0, 1], (sc, "正确候选应胜出（修复前 buggy gather 会判错）")
    assert (sc.argmax(-1) == true).all()


def test_perfect_model_em_rank_follow_all_one():
    """A.3.b / B：完美复制模型 EM=排名=跟随=1.0。"""
    ds, meta = T.load_dataset(2, "heldout_seq")
    ds = _subset(ds, 200)
    plen = int(ds["prompt_len"])
    pairs = []
    for i in range(len(ds["test"])):
        vals = [TOK.decode(list(r)) for r in ds["test_cand"][i]]
        ans = TOK.encode(vals[int(ds["test_true"][i])])
        pairs.append((ds["test"][i][:plen], ans))
        sw_ans = TOK.encode(TOK.decode(list(ds["test_swap_ans"][i])))
        pairs.append((ds["test_swap"][i][:plen], sw_ans))
    model = PerfectMQAR(pairs, plen, TOK.vocab_size).to(DEV)
    ev = sealed_eval(model, ds)
    assert ev["em_free"] == 1.0, ev["em_free"]
    assert ev["rank_acc"] == 1.0, ev["rank_acc"]
    assert ev["follow"] == 1.0, ev["follow"]


def test_uniform_model_near_random():
    """B：随机模型排名 ≈ 1/N，EM ≈ 0。"""
    ds, _ = T.load_dataset(2, "heldout_seq")
    ds = _subset(ds, 200)
    model = UniformModel().to(DEV)
    ev = sealed_eval(model, ds)
    assert abs(ev["rank_acc"] - 0.5) < 0.12, ev["rank_acc"]
    assert ev["em_free"] < 0.05, ev["em_free"]


def test_shuffle_labels_rank_drops_to_random():
    """A.3.d 打乱标签负对照：排名应接近 1/N。"""
    ds, _ = T.load_dataset(2, "heldout_seq")
    ds = _subset(ds, 200)
    plen = int(ds["prompt_len"])
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(ds["test"]))
    pairs = []
    for i in range(len(ds["test"])):
        j = perm[i]
        vals = [TOK.decode(list(r)) for r in ds["test_cand"][i]]
        ans = TOK.encode(vals[int(ds["test_true"][j])])  # 错配答案
        pairs.append((ds["test"][i][:plen], ans))
    model = PerfectMQAR(pairs, plen, TOK.vocab_size).to(DEV)
    ev = sealed_eval(model, ds)
    assert abs(ev["rank_acc"] - 0.5) < 0.12, ev["rank_acc"]


def test_real_ckpt_greedy_implies_rank():
    """A.3.c 一致性不变量：真 checkpoint 上贪心对 ⇒ 排名对（违反率≈0）。"""
    ck_path = ROOT / "experiments/phase2_baselines/mqar_20261125/arms/transformer_N2_heldout_seq_s20270101_step30000.pt"
    if not ck_path.exists():
        import pytest
        pytest.skip("checkpoint 不存在")
    ck = torch.load(ck_path, weights_only=True)
    model = R._model("transformer")
    model.load_state_dict(ck["model"])
    ds, _ = T.load_dataset(2, "heldout_seq")
    ds = _subset(ds, 300)
    plen = int(ds["prompt_len"])
    gen = free_gen(model, ds["dev"][:, :plen])
    true_ids = ds["dev_cand"][np.arange(len(ds["dev"])), ds["dev_true"]]
    true_str = [TOK.decode(list(r)) for r in true_ids]
    gen_str = [TOK.decode(list(r)) for r in gen]
    em = np.asarray([int(g == t) for g, t in zip(gen_str, true_str)])
    rank = rank_eval(model, ds["dev"], ds["dev_cand"], ds["dev_true"], plen)
    viol = int((em * (1 - rank)).sum())
    assert viol <= 3, f"贪心对但排名错的样本 {viol}，违反率 {(em*(1-rank)).mean():.4f}"


def test_prompt_does_not_contain_answer_or_eos():
    """B：prompt 段与答案段边界。"""
    for N in (2, 4):
        ds, meta = T.load_dataset(N, "heldout_seq")
        plen, L = int(ds["prompt_len"]), int(ds["L"])
        assert L == plen + NAME_LEN + 1
        eos = TOK.vocab["<eos>"]
        bos = TOK.vocab["<bos>"]
        for arr in (ds["train"][:50], ds["dev"][:50], ds["test"][:50]):
            assert (arr[:, :plen] != eos).all(), "prompt 段不得含 EOS"
            assert (arr[:, 0] == bos).all()
            assert (arr[:, plen - 1] != eos).all()
            # 答案段 = 4 字值 + eos
            assert (arr[:, -1] == eos).all()
            assert (arr[:, plen:] != bos).all()


def test_data_hash_mismatch_raises():
    """B：数据哈希不一致必须报错。"""
    import tempfile
    src = T.DATA_DIR / "mqar_N2_heldout_seq.npz"
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        shutil.copy(src, td / "mqar_N2_heldout_seq.npz")
        shutil.copy(T.DATA_DIR / "mqar_N2_heldout_seq.meta.json",
                    td / "mqar_N2_heldout_seq.meta.json")
        raw = bytearray((td / "mqar_N2_heldout_seq.npz").read_bytes())
        raw[100] ^= 0xFF
        (td / "mqar_N2_heldout_seq.npz").write_bytes(bytes(raw))
        try:
            T.load_dataset(2, "heldout_seq", out_dir=td)
            raise AssertionError("哈希不一致未报错")
        except RuntimeError as e:
            assert "DATA_HASH_MISMATCH" in str(e)


def test_cluster_bootstrap_ci():
    """E：簇级 bootstrap CI。"""
    vals = [1.0] * 50 + [0.0] * 50
    clusters = [f"c{i//10}" for i in range(100)]  # 10 簇
    ci = R.cluster_bootstrap_ci(vals, clusters, b=500, seed=0)
    assert 0.2 <= ci[0] <= 0.8 <= ci[1] or ci[0] <= 0.5 <= ci[1]
    # 全 1 应 CI=[1,1]
    ci2 = R.cluster_bootstrap_ci([1.0] * 20, [f"c{i}" for i in range(20)], b=200, seed=0)
    assert ci2 == [1.0, 1.0]
