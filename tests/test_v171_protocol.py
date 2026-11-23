"""V171 P2/P5 regression: full-answer conditional probability math + EOS diagnostics + answer mask."""
from __future__ import annotations
import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
V171 = ROOT / "experiments/language_v1/v171"; sys.path.insert(0, str(V171))
V17 = ROOT / "experiments/language_v1/v17"

import pytest
from nmind.language.tokenizer_v2 import ByteBPETokenizer, EOS
from nmind.language.neural_language_v2 import load_language_v2
import v171_protocol as P

PARENT = ROOT / "experiments/language_v1/v15/candidate_v15_baseline.pt"


@pytest.fixture(scope="module")
def setup():
    torch.set_num_threads(1)
    tok = P.build_tokenizer(); valid, lookup = P.build_vocab(tok)
    enc, fd = P.discriminative_positions(tok)
    m, _ = load_language_v2(PARENT, tok)
    return tok, valid, lookup, fd, m


def test_full_logprob_teacher_forced_equals_incremental(setup):
    """完整答案评分：单次 teacher-forced 前向 == 逐 token 增量 state passing。"""
    tok, valid, lookup, fd, m = setup
    p = "已知：小明有红色苹果，小红有绿色苹果。\n问：小明的苹果是什么颜色？\n答："
    for a in P.COLORS:
        tf = P.full_answer_logprob(m, tok, p, a, valid, lookup)
        inc = P.full_answer_incremental_logprob(m, tok, p, a, valid, lookup)
        assert abs(tf - inc) < 1e-3, f"{a}: tf={tf} inc={inc}"


def test_batch_equals_individual(setup):
    """批量候选评分 == 逐个评分（数学一致）。"""
    tok, valid, lookup, fd, m = setup
    p = "已知：小明有红色苹果，小红有绿色苹果。\n问：小明的苹果是什么颜色？\n答："
    batch = P.full_answer_logprob_batch(m, tok, p, P.COLORS, valid, lookup)
    for a in P.COLORS:
        ind = P.full_answer_logprob(m, tok, p, a, valid, lookup)
        assert abs(batch[a] - ind) < 1e-3, f"{a}: batch={batch[a]} ind={ind}"


def test_full_scoring_includes_all_tokens_and_eos(setup):
    """完整答案评分必须包含所有答案 token 与 EOS：长度更长分数不同。"""
    tok, valid, lookup, fd, m = setup
    p = "用户：你好\n助手："
    # 从 per-token 长度验证：答案+EOS 的 token 数 = len(encode(ans))+1
    _, per = P.full_answer_logprob_batch(m, tok, p, ["你好！"], valid, lookup, return_per_token=True)
    assert len(per["你好！"]) == len(tok.encode("你好！")) + 1


def test_eos_diagnostics_immediate_eos():
    """立即 EOS + 空串不得计为正确/非空。"""
    class G:
        raw_byte_hex = ""; token_ids = []; finish_reason = "eos"; text = ""
    d = P.eos_diagnostics(G())
    assert d["immediate_eos"] is True
    assert d["non_empty_answer"] is False
    assert d["valid_utf8_non_empty"] is False
    assert d["correct_eos"] is False


def test_eos_diagnostics_correct_eos():
    """有内容 + EOS 才算 correct_eos。"""
    class G:
        raw_byte_hex = "e7baa2"; token_ids = [1, 2]; finish_reason = "eos"; text = "红"
    d = P.eos_diagnostics(G())
    assert d["immediate_eos"] is False
    assert d["non_empty_answer"] is True
    assert d["correct_eos"] is True
    assert d["utf8_status"] == "complete"


def test_answer_mask_includes_eos():
    """训练答案掩码必须包含 EOS（V1.7 修复点回归）。"""
    sys.path.insert(0, str(V17))
    import train_v17 as T
    tok = P.build_tokenizer()
    enc, fd = T.discriminative_positions(tok)
    row = {"prompt": "用户：你好\n助手：", "answer": "你好！", "task": "A_dialogue"}
    x, mask, tw = T.make_batch(tok, [row], [0], enc, fd)
    # x[0] 末尾是 EOS, target=x[:,1:]; EOS 的 target 位置必须被 mask 覆盖
    seq = tok.encode(row["prompt"] + row["answer"], add_eos=True)
    assert seq[-1] == EOS
    assert bool(mask[0, len(seq) - 2]), "EOS target not in answer mask"


def test_proxy_label_distinct_from_full(setup):
    """PROXY 与 FULL 是不同口径：同一 prompt 下两者预测可以不同。"""
    tok, valid, lookup, fd, m = setup
    p = "已知：小明有红色苹果，小红有绿色苹果。\n问：小明的苹果是什么颜色？\n答："
    plp = P.proxy_first_token_lp(m, tok, p, P.COLORS, valid, fd)
    flp = P.full_answer_logprob_batch(m, tok, p, P.COLORS, valid, lookup)
    assert set(plp) == set(flp)
    # 两者数值不同（FULL 累积多 token, PROXY 只看判别 token）
    assert any(abs(plp[c] - flp[c]) > 0.01 for c in P.COLORS)
