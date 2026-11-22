"""M5-Language 回归测试（任务书第 9 节 10 项）。

全部使用小型模型配置（快）；训练态 buffer 审计口径与 M4 一致。
"""

import pytest
import torch

from nmind.language.decoder import generate
from nmind.language.dialogue_data import (OOD_PERSONS, SPLITS, TRAIN_PERSONS, TRAIN_OBJECTS,
                                           build_dialogue, build_split)
from nmind.language.neural_dialogue import LMConfig, NMindLM, load_lm, save_lm
from nmind.learning.online_adaptation import seal_eval_state, state_hash
from nmind.language.session import ChatSession
from nmind.language.tokenizer import BOS, EOS, PAD, UNK, CharTokenizer


def _tok() -> CharTokenizer:
    texts = ["用户：小明在3号位置。\n助手：好的。\n",
             "用户：请计算：3加4等于多少？\n助手：TOOL_CALCULATOR 3+4\n",
             "用户：它在哪里？\n助手：小明在5号位置。\n",
             "用户：月球上有几只兔子？\n助手：我不知道。\n"]
    return CharTokenizer.train_from_texts(texts)


def _cfg(vocab_size: int) -> LMConfig:
    return LMConfig(vocab_size=vocab_size, d_model=32, n_blocks=4, block_hidden=32,
                     workspace_dim=16, workspace_slots=2, top_k=2)


def test_1_tokenizer_roundtrip_unknown_specials():
    tok = _tok()
    s = "小明在3号位置。"
    ids = tok.encode(s, add_eos=True)
    assert ids[-1] == EOS
    assert tok.decode(ids[:-1]) == s
    assert tok.encode("", add_bos=True) == [BOS]
    unseen = tok.encode("\U0001400a")            # 训练集外的生僻字
    assert unseen == [UNK]
    assert tok.decode([UNK]) == "\ufffd"
    assert tok.vocab.get("<pad>") == PAD


def test_2_free_decode_has_no_label_leak():
    tok = _tok()
    torch.manual_seed(1)
    m = NMindLM(_cfg(tok.vocab_size))
    m.eval()
    prompt = tok.encode("用户：它在哪里？\n助手：")
    r1, _ = generate(m, tok, prompt, max_new_tokens=8)
    r2, _ = generate(m, tok, prompt, max_new_tokens=8)
    assert r1.token_ids == r2.token_ids and r1.text == r2.text   # 只依赖 prompt，与任何"答案"无关
    assert r1.finish_reason in ("eos", "newline", "max_tokens")


def test_3_session_isolation_and_reset():
    tok = _tok()
    torch.manual_seed(2)
    m = NMindLM(_cfg(tok.vocab_size))
    m.eval()
    s1 = ChatSession(m, tok, session_id="a")
    s2 = ChatSession(m, tok, session_id="b")
    s1.send("小明在3号位置。")
    assert s1.turns == 1 and s2.turns == 0
    s1.reset()
    assert s1.turns == 0 and s1.session_id.startswith("a")


def test_4_generation_stop_conditions():
    tok = _tok()
    torch.manual_seed(3)
    m = NMindLM(_cfg(tok.vocab_size))
    m.eval()
    prompt = tok.encode("用户：小明在哪里？\n助手：")
    rec, _ = generate(m, tok, prompt, max_new_tokens=3)
    assert rec.finish_reason == "max_tokens" and len(rec.token_ids) <= 3
    assert EOS not in rec.token_ids and tok.vocab.get("\n") not in rec.token_ids


def test_5_logits_trainable_grad_nonzero_random_init():
    tok = _tok()
    torch.manual_seed(4)
    m1 = NMindLM(_cfg(tok.vocab_size))
    torch.manual_seed(5)
    m2 = NMindLM(_cfg(tok.vocab_size))
    a = next(m1.parameters()).flatten()[:8]
    b = next(m2.parameters()).flatten()[:8]
    assert not torch.equal(a, b)                 # 权重来自不同随机初始化
    m1.train()
    ids = torch.tensor([tok.encode("用户：小明在3号位置。\n助手：", max_len=24)])
    logits, _st = m1(ids)
    loss = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1),
        ignore_index=PAD)
    loss.backward()
    assert loss.isfinite()
    assert tok_emb_grad_nonzero(m1)
    assert m1.head.weight.grad is not None and float(m1.head.weight.grad.norm()) > 0


def tok_emb_grad_nonzero(m: NMindLM) -> bool:
    g = m.tok_emb.weight.grad
    return g is not None and float(g.norm()) > 0


def test_6_save_reload_consistency_vocab_mismatch_rejected(tmp_path):
    tok = _tok()
    torch.manual_seed(6)
    m = NMindLM(_cfg(tok.vocab_size))
    m.eval()
    p = tmp_path / "lm.pt"
    save_lm(p, model=m, tokenizer=tok, cfg=m.cfg, steps=3, seed=6,
             parent_sha="random-init", manifest_hash="x")
    m2, meta = load_lm(p, tok)
    x = torch.tensor([tok.encode("用户：小明在哪里？\n助手：")])
    with torch.no_grad():
        l1, _ = m(x)
        l2, _ = m2(x)
    assert torch.equal(l1, l2) and meta["steps"] == 3
    other = CharTokenizer.train_from_texts(["甲乙丙丁戊己庚辛"])
    with pytest.raises(ValueError):
        load_lm(p, other)


def test_7_core_and_continual_entries_still_importable():
    """冒烟：既有入口未被语言栈破坏（完整回归由全量 pytest 承担）。"""
    from nmind.learning.trainer import train_continual  # noqa: F401
    from nmind.core.ncell import NCell  # noqa: F401
    from nmind.learning.online_adaptation import fit_existing_model  # noqa: F401
    assert True


def test_8_oracle_answers_and_split_isolation():
    spec = SPLITS["sealed_test"]
    d = build_dialogue("sealed_test", "sealed_test-0000", n_turns=3,
                        persons=spec.persons, objects=spec.objects,
                        seed=spec.seed * 1000)
    for i, t in enumerate(d.turns):
        if t.role == "user" and t.intent == "qa_pos":
            import re
            e = re.search(r"([^\s，。]+?)在哪里", t.text).group(1)
            ans = d.turns[i + 1].text
            assert ans.startswith(e) and "号位置" in ans      # oracle 与世界一致
    train_texts = "\n".join(t.text for t in
                              build_split(SPLITS["train"])[0].turns)
    for name in OOD_PERSONS:
        assert name not in train_texts                        # ood_name 实体训练不可见


def test_9_frozen_reproducible_and_buffers_cannot_fake_gains():
    tok = _tok()
    torch.manual_seed(9)
    m = NMindLM(_cfg(tok.vocab_size))
    m.eval()
    x = torch.tensor([tok.encode("用户：小明在哪里？\n助手：")])
    with torch.no_grad():
        l_base, _ = m(x)
    # 机制断言：单块偏置漂移改变路由概率 → 至少一个 prompt 的前向输出改变
    #（buffer 影响前向，必须复位；全块统一平移对 softmax 无效，所以单块漂移）
    prompts = [tok.encode("用户：小明在哪里？\n助手："),
                tok.encode("用户：请计算：3加4等于多少？\n助手："),
                tok.encode("助手："),
                tok.encode("用户：它在哪里？\n助手：")]
    snap = {k: v.clone() for k, v in m.state_dict().items()}   # 基线快照（漂移前）
    with torch.no_grad():
        base = [m(torch.tensor([p]))[0] for p in prompts]
    with torch.no_grad():
        m.router.load_bias[0] += 3.0
    with torch.no_grad():
        drift = [m(torch.tensor([p]))[0] for p in prompts]
    assert any(not torch.equal(a, b) for a, b in zip(base, drift))
    seal = seal_eval_state(m, snap)
    assert seal["reset_applied"] and seal["drift"]["router.load_bias"]["changed"]
    with torch.no_grad():
        restored = [m(torch.tensor([p]))[0] for p in prompts]
    assert all(torch.equal(a, b) for a, b in zip(base, restored))   # 复位后逐位一致


def test_10_no_network_no_external_llm(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("语言栈不得发起网络请求")
    monkeypatch.setattr("urllib.request.urlopen", boom)
    tok = _tok()
    d = build_dialogue("train", "t-0", n_turns=2, persons=TRAIN_PERSONS,
                        objects=TRAIN_OBJECTS, seed=1)
    torch.manual_seed(10)
    m = NMindLM(_cfg(tok.vocab_size))
    m.eval()
    generate(m, tok, tok.encode("用户：它在哪里？\n助手："), max_new_tokens=4)
    assert len(d.turns) > 0                        # 数据生成同样零网络
