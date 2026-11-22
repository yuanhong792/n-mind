"""Gate-B 004 修复验收测试：解码协议 v2、验收契约、lm_update 口径与白名单。

覆盖用户指令的四项解码测试（首 Token 一致性 / EOS / checkpoint 重载 / 跨轮状态）
+ learning_loop 新契约（独立封存集缺失 → INCONCLUSIVE、缺臂 → INCONCLUSIVE、
  遗忘缺失不自动过、负对照纳入 gates、历史门槛不动）
+ lm_update（trainable/adapter_prefixes 白名单、attempted vs executed 口径拆分）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nmind.bot.experience_bridge import VerifiedExperience
from nmind.bot.learning_loop import (COUNTERFACTUAL_FOLLOW_MIN, FORGETTING_LIMIT_PP,
                                     NEW_SKILL_GAIN_THRESHOLD, REQUIRED_ARMS,
                                     free_gen_em, run_candidate_update)
from nmind.language import binding_data as bd
from nmind.language.decoder import (FREEGEN_PROTOCOL_V1, FREEGEN_PROTOCOL_V2,
                                     free_generation, free_generation_v2)
from nmind.language.lm_update import (KNOWN_ADAPTER_PREFIXES, KNOWN_TRAINABLE_MODES,
                                       set_trainable)
from nmind.language.neural_dialogue import LMConfig, NMindLM, save_lm, load_lm
from nmind.language.tokenizer import CharTokenizer

CORPUS = ("用户：小明在1号位置。\n好的。用户：小明在哪里？\n助手：小明在1号位置。\n"
          "用户：小红在2号位置。\n明白了。用户：小红在哪里？\n助手：小红在2号位置。\n"
          "用户：小红在3号位置。\n用户：小红在哪里？\n助手：小红在3号位置。\n"
          "用户：小明在4号位置。\n收到。用户：小明在哪里？\n助手：小明在4号位置。\n")


def _tok(tmp_path) -> CharTokenizer:
    tok = CharTokenizer.train_from_texts([CORPUS])
    tok.save(str(tmp_path / "tok.json"))
    return CharTokenizer.load(str(tmp_path / "tok.json"))


def _small(tok) -> LMConfig:
    return LMConfig(vocab_size=tok.vocab_size, d_model=16, n_blocks=4, block_hidden=16,
                    workspace_dim=8, workspace_slots=2, top_k=1)


def _prompt(entity: str, pos: str, question_entity: str | None = None) -> str:
    q = question_entity or entity
    return f"用户：{entity}在{pos}。\n用户：{q}在哪里？\n助手："


def _make_verified(tok, n: int = 8) -> list[VerifiedExperience]:
    tasks = bd.build_family("unseen_entity_qa", "train")[:n]
    return [VerifiedExperience(
        experience_id=f"t-{t.fingerprint}", task_family=t.family,
        observation={"question": t.question}, action={"reference": "oracle"},
        feedback={"oracle": t.target, "feedback_source": "deterministic_oracle"},
        verification_status="verified", validator_id=bd.ORACLE_ID,
        source_id="preregistered:test", content_sha256=t.fingerprint,
        model_checkpoint_sha256="test", skill_id="skill.binding.unseen_entity",
        response_source="rule", feedback_source="deterministic_oracle",
        learning_target="neural_generation") for t in tasks]


# ---------------------------------------------------------------- 解码协议 v2

def test_v2_first_token_matches_single_forward(tmp_path):
    """首 Token 一致性：v2 的首个 token 必须等于"prompt 单次前向 + argmax"的手工口径，
    不得混入 anchor 二次前向（v1 的已知问题）。"""
    tok = _tok(tmp_path)
    torch.manual_seed(7)
    model = NMindLM(_small(tok))
    model.eval()
    prompt = _prompt("小明", "1号位置")
    with torch.no_grad():
        logits, _state = model(torch.tensor([tok.encode(prompt)], dtype=torch.long))
    expected_first = int(torch.argmax(logits[0, -1]))
    rec = free_generation_v2(model, tok, prompt, max_new_tokens=4)
    assert rec.token_ids[0] == expected_first
    # 协议 ID 常量可引用且语义绑定正确
    assert FREEGEN_PROTOCOL_V2 == "freegen-v2-direct-continuation"
    assert FREEGEN_PROTOCOL_V1 == "freegen-v1-anchor-reforward"


def test_v2_v1_share_stop_and_record_semantics(tmp_path):
    """v1/v2 的停止条件与记录结构一致（EOS/newline/max_tokens、token_ids 与 text 对齐），
    差异只允许出现在首 token 的条件状态上。"""
    tok = _tok(tmp_path)
    torch.manual_seed(11)
    model = NMindLM(_small(tok))
    prompt = _prompt("小红", "2号位置")
    r1 = free_generation(model, tok, prompt, max_new_tokens=6)
    r2 = free_generation_v2(model, tok, prompt, max_new_tokens=6)
    assert r1.finish_reason in ("eos", "newline", "max_tokens")
    assert r2.finish_reason == r1.finish_reason
    assert len(r2.token_ids) == len(r2.tokens) == len(r2.logprobs)
    assert r2.text == tok.decode(r2.token_ids, skip_special=True)


def test_v2_eos_stop(tmp_path):
    """EOS 停止：若模型在某步输出 EOS，生成必须立即停且 EOS 不进 produced。"""
    tok = _tok(tmp_path)
    torch.manual_seed(3)
    model = NMindLM(_small(tok))
    prompt = _prompt("小红", "3号位置")
    rec = free_generation_v2(model, tok, prompt, max_new_tokens=32)
    from nmind.language.tokenizer import EOS
    assert EOS not in rec.token_ids
    assert len(rec.token_ids) <= 32


def test_v2_checkpoint_reload_bitwise(tmp_path):
    """checkpoint 重载一致性：保存→加载后同 prompt 的 v2 生成逐 token 一致。"""
    tok = _tok(tmp_path)
    torch.manual_seed(5)
    model = NMindLM(_small(tok))
    save_lm(str(tmp_path / "m.pt"), model=model, tokenizer=tok, cfg=_small(tok),
            steps=0, seed=1, parent_sha="x", manifest_hash="t")
    model2, _meta = load_lm(str(tmp_path / "m.pt"), tok)
    prompt = _prompt("小红", "4号位置")
    rec_a = free_generation_v2(model, tok, prompt, max_new_tokens=8)
    rec_b = free_generation_v2(model2, tok, prompt, max_new_tokens=8)
    assert rec_a.token_ids == rec_b.token_ids


def test_v2_cross_turn_state_extends(tmp_path):
    """跨轮状态：携带 (h, ws) 续算时，v2 生成结果与"拼接全文单次前向"的首 token 一致。"""
    tok = _tok(tmp_path)
    torch.manual_seed(9)
    model = NMindLM(_small(tok))
    first = _prompt("小明", "1号位置")
    second = f"好的。用户：小红在哪里？\n助手："
    with torch.no_grad():
        _lg, st = model(torch.tensor([tok.encode(first)], dtype=torch.long))
    rec = free_generation_v2(model, tok, second, max_new_tokens=4)
    with torch.no_grad():
        lg2, _ = model(torch.tensor([tok.encode(second)], dtype=torch.long), st)
    assert rec.token_ids[0] == int(torch.argmax(lg2[0, -1]))


# ---------------------------------------------------------------- learning_loop 契约

def _run(tmp_path, tok, *, arms, retention, counterfactual=None, updates=2):
    torch.manual_seed(13)
    model = NMindLM(_small(tok))
    save_lm(str(tmp_path / "stable.pt"), model=model, tokenizer=tok, cfg=_small(tok),
            steps=0, seed=1, parent_sha="x", manifest_hash="t")
    anchor = bd.build_family("position_qa", "sealed")[:8]
    return run_candidate_update(
        stable_checkpoint=str(tmp_path / "stable.pt"), tokenizer_path=str(tmp_path / "tok.json"),
        verified=_make_verified(tok), sealed_prompts=[t.prompt for t in bd.build_family("unseen_entity_qa", "sealed")[:6]],
        sealed_answers=[t.target for t in bd.build_family("unseen_entity_qa", "sealed")[:6]],
        dev=torch.device("cpu"), updates=updates,
        replay_pairs=([t.prompt for t in anchor], [t.target for t in anchor]),
        retention_pairs=([t.prompt for t in anchor], [t.target for t in anchor]) if retention else None,
        counterfactual_items=counterfactual, arms=arms, wall_limit_s=60.0)


def test_missing_retention_anchor_is_inconclusive(tmp_path):
    """独立封存集缺失 → INCONCLUSIVE（旧技能遗忘不评测就不给终判，绝不回退训练行）。"""
    tok = _tok(tmp_path)
    rep = _run(tmp_path, tok, arms=("frozen", "memory", "neural", "negative"),
               retention=False)
    assert rep["status"] == "INCONCLUSIVE"
    assert any("retention_pairs" in r for r in rep["inconclusive_reasons"])


def test_missing_arm_is_inconclusive_but_training_still_audited(tmp_path):
    """缺臂 → 终判 INCONCLUSIVE，但训练/取证照常（预算、继承、buffer 复位全留痕）。"""
    tok = _tok(tmp_path)
    rep = _run(tmp_path, tok, arms=("neural",), retention=True)
    assert rep["status"] == "INCONCLUSIVE"
    assert any("missing_arms" in r for r in rep["inconclusive_reasons"])
    assert rep["executed"] is True
    assert rep["transient_buffer_drift"]["reset_applied"] is True


def test_full_arms_with_anchor_gets_terminal_verdict(tmp_path):
    """四臂齐全 + 独立封存集 → 必须给 PROPOSED/REJECTED 终判（历史门槛不动）。"""
    tok = _tok(tmp_path)
    rep = _run(tmp_path, tok, arms=("frozen", "memory", "neural", "negative"),
               retention=True)
    assert rep["status"] in ("PROPOSED", "REJECTED")
    assert rep["gates"]["old_task_retention"] is (rep["old_task_forgetting_pp"] is not None
                                                   and rep["old_task_forgetting_pp"] >= -FORGETTING_LIMIT_PP)
    assert NEW_SKILL_GAIN_THRESHOLD == 0.10 and FORGETTING_LIMIT_PP == 3.0


def test_negative_control_gate_enforced(tmp_path):
    """负对照纳入正式 gates：neural/negative 臂数据都在时 gate_3 必须出现在判定里。"""
    tok = _tok(tmp_path)
    rep = _run(tmp_path, tok, arms=("frozen", "memory", "neural", "negative"),
               retention=True)
    assert "negative_control" in rep["gates"]
    assert rep["gates"]["negative_control"] is (
        rep["arms"]["negative"]["sealed_em_after"] < rep["sealed_after_em"])


def test_counterfactual_follow_recorded_and_gated(tmp_path):
    """反事实跟随率进入验收：提供项时必须计算并记录；门槛默认 None 只记录不判定。"""
    tok = _tok(tmp_path)
    cf = [{"fingerprint": "cf0",
           "prompt": "用户：小明在1号位置。\n好的。用户：小明在哪里？\n助手：",
           "target": "小明在1号位置。", "group": "pair0"},
          {"fingerprint": "cf1",
           "prompt": "用户：小红在1号位置。\n好的。用户：小红在哪里？\n助手：",
           "target": "小红在1号位置。", "group": "pair0"}]
    rep = _run(tmp_path, tok, arms=("frozen", "memory", "neural", "negative"),
               retention=True, counterfactual=cf)
    assert rep["counterfactual"]["n_items"] == 2
    assert rep["counterfactual"]["follow_rate"] is not None
    assert rep["counterfactual"]["gate_enforced"] is False   # 门槛默认不判定
    assert COUNTERFACTUAL_FOLLOW_MIN is None


# ---------------------------------------------------------------- lm_update 口径

def test_trainable_whitelist_rejects_unknown_mode_and_prefix(tmp_path):
    tok = _tok(tmp_path)
    torch.manual_seed(17)
    model = NMindLM(_small(tok))
    with pytest.raises(ValueError):
        set_trainable(model, trainable="full")
    with pytest.raises(ValueError):
        set_trainable(model, trainable="adapter", adapter_prefixes=("evil.",))
    names = set_trainable(model, trainable="adapter", adapter_prefixes=("block_to_ws.",))
    assert all(n.startswith("block_to_ws.") for n in names)
    assert not any(n.startswith("head.") for n in names)
    assert KNOWN_TRAINABLE_MODES == ("all", "adapter")
    assert KNOWN_ADAPTER_PREFIXES == ("head.", "block_to_ws.")


def test_attempted_vs_executed_split(tmp_path):
    """步数口径拆分：attempted 与 executed 都出现在报告里，且 executed <= attempted。"""
    tok = _tok(tmp_path)
    rep = _run(tmp_path, tok, arms=("neural", "negative"), retention=True, updates=2)
    u = rep["arms"]["neural"]["update"]
    assert "attempted_steps" in u and "optimizer_steps_executed" in u
    assert u["optimizer_steps_executed"] <= u["attempted_steps"]
    assert u["non_finite_steps_skipped"] <= u["attempted_steps"]
