"""V1.7.2 回归测试：BPE 协议、评分修复、模式/状态管理、Checkpoint 安全。

全部测试不执行 optimizer.step，不依赖真实父 checkpoint（缺失时使用同配置随机权重）。
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments/language_v1/v172"))

from nmind.language.neural_dialogue import LMConfig, NMindLM
from nmind.language.tokenizer_v2 import EOS
from v172_protocol import (build_tokenizer, conditional_token_lps, discriminative_positions,
                           encode_autoregressive, fresh_state, preserved_mode,
                           prompt_last_logits, valid_vocab)

CFG = LMConfig(vocab_size=4096, d_model=32, n_blocks=2, block_hidden=32,
               workspace_dim=16, workspace_slots=2, top_k=2)
PROMPT = "已知：小明有红色苹果，小红有绿色苹果。\n问：小明的苹果是什么颜色？\n答："


@pytest.fixture(scope="module")
def tok():
    return build_tokenizer()


@pytest.fixture
def model():
    torch.manual_seed(20261124)
    m = NMindLM(CFG)
    m.eval()
    m.use_router_load_update = False
    return m


def test_tokenizer_hash_frozen(tok):
    assert tok.vocab_hash() == "26af116ebd13ff00"


def test_segmented_concat_not_equal_joint_encode(tok):
    """协议层：分段拼接与联合编码可能不同；协议必须固定为前者。"""
    ts = encode_autoregressive(tok, "用户：你好", "你好！", audit_legacy=True)
    assert ts.sequence == tuple(ts.prompt_ids) + tuple(ts.answer_ids) + (EOS,)
    assert ts.answer_start == len(ts.prompt_ids)
    assert ts.eos_index == len(ts.sequence) - 1
    assert ts.target_start == len(ts.prompt_ids) - 1
    # 无论是否相同，sequence 必须来自分段拼接，绝不用 legacy
    assert isinstance(ts.legacy_sequence, tuple)


def test_target_mask_starts_at_prompt_last_token(tok):
    ts = encode_autoregressive(tok, PROMPT, "红色")
    assert ts.target_start == len(ts.prompt_ids) - 1
    # 监督 target 为 sequence[1:]，答案首 token 的 target 下标 = answer_start - 1
    labels = list(ts.sequence[1:])
    assert labels[ts.target_start] == ts.answer_ids[0]
    assert labels[ts.eos_index - 1] == EOS


def test_eos_position_and_count(tok):
    ts = encode_autoregressive(tok, PROMPT, "红色")
    assert ts.sequence.count(EOS) == 1
    assert ts.sequence[-1] == EOS


def test_teacher_forcing_equals_incremental_inference(model, tok):
    """一次整段 forward 的第 t 步 logits 必须等于逐 token 增量推理的 logits。"""
    ts = encode_autoregressive(tok, PROMPT, "红色")
    full = torch.tensor([list(ts.sequence)])
    with torch.inference_mode():
        logits_all, _ = model(full)
        state = model.initial_state(1, torch.device("cpu"))
        max_diff = 0.0
        for t, tid in enumerate(ts.sequence):
            lg, state = model(torch.tensor([[tid]]), state)
            max_diff = max(max_diff, float((lg[0, -1] - logits_all[0, t]).abs().max()))
    assert max_diff < 1e-4, f"teacher forcing 与增量推理不一致: {max_diff}"


def test_preserved_mode_restores_training_and_router_state(model):
    model.train()
    model.use_router_load_update = True
    model.router_gradient_path_enabled = True
    bias_before = model.router.load_bias.detach().clone()
    count_before = model.block_load_count.detach().clone()
    with preserved_mode(model):
        model.eval()
        model.use_router_load_update = False
        model.router_gradient_path_enabled = False
        with torch.no_grad():
            model.router.load_bias.add_(1.0)
            model.block_load_count.add_(1.0)
        assert not model.training
    assert model.training is True
    assert model.use_router_load_update is True
    assert model.router_gradient_path_enabled is True
    assert torch.equal(model.router.load_bias, bias_before)
    assert torch.equal(model.block_load_count, count_before)


def test_per_sample_state_isolated(model, tok):
    """同一模型连续处理两个无关 prompt，不得共享递归状态。"""
    p2 = "问：小红的苹果是什么颜色？答："
    with torch.inference_mode():
        with fresh_state(model, 1) as s1:
            l1, s1 = model(torch.tensor([tok.encode(PROMPT)]), s1)
        with fresh_state(model, 1) as s2:
            l2, s2 = model(torch.tensor([tok.encode(p2)]), s2)
        with fresh_state(model, 1) as s3:
            l3, s3 = model(torch.tensor([tok.encode(PROMPT)]), s3)
    assert torch.allclose(l1, l3, atol=1e-6)
    assert not torch.allclose(l1[0, -1], l2[0, -1], atol=1e-6)


def test_discriminative_position_gt_zero_invalidates_prompt_tail_proxy(model, tok):
    """判别位 > 0 时，旧 prompt 末尾 logprob 不等于正确的条件 logprob。"""
    cands = ["红色", "绿色", "蓝色", "黄色", "紫色", "橙色", "黑色", "白色"]
    enc, pos = discriminative_positions(tok, cands)
    valid, _ = valid_vocab(tok)
    gt0 = [c for c in cands if pos[c] > 0]
    assert gt0, "tokenizer 中应至少存在判别位非首 token 的候选"
    tail = prompt_last_logits(model, tok, PROMPT)
    for c in gt0:
        correct = conditional_token_lps(model, tok, PROMPT, c, valid=valid, start=pos[c])
        tid = enc[c][pos[c]]
        idx = int((valid == tid).nonzero(as_tuple=True)[0][0].item())
        old = float(torch.log_softmax(tail.index_select(0, valid), -1)[idx])
        assert abs(old - correct[0]) > 1e-6, f"{c} 的旧代理评分被错误复用"


def test_no_optimizer_step_in_dry_run():
    from train_v172 import dry_run_report
    rep = dry_run_report()
    assert rep["main_model_optimizer_steps"] == 0
    assert rep["parent_checkpoint_available"] is False
    assert set(rep["data_counts"]) == {"train", "val", "sealed", "counter"}


def test_checkpoint_saves_optimizer_step_and_recoverable_state(tmp_path, model, tok):
    from train_v172 import save_checkpoint
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    opt.step()  # 只为产生非空 optimizer state，不更新主模型权重
    p = tmp_path / "candidate.pt"
    sha = save_checkpoint(p, model, tok, step=7, seed=1, optimizer=opt,
                          rng_state=random.getstate())
    blob = torch.load(p, map_location="cpu", weights_only=False)
    assert blob["optimizer_steps_executed"] == 7
    assert blob["optimizer_state_dict"] is not None
    assert blob["rng_state"] is not None
    assert len(sha) == 64
