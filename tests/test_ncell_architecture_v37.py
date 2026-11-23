"""V3.7 opt-in architecture guards: no neural training and no teacher/API calls."""
from __future__ import annotations

import torch
import torch.nn.functional as F
import pytest

from nmind.core.ncell import NCell
from nmind.language.neural_dialogue import LMConfig, NMindLM, save_lm, load_lm
from nmind.language.tokenizer import CharTokenizer


def _cfg(**extra):
    cfg = dict(vocab_size=24, d_model=16, n_blocks=4,
               block_hidden=16, workspace_dim=8, workspace_slots=2, top_k=2)
    cfg.update(extra)
    return LMConfig(**cfg)


def test_legacy_configuration_keeps_exact_forward_and_state_dict():
    torch.manual_seed(12)
    baseline = NMindLM(_cfg()).eval()
    torch.manual_seed(12)
    explicit = NMindLM(_cfg(use_reset_gate=False, use_router_ste=False,
                            readout_mode="mean")).eval()
    x = torch.tensor([[4, 6, 8, 9], [5, 7, 8, 10]])
    with torch.no_grad():
        a, sa = baseline(x)
        b, sb = explicit(x)
    assert torch.equal(a, b)
    assert torch.equal(sa.h, sb.h)
    assert list(baseline.state_dict()) == list(explicit.state_dict())
    assert not any("readout_" in k or "W_r" in k or "b_r" in k
                   for k in baseline.state_dict())


def test_optional_reset_gate_dense_sparse_equivalent():
    torch.manual_seed(21)
    cell = NCell(8, 8, 5, 4, use_reset_gate=True).eval()
    z = torch.randn(3, 8)
    h = torch.randn(3, 4, 8)
    w = torch.randn(3, 5)
    idx = torch.tensor([[0, 1], [1, 3], [0, 2]])
    active = torch.zeros(3, 4).scatter_(1, idx, 1.0)
    dense, *_ = cell(z, h, w, active)
    sparse, *_ = cell.forward_sparse(z, h, w, idx)
    assert torch.allclose(dense, sparse, atol=1e-6)
    assert torch.equal(sparse[active == 0], h[active == 0])


def test_reset_gate_and_attention_readout_have_real_gradients():
    torch.manual_seed(7)
    m = NMindLM(_cfg(use_reset_gate=True, readout_mode="query_attention")).train()
    m.use_router_load_update = False
    x = torch.tensor([[4, 5, 6, 7], [8, 9, 10, 11]])
    logits, _ = m(x)
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, 24),
                           x[:, 1:].reshape(-1))
    loss.backward()
    for name in ("blocks.W_r", "readout_query.weight", "readout_key.weight",
                 "readout_ws.weight", "head.weight"):
        g = dict(m.named_parameters())[name].grad
        assert g is not None and torch.isfinite(g).all(), name
        assert g.abs().sum().item() > 0, name


def test_router_ste_allows_task_loss_gradient_in_sparse_language_path():
    torch.manual_seed(31)
    hard = NMindLM(_cfg(use_router_ste=False)).train()
    soft = NMindLM(_cfg(use_router_ste=True)).train()
    soft.load_state_dict(hard.state_dict())
    hard.use_router_load_update = False
    soft.use_router_load_update = False
    x = torch.tensor([[4, 7, 6, 5], [8, 10, 9, 11]])
    h_out, _ = hard(x)
    s_out, _ = soft(x)
    # Straight-through routing preserves the exact numerical forward.
    assert torch.allclose(h_out, s_out, atol=1e-6)
    F.cross_entropy(s_out[:, :-1].reshape(-1, 24),
                    x[:, 1:].reshape(-1)).backward()
    g = soft.router.score_mlp[0].weight.grad
    assert g is not None and torch.isfinite(g).all()
    assert g.abs().sum().item() > 0
    F.cross_entropy(h_out[:, :-1].reshape(-1, 24),
                    x[:, 1:].reshape(-1)).backward()
    gh = hard.router.score_mlp[0].weight.grad
    assert gh is None or gh.abs().sum().item() == 0


def test_opt_in_checkpoint_roundtrip_and_legacy_checkpoint(tmp_path):
    torch.manual_seed(8)
    tok = CharTokenizer.train_from_texts(["小明在左边。", "小红在右边。", "助手：好的。"])
    cfg = LMConfig(vocab_size=tok.vocab_size, d_model=16, n_blocks=4,
                   block_hidden=16, workspace_dim=8, workspace_slots=2,
                   top_k=2, use_reset_gate=True, use_router_ste=True,
                   readout_mode="query_attention")
    model = NMindLM(cfg).eval()
    p = tmp_path / "candidate.pt"
    save_lm(p, model=model, tokenizer=tok, cfg=cfg, steps=0,
            seed=8, parent_sha="random", manifest_hash="test")
    reloaded, meta = load_lm(p, tok)
    assert reloaded.cfg.to_dict() == cfg.to_dict()
    x = torch.tensor([tok.encode("小明在左边。")])
    with torch.no_grad():
        assert torch.equal(model(x)[0], reloaded(x)[0])
    assert meta["steps"] == 0
    # Old nmind-lm/1 checkpoints lack the three opt-in config keys.
    base = NMindLM(LMConfig(vocab_size=tok.vocab_size, d_model=16, n_blocks=4,
                             block_hidden=16, workspace_dim=8,
                             workspace_slots=2, top_k=2)).eval()
    old_blob = torch.load(p, weights_only=True)
    old_blob["state_dict"] = base.state_dict()
    for key in ("use_reset_gate", "use_router_ste", "readout_mode"):
        old_blob["config"].pop(key)
    legacy = tmp_path / "old.pt"
    torch.save(old_blob, legacy)
    older, _ = load_lm(legacy, tok)
    assert older.cfg.readout_mode == "mean"
    with torch.no_grad():
        assert torch.equal(base(x)[0], older(x)[0])


def test_invalid_architecture_config_is_rejected():
    with pytest.raises(ValueError):
        _cfg(top_k=5)
    with pytest.raises(ValueError):
        _cfg(readout_mode="magic")
