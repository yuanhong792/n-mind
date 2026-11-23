"""V171 P1 regression: Best/Final save order + atomic interruption safety + Stable protection."""
from __future__ import annotations
import sys, hashlib
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
V17 = ROOT / "experiments/language_v1/v17"; sys.path.insert(0, str(V17))

from nmind.language.tokenizer_v2 import ByteBPETokenizer
from nmind.language.neural_language_v2 import load_language_v2, save_language_v2
from checkpoint_v17 import (capture_state, atomic_save, save_periodic,
                            save_final_and_best, sha256, FINAL_NAME, BEST_NAME, LAST_NAME)

LEGACY = [("用户：你好\n助手：","你好！"),("用户：你叫什么？\n助手：","我叫N-MIND。"),("用户：谢谢\n助手：","不客气！"),
          ("用户：小明有红色苹果\n助手：","小明的苹果是红色的。"),("用户：小红有绿色苹果\n助手：","小红的苹果是绿色的。"),
          ("用户：小刚有蓝色苹果\n助手：","小刚的苹果是蓝色的。"),("用户：小丽有黄色苹果\n助手：","小丽的苹果是黄色的。"),
          ("用户：我叫小明\n助手：","你好，小明！"),("用户：我叫什么？\n助手：","你叫小明。")]
PARENT = ROOT / "experiments/language_v1/v15/candidate_v15_baseline.pt"


@pytest.fixture(scope="module")
def tok():
    return ByteBPETokenizer.train([p + a for p, a in LEGACY], vocab_size=4096)


@pytest.fixture()
def model(tok):
    m, _ = load_language_v2(PARENT, tok)
    return m


def perturbed_state(base: dict, tag: float = 1.0) -> dict:
    """Simulate a genuine trained-step state: add a deterministic nonzero delta."""
    out = {}
    for i, (k, v) in enumerate(sorted(base.items())):
        delta = torch.randn_like(v.float()) * (0.01 * (1 + i % 3)) if v.is_floating_point() else torch.zeros_like(v)
        out[k] = (v.float() + delta).to(v.dtype) if v.is_floating_point() else v.clone()
    return out


def assert_state_equal(path, expected: dict, tok):
    m, _ = load_language_v2(path, tok)
    got = capture_state(m)
    assert set(got) == set(expected)
    for k in expected:
        assert torch.equal(got[k], expected[k]), f"tensor mismatch: {k}"


def test_best_final_different_steps_save_different_real_weights(model, tok, tmp_path):
    """1. Best/Final at different training steps must hold different genuine weights."""
    parent_state = capture_state(model)
    final_state = perturbed_state(parent_state)
    save_final_and_best(model, tok, tmp_path, final_state=final_state,
                        best_state=parent_state, final_step=1536, best_step=512)
    assert_state_equal(tmp_path / FINAL_NAME, final_state, tok)
    assert_state_equal(tmp_path / BEST_NAME, parent_state, tok)
    # the two saved files must NOT be weight-identical
    mf, _ = load_language_v2(tmp_path / FINAL_NAME, tok)
    mb, _ = load_language_v2(tmp_path / BEST_NAME, tok)
    assert any(not torch.equal(mf.state_dict()[k], mb.state_dict()[k]) for k in parent_state)


def test_best_step0_final_still_last_step(model, tok, tmp_path):
    """2. When best is step0 (=parent), FINAL must still be the last trained step."""
    parent_state = capture_state(model)
    trained = perturbed_state(parent_state)
    save_final_and_best(model, tok, tmp_path, final_state=trained,
                        best_state=parent_state, final_step=1536, best_step=0)
    assert_state_equal(tmp_path / FINAL_NAME, trained, tok)   # NOT parent
    assert_state_equal(tmp_path / BEST_NAME, parent_state, tok)
    _, bmeta = load_language_v2(tmp_path / BEST_NAME, tok)
    _, fmeta = load_language_v2(tmp_path / FINAL_NAME, tok)
    assert bmeta["steps"] == 0 and fmeta["steps"] == 1536


def test_reload_tensors_identical(model, tok, tmp_path):
    """3. After save, reloading reproduces every tensor exactly."""
    parent_state = capture_state(model)
    trained = perturbed_state(parent_state)
    save_final_and_best(model, tok, tmp_path, final_state=trained,
                        best_state=parent_state, final_step=100, best_step=60)
    for name, exp in [(FINAL_NAME, trained), (BEST_NAME, parent_state)]:
        assert_state_equal(tmp_path / name, exp, tok)
    assert sha256(tmp_path / FINAL_NAME) == hashlib.sha256((tmp_path / FINAL_NAME).read_bytes()).hexdigest()


def test_interrupted_run_keeps_last_valid(model, tok, tmp_path, monkeypatch):
    """4. A crash mid-save must leave the last successfully saved valid checkpoint intact."""
    step128 = capture_state(model)
    save_periodic(model, tok, tmp_path, step=128)              # valid LAST at step 128
    good_sha = sha256(tmp_path / LAST_NAME)
    # simulate crash: save_periodic writes a partial tmp then raises before replace
    import nmind.language.neural_language_v2 as nl2
    real_save = nl2.save_language_v2
    import checkpoint_v17 as cv
    def boom(*a, **k):
        p = Path(k.get("path", a[0] if a else tmp_path / "x")); p.write_bytes(b"partial")
        raise RuntimeError("simulated crash")
    monkeypatch.setattr(cv, "save_language_v2", boom)
    with pytest.raises(RuntimeError):
        save_periodic(model, tok, tmp_path, step=256)
    monkeypatch.setattr(cv, "save_language_v2", real_save)
    # original valid checkpoint still there and loadable, still step 128
    assert sha256(tmp_path / LAST_NAME) == good_sha
    m, meta = load_language_v2(tmp_path / LAST_NAME, tok)
    assert meta["steps"] == 128
    assert_state_equal(tmp_path / LAST_NAME, step128, tok)


def test_stable_never_overwritten(model, tok, tmp_path):
    """5. Stable weights must never be touched by the candidate save helpers."""
    stable = tmp_path / "stable_model.pt"
    save_language_v2(stable, model, tok, steps=999, manifest_hash="stable-immutable")
    stable_sha = sha256(stable)
    parent_state = capture_state(model)
    trained = perturbed_state(parent_state)
    save_final_and_best(model, tok, tmp_path, final_state=trained,
                        best_state=parent_state, final_step=30, best_step=10)
    save_periodic(model, tok, tmp_path, step=30)
    assert sha256(stable) == stable_sha, "stable_model.pt was modified!"


def test_no_stale_tmp_after_success(model, tok, tmp_path):
    """Successful atomic save leaves no *.tmp residue."""
    parent_state = capture_state(model)
    save_final_and_best(model, tok, tmp_path, final_state=parent_state,
                        best_state=parent_state, final_step=2, best_step=2)
    assert not list(tmp_path.glob("*.tmp"))
