import hashlib
import json
import torch
import pytest

from nmind.language.tokenizer_v2 import TokenizerV2, PAD, BOS, EOS, UNK, BYTE_OFFSET
from nmind.language.data_v2 import build_counterfactual_pairs, manifest
from nmind.language.neural_language_v2 import LanguageV2Config, LanguageV2Model, save_language_v2, load_language_v2
from nmind.language.train_v2 import dry_run

def test_utf8_roundtrip_and_no_unknowns():
    tok = TokenizerV2()
    for text in ["", "中文，标点。🙂", "path/C:\\数据\n下一行", "𠀀"]:
        ids = tok.encode(text, add_bos=True, add_eos=True)
        assert tok.decode(ids) == text
        assert all(i >= BYTE_OFFSET or i in (BOS, EOS) for i in ids)
        assert UNK not in ids

def test_tokenizer_persistence_and_tamper_rejection(tmp_path):
    tok = TokenizerV2(); path = tmp_path / "tok.json"; tok.save(path)
    assert TokenizerV2.load(path).vocab_hash() == tok.vocab_hash()
    blob = json.loads(path.read_text(encoding="utf-8")); blob["vocab_size"] += 1
    path.write_text(json.dumps(blob), encoding="utf-8")
    with pytest.raises(ValueError): TokenizerV2.load(path)

def test_counterfactual_manifest_is_80_pairs():
    cases = build_counterfactual_pairs()
    assert len(cases) == 160
    assert manifest(cases)["pair_count"] == 80
    assert all(c.answer == cases[i + 1].answer for i, c in enumerate(cases) if i % 2 == 0)

def test_random_model_forward_does_not_mutate_parameters():
    torch.manual_seed(7); tok = TokenizerV2(); model = LanguageV2Model(LanguageV2Config())
    model.eval()
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    logits, _ = model(torch.tensor([tok.encode("中文")]))
    assert logits.shape[-1] == tok.vocab_size
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())

def test_dry_run_has_zero_optimizer_steps():
    result = dry_run()
    assert result["optimizer_steps_executed"] == 0
    assert result["attempted_steps"] == 0

def test_language_v2_checkpoint_roundtrip(tmp_path):
    tok = TokenizerV2(); model = LanguageV2Model(LanguageV2Config())
    path = tmp_path / "v2.pt"
    save_language_v2(path, model=model, tokenizer=tok, steps=0, seed=1, manifest_hash="x")
    loaded, meta = load_language_v2(path, tok)
    assert loaded.cfg.vocab_size == 260 and meta["steps"] == 0
