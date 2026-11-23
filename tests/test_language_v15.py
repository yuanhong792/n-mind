from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nmind.language.neural_language_v2 import LanguageV2Config, LanguageV2Model


def test_query_readout_is_opt_in_and_changes_parameter_count():
    base = LanguageV2Model(LanguageV2Config(vocab_size=4096))
    variant = LanguageV2Model(LanguageV2Config(vocab_size=4096, query_readout=True))
    assert base.query_readout is False
    assert variant.query_readout is True
    assert variant.n_params() > base.n_params()
    missing, unexpected = variant.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected
    assert set(missing) == {"query_readout_proj.weight", "query_readout_proj.bias"}


def test_query_readout_forward_shape_and_zero_optimizer_steps():
    model = LanguageV2Model(LanguageV2Config(vocab_size=4096, query_readout=True))
    model.eval()
    with torch.inference_mode():
        logits, state = model(torch.tensor([[260, 261, 262]]))
    assert logits.shape == (1, 3, 4096)
    assert state.h.shape[1] == model.v2_config.n_blocks
