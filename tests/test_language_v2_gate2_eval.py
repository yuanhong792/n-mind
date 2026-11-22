from nmind.language.gate2_dataset import build_gate2_dataset
from nmind.language.gate2_evaluator import evaluate_checkpoint
from nmind.language.neural_language_v2 import LanguageV2Config, LanguageV2Model, save_language_v2
from nmind.language.tokenizer_v2 import TokenizerV2

def test_evaluator_reports_neural_source_and_pairs(tmp_path):
    tok=TokenizerV2(); model=LanguageV2Model(LanguageV2Config()); path=tmp_path/"x.pt"; save_language_v2(path,model,tok)
    result=evaluate_checkpoint(path, build_gate2_dataset()["sealed"][:4])
    assert result["response_source"] == "neural" and result["counterfactual_pairs"] == 0
