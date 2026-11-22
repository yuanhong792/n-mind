from nmind.language.gate1 import build_gate1
from nmind.language.tokenizer_v2 import TokenizerV2
from nmind.language.baseline_v2 import measure_untrained

def test_gate1_manual_corpus_is_offline_and_leakage_free():
    result = build_gate1("fixtures/language_v2/gate1_manual_zh.txt")
    assert result["license"] == "user-authored-explicit-research-consent"
    assert result["network_bytes"] == 0 and result["optimizer_steps"] == 0
    assert result["leakage_free"] is True
    assert sum(len(x) for x in result["sets"].values()) >= 10

def test_gate1_samples_roundtrip():
    result = build_gate1("fixtures/language_v2/gate1_manual_zh.txt")
    tok = TokenizerV2()
    for rows in result["sets"].values():
        for row in rows:
            assert tok.decode(tok.encode(row["text"])) == row["text"]

def test_untrained_baseline_is_measured_without_steps_or_answer_leakage():
    result = measure_untrained()
    assert result["optimizer_steps"] == 0
    assert result["answer_leakage"] is False
    assert len(result["tasks"]) == 3
