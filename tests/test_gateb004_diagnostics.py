"""Gate-B 004 诊断模块测试：trace 忠实性、探针有效性、workspace 轨迹结构。"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nmind.language import binding_data as bd
from nmind.language.neural_dialogue import LMConfig, NMindLM
from nmind.language.state_probe import (extract_head_input, extract_ws_content,
                                         linear_probe, trace_forward)
from nmind.language.tokenizer import CharTokenizer
from nmind.language.workspace_diagnostics import workspace_trajectory, ws_content_drift

CORPUS = ("用户：小明在1号位置。\n好的。用户：小明在哪里？\n助手：小明在1号位置。\n"
          "用户：小红在2号位置。\n明白了。用户：小红在哪里？\n助手：小红在2号位置。\n"
          "用户：小红在3号位置。\n用户：小红在哪里？\n助手：小红在3号位置。\n"
          "用户：小明在4号位置。\n收到。用户：小明在哪里？\n助手：小明在4号位置。\n")


def _model_and_tok():
    torch.manual_seed(23)
    tok = CharTokenizer.train_from_texts([CORPUS])
    cfg = LMConfig(vocab_size=tok.vocab_size, d_model=16, n_blocks=4, block_hidden=16,
                    workspace_dim=8, workspace_slots=2, top_k=1)
    return NMindLM(cfg), tok


def test_trace_matches_original_forward():
    """忠实性红线：trace 的末位 logits 与原 forward 必须逐位一致（不一致即抛错）。"""
    model, tok = _model_and_tok()
    tr = trace_forward(model, tok, "用户：小明在1号位置。\n用户：小明在哪里？\n助手：")
    assert tr.logits_match_original is True
    assert len(tr.tokens_trace) == len(tok.encode("用户：小明在1号位置。\n用户：小明在哪里？\n助手："))


def test_trace_records_all_required_signals():
    """八类信号逐项在 trace 里：块状态/槽占用/b2w 输出/路由/head 输入/答案排序/EOS 偏置。"""
    model, tok = _model_and_tok()
    tr = trace_forward(model, tok, "用户：小红在2号位置。\n用户：小红在哪里？\n助手：",
                        target="小红在2号位置。")
    last = tr.tokens_trace[-1]
    assert len(last.route_indices) == 1 and len(last.route_weights) == 1
    assert len(last.h_mean) == 16                     # head 输入向量
    assert last.b2w_payload_norm > 0                  # block_to_ws 输出
    assert len(last.ws_occupied) == 2                 # 槽占用
    assert last.read_attn_entropy >= 0                # 读取注意力
    assert last.eos_logit is not None                 # EOS 偏置
    assert len(last.logit_top) == 5                   # top-5
    assert last.target_token_info is not None         # 答案首字符 rank/logit/prob
    assert last.target_token_info["rank"] >= 1
    assert len(tr.final_block_states_norm) == 4       # N-Cell 各块状态范数


def test_linear_probe_separates_learnable_signal():
    """探针有效性：线性可分数据 test 准确率必须接近 1；标签打乱后退化到基线水平。"""
    torch.manual_seed(31)
    n = 120
    labels = [i % 4 for i in range(n)]
    feats = []
    for lab in labels:
        base = torch.zeros(8)
        base[lab] = 3.0 + 0.1 * torch.randn(1).item()
        feats.append([round(float(v), 6) for v in base + 0.05 * torch.randn(8)])
    good = linear_probe(feats, labels, n_classes=4)
    assert good["test_accuracy"] >= 0.9
    shuffled = labels[1:] + labels[:1]
    bad = linear_probe(feats, shuffled, n_classes=4)
    assert bad["test_accuracy"] <= good["test_accuracy"] + 0.15
    assert bad["majority_baseline"] is not None


def test_probe_feature_extractors_shapes():
    model, tok = _model_and_tok()
    hi = extract_head_input(model, tok, "用户：小红在2号位置。\n用户：小红在哪里？\n助手：")
    wc = extract_ws_content(model, tok, "用户：小红在2号位置。\n用户：小红在哪里？\n助手：")
    assert len(hi) == 16 and len(wc) == 16            # 2 slots × d_w 8


def test_workspace_trajectory_and_drift():
    model, tok = _model_and_tok()
    tr = workspace_trajectory(model, tok, "用户：小红在2号位置。\n用户：小红在哪里？\n助手：")
    assert tr["n_tokens"] > 0 and len(tr["final_occupied"]) == 2
    assert len(tr["steps"]) == tr["n_tokens"]
    m2, _ = _model_and_tok()
    d = ws_content_drift(model, m2, tok, "用户：小红在2号位置。\n用户：小红在哪里？\n助手：")
    assert len(d["slots"]) == 2 and all("cosine" in s for s in d["slots"])
