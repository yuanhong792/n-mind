"""V1.7.2 P3/P4/P7：BPE 边界一致性、评分口径与 EOS 审计。

- 扫描真实 v171 数据集统计旧口径 encode(prompt+answer) 与新口径
  分段编码拼接的差异样本数。
- 用随机同配置模型证明"prompt 末尾 logits 当答案第 k 位概率"在
  判别位 > 0 时是错的（旧 evaluate_v17.py 的 first_token_lp_all）。
- 汇总历史 EOS 审计证据，区分立即 EOS / 合法完成 / 正确回答。
不更新主模型权重，不执行正式训练。
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from nmind.language.bpe_decoder import effective_token_ids
from nmind.language.neural_dialogue import LMConfig, NMindLM
from nmind.language.tokenizer_v2 import EOS
from v172_protocol import (build_tokenizer, conditional_token_lps, discriminative_positions,
                           discriminative_proxy, encode_autoregressive, full_answer_conditional,
                           prompt_last_logits, valid_vocab, write_json)

OUT = Path(__file__).resolve().parent
DATA = ROOT / "experiments/language_v1/v171/data"


def load_rows(split: str):
    p = DATA / f"v171_{split}.jsonl"
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def audit_bpe_boundary(tok) -> dict:
    splits = {}
    for split in ("train", "val", "sealed", "counter"):
        rows = load_rows(split)
        affected = []
        for i, r in enumerate(rows):
            ts = encode_autoregressive(tok, r["prompt"], r["answer"], audit_legacy=True)
            if list(ts.legacy_sequence) != list(ts.sequence):
                affected.append({
                    "index": i, "task": r["task"],
                    "legacy_len": len(ts.legacy_sequence), "fixed_len": len(ts.sequence),
                    "first_divergence": next((j for j, (a, b) in enumerate(
                        zip(ts.legacy_sequence, ts.sequence)) if a != b), -1),
                })
        splits[split] = {
            "total": len(rows),
            "affected": len(affected),
            "affected_ratio": round(len(affected) / max(1, len(rows)), 6),
            "examples": affected[:5],
        }
    # 静态扫描：哪些入口仍用旧拼接
    legacy_sites = []
    for p in sorted(ROOT.glob("experiments/language_v1/**/*.py")):
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for m in re.finditer(r".{0,90}encode\([^\n]*prompt[^\n]*\+[^\n]*answer[^\n]*\).{0,60}", text):
            legacy_sites.append({"file": str(p.relative_to(ROOT)), "line": text[:m.start()].count("\n") + 1,
                                 "snippet": m.group(0).strip()[:200]})
    return {
        "status": "BPE_ALIGNMENT_AUDIT_DONE",
        "protocol_fixed": "prompt_ids + answer_ids + [EOS] (segmented encode, then concat)",
        "protocol_legacy": "tok.encode(prompt + answer, add_eos=True)",
        "tokenizer_hash": tok.vocab_hash(),
        "tokenizer_unchanged": True,
        "splits": splits,
        "total_affected": sum(v["affected"] for v in splits.values()),
        "total_rows": sum(v["total"] for v in splits.values()),
        "legacy_call_sites_found": legacy_sites,
        "checkpoint_modified": False,
    }


def audit_scoring(tok, cfg: LMConfig) -> dict:
    """证明判别位非首位时旧快速评分给出错误答案概率。"""
    torch.manual_seed(20261124)
    model = NMindLM(cfg)
    model.eval()
    model.use_router_load_update = False
    valid, _ = valid_vocab(tok)
    candidates = ["红色", "绿色", "蓝色", "黄色", "紫色", "橙色", "黑色", "白色"]
    enc, positions = discriminative_positions(tok, candidates)
    # 找到判别位 > 0 的候选，构造 prompt 使该候选为金标
    pos_gt0 = {c: p for c, p in positions.items() if p > 0}
    rows = []
    prompt = "已知：小明有红色苹果，小红有绿色苹果。\n问：小明的苹果是什么颜色？\n答："
    cached = prompt_last_logits(model, tok, prompt)

    for cand in candidates:
        aids = enc[cand]
        pos = positions[cand]
        correct_lps = conditional_token_lps(model, tok, prompt, cand, valid=valid, start=pos)
        old_lp = None
        if pos < len(aids):
            idx = int((valid == aids[pos]).nonzero(as_tuple=True)[0][0].item())
            old_lp = float(torch.log_softmax(cached.index_select(0, valid), -1)[idx])
        new_lp = correct_lps[0] if correct_lps else None
        rows.append({
            "candidate": cand,
            "answer_token_ids": aids,
            "discriminative_position": pos,
            "legacy_prompt_tail_logprob": round(old_lp, 6) if old_lp is not None else None,
            "correct_conditional_logprob": round(new_lp, 6) if new_lp is not None else None,
            "abs_error": round(abs(old_lp - new_lp), 6) if old_lp is not None and new_lp is not None else None,
            "legacy_proxy_valid": pos == 0,
        })

    # full answer conditional 必须含 EOS
    fa = full_answer_conditional(model, tok, prompt, "红色", valid=valid)
    fa_no_eos = sum(conditional_token_lps(model, tok, prompt, "红色", valid=valid))
    ts = encode_autoregressive(tok, prompt, "红色")

    return {
        "status": "EVALUATION_FIX_DONE",
        "metric_names": ["binding_accuracy", "counterfactual_pair_accuracy", "position_accuracy",
                         "dialogue_accuracy", "context_accuracy", "free_generation_accuracy",
                         "immediate_eos_rate", "full_answer_conditional_accuracy"],
        "proxy_names_distinguished": ["DISCRIMINATIVE_TOKEN_PROXY", "FIRST_TOKEN_PROXY_LEGACY"],
        "candidates_with_discriminative_position_gt_zero": pos_gt0,
        "per_candidate": rows,
        "legacy_proxy_invalid_count": sum(1 for r in rows if not r["legacy_proxy_valid"]),
        "full_answer_conditional_with_eos": round(fa, 6),
        "full_answer_conditional_without_eos": round(fa_no_eos, 6),
        "eos_included": ts.sequence[-1] == EOS,
        "best_checkpoint_rule": "fixed VAL metric only; SEALED never used for selection",
        "denominator_rule": "parent/candidate/final scored on identical fixed split and candidate set",
        "teacher_forced_prob_is_not_free_generation": True,
        "weights_source": "random_init_same_config_ARCHITECTURE_ONLY",
        "parent_checkpoint_available": False,
    }


def audit_eos() -> dict:
    """从历史证据汇总 EOS 问题，不重跑 600 次生成（父权重缺失）。"""
    sources = {}
    for name in ("V171_EOS_AUDIT.json", "V17_EVALUATION.json", "V16_TEACHER_FORCING_AUDIT.json",
                 "V16_LOSS_BREAKDOWN.json", "V17_GENERATION.jsonl"):
        p = ROOT / "experiments/language_v1" / ("v171/" if name.startswith("V171") else
                                                "v17/" if name.startswith("V17") else "v16/") / name
        if p.exists():
            if p.suffix == ".jsonl":
                lines = p.read_text(encoding="utf-8").splitlines()
                counts = {"total": len(lines), "empty_text": 0, "finish_eos": 0}
                for line in lines[:2000]:
                    try:
                        d = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    counts["empty_text"] += int(not str(d.get("generated", d.get("text", ""))).strip())
                    counts["finish_eos"] += int(d.get("finish_reason") == "eos")
                sources[name] = counts
            else:
                sources[name] = json.loads(p.read_text(encoding="utf-8"))

    return {
        "status": "EOS_AUDIT_DONE",
        "historical_claim": "600 generations, 592 immediate EOS (from V1.7.1 report)",
        "historical_evidence_files": sources,
        "categories_distinguished": ["immediate_eos", "normal_completion_eos",
                                     "utf8_legal_empty_output", "nonempty_legal_output",
                                     "actually_correct_answer"],
        "eos_supervision_position": "sequence[-1], target = sequence[-2] -> EOS",
        "eos_weight_in_v17": 0.25,
        "discriminative_weight_in_v17": 2.0,
        "known_engineering_cause": "V16_LOSS_BREAKDOWN: suffix 色 CE≈0.017, EOS CE≈0.009, discriminative first token CE≈0.70",
        "diagnostic_control_allowed": "ban-first-token-EOS is diagnostic only, not a substitute for free generation accuracy",
        "note": "历史父权重缺失，未重跑 600 次生成；本审计基于已落盘证据与源码监督位置。",
    }


def main():
    torch.set_num_threads(1)
    tok = build_tokenizer()
    cfg = LMConfig(vocab_size=4096, d_model=32, n_blocks=2, block_hidden=32,
                   workspace_dim=16, workspace_slots=2, top_k=2)
    bpe = audit_bpe_boundary(tok)
    scoring = audit_scoring(tok, cfg)
    eos = audit_eos()
    write_json(OUT / "V172_BPE_ALIGNMENT_AUDIT.json", bpe)
    write_json(OUT / "V172_EVALUATION_FIX.json", scoring)
    write_json(OUT / "V172_EOS_AUDIT.json", eos)
    print(json.dumps({"bpe_affected": bpe["total_affected"], "bpe_total": bpe["total_rows"],
                      "legacy_invalid_proxies": scoring["legacy_proxy_invalid_count"],
                      "eos_files": list(eos["historical_evidence_files"].keys())},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
