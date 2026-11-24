"""M3.1 P0：训练监督协议审计与双协议定义（零 optimizer.step）。

M3 现状：evaluate_m3.make_batch 对 prompt 位置也赋权 1.0，属于 Full-sequence；
且把 2.0 的首答案权重放在 w[sep_idx+1]（该项对应目标 s[sep_idx+1]，即第一个答案 token，
对齐本身正确），但 EOS 权重为 0.25 而非 1.0，归一化是 batch 全局而非按样本。

本协议实现两种显式协议：
  A Answer-only：仅答案 token + EOS 参与监督，prompt/SEP/PAD 权重 0。
  B Full-sequence：除首 token 与 PAD 外全部参与（M3 近似此协议，EOS 权重不同）。

关键索引约定：seq = prompt + [SEP] + answer + [EOS]；
logits[:, t] 预测 target[:, t] = seq[t+1]，故权重数组 w[j] 对应目标 seq[j]，
第一个答案 token 的位置是 sep_idx+1，EOS 是 len(seq)-1。

运行：python m3_protocol_repair.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parents[1]
M3 = HERE.parent / "math_m3"
sys.path.insert(0, str(M3))

from tokenizer_m3 import M3Tokenizer, SEP, EOS, PAD  # noqa: E402


def protocol_weights(seq: list[int], protocol: str) -> tuple[list[float], dict]:
    """返回与 target=seq[1:] 对齐的权重（长度 len(seq)）及监督位置清单。"""
    if protocol not in ("A", "B"):
        raise ValueError("protocol must be A or B")
    L = len(seq)
    sep = seq.index(SEP)
    if sep < 1 or sep >= L - 2:
        raise ValueError(f"SEP 位置非法: sep={sep}, L={L}")
    w = [0.0] * L
    supervised = []
    if protocol == "A":
        # 答案 token + EOS；w[j] 对应目标 seq[j] = logits 在 j-1 的预测
        for j in range(sep + 1, L):
            w[j] = 1.0
            supervised.append({"target_index": j, "target_id": seq[j],
                               "role": "eos" if j == L - 1 else "answer"})
    else:
        for j in range(1, L):
            w[j] = 1.0
            supervised.append({"target_index": j, "target_id": seq[j],
                               "role": "eos" if j == L - 1 else ("sep" if j == sep else "prompt_or_answer")})
    return w, {"sep_index": sep, "answer_start": sep + 1, "eos_index": L - 1,
               "n_supervised": len(supervised), "supervised": supervised}


def m3_legacy_weights(seq: list[int]) -> tuple[list[float], dict]:
    """复刻 M3 evaluate_m3.make_batch 的权重（用于对比，不推荐）。"""
    L = len(seq)
    sep = seq.index(SEP)
    w = [0.0] * L
    for j in range(L):
        w[j] = 0.25 if j == L - 1 else 1.0
    if sep + 1 < L - 1:
        w[sep + 1] = 2.0
    return w, {"sep_index": sep, "n_supervised": sum(1 for x in w[1:] if x > 0)}


def per_sample_normalized_loss(logits: torch.Tensor, target: torch.Tensor,
                               weights: torch.Tensor) -> torch.Tensor:
    """按样本有效监督权重归一化，再对 batch 取平均（M4 Protocol A 要求）。"""
    V = logits.shape[-1]
    ce = torch.nn.functional.cross_entropy(
        logits.reshape(-1, V), target.reshape(-1), reduction="none").reshape(target.shape)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return ((ce * weights).sum(dim=1) / denom).mean()


def main():
    tok = M3Tokenizer()
    report = {
        "status": "COMPLETE", "protocol": "zero_optimizer_step_audit",
        "tokenizer_sha": tok.vocab_sha,
        "conventions": {"sequence": "prompt+[SEP]+answer+[EOS]",
                        "weight_index": "w[j] 对应目标 seq[j]，logits 在 j-1 处预测"},
        "splits": {},
    }
    for split in ("train", "val", "sealed"):
        rows = [json.loads(l) for l in (M3 / "data" / f"m3_{split}.jsonl")
                .read_text(encoding="utf-8").splitlines() if l.strip()]
        stats = {"n": len(rows), "truncated_at_64": 0, "sep_missing": 0,
                 "answer_missing": 0, "eos_missing": 0, "pad_in_supervision": 0,
                 "m3_prompt_supervised_positions": 0, "a_supervised_positions": 0,
                 "a_supervised_tokens": 0, "m3_eos_weight": None, "a_eos_weight": None}
        checked = 0
        for r in rows:
            seq, meta = tok.encode_pair(r["prompt"], r["answer"])
            if len(seq) > 64:
                stats["truncated_at_64"] += 1
                seq = seq[:64]
            if SEP not in seq:
                stats["sep_missing"] += 1
                continue
            if meta["answer_len"] == 0:
                stats["answer_missing"] += 1
                continue
            if seq[-1] != EOS:
                stats["eos_missing"] += 1
                continue
            wa, info = protocol_weights(seq, "A")
            wl, _ = m3_legacy_weights(seq)
            stats["a_supervised_positions"] += info["n_supervised"]
            stats["a_supervised_tokens"] += sum(1 for j, x in enumerate(wa) if x > 0 and seq[j] != EOS)
            stats["m3_prompt_supervised_positions"] += sum(
                1 for j in range(1, info["sep_index"] + 1) if wl[j] > 0)
            if any(seq[j] == PAD for j, x in enumerate(wa) if x > 0):
                stats["pad_in_supervision"] += 1
            stats["m3_eos_weight"] = wl[-1]
            stats["a_eos_weight"] = wa[-1]
            # 严格对齐断言：SEP 位置的 logits 必须预测第一个答案 token
            sep = info["sep_index"]
            assert wa[sep] == 0.0, "SEP 作为目标不应在 Answer-only 中监督"
            assert wa[sep + 1] == 1.0 and seq[sep + 1] == tok.encode(r["answer"])[0], \
                "SEP 后一位必须是第一个答案 token 且被监督"
            assert wa[-1] == 1.0 and seq[-1] == EOS, "EOS 必须作为被监督目标"
            checked += 1
        stats["checked"] = checked
        stats["m3_prompt_supervised_ratio"] = round(
            stats["m3_prompt_supervised_positions"] /
            max(1, stats["m3_prompt_supervised_positions"] + stats["a_supervised_positions"]), 4)
        report["splits"][split] = stats

    # 数值一致性：Protocol A 的按样本归一化 loss 与手工计算一致（零梯度前向）
    torch.manual_seed(0)
    B, T, V = 3, 12, 102
    logits = torch.randn(B, T - 1, V)
    target = torch.randint(4, 100, (B, T - 1))
    weights = torch.zeros(B, T)
    for i in range(B):
        weights[i, 5:] = 1.0
    got = per_sample_normalized_loss(logits, target, weights[:, 1:]).item()
    ce = torch.nn.functional.cross_entropy(logits.reshape(-1, V), target.reshape(-1), reduction="none")
    expected = ((ce.reshape(B, T - 1) * weights[:, 1:]).sum(1) / weights[:, 1:].sum(1)).mean().item()
    report["loss_normalization_check"] = {"computed": round(got, 8), "expected": round(expected, 8),
                                          "max_abs_diff": abs(got - expected), "pass": abs(got - expected) < 1e-6}
    out = HERE / "M31_PROTOCOL_AUDIT.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[protocol] checked={sum(s['checked'] for s in report['splits'].values())} "
          f"M3 prompt 监督占比={report['splits']['train']['m3_prompt_supervised_ratio']}")
    print(f"[protocol] loss 归一化一致性 pass={report['loss_normalization_check']['pass']}")
    print(f"[protocol] 写出 {out}")


if __name__ == "__main__":
    main()
