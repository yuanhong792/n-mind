"""V171 P5: EOS退化审计。检查立即EOS、监督位置、decoder最小Token对照。"""
from __future__ import annotations
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments/language_v1/v171"))
import torch
from nmind.language.tokenizer_v2 import EOS
import v171_protocol as P

V17 = ROOT / "experiments/language_v1/v17"
V171 = ROOT / "experiments/language_v1/v171"


def main():
    torch.set_num_threads(1)
    tok = P.build_tokenizer(); valid, lookup = P.build_vocab(tok); enc, fd = P.discriminative_positions(tok)

    # 1) V1.7 历史评测的立即EOS（旧口径把 eos+空串 当成合法）
    old = json.load(open(V17 / "V17_EVALUATION.json", encoding="utf-8"))
    od = old.get("details", [])
    old_eos = sum(1 for x in od if x.get("finish") == "eos")
    old_empty = sum(1 for x in od if not (x.get("free") or "").strip())
    old_eos_empty = sum(1 for x in od if x.get("finish") == "eos" and not (x.get("free") or "").strip())
    old_claim = old.get("summary", {})

    # 2) 新协议公平评测（若已完成）
    fair_path = V171 / "V171_FAIR_COMPARISON.json"
    new_results = {}
    if fair_path.exists():
        fair = json.load(open(fair_path, encoding="utf-8"))
        for name, d in fair.get("models", {}).items():
            ov = d["evaluation"]["overall"]
            new_results[name] = {
                "n": ov["n"],
                "immediate_eos_rate": ov.get("immediate_eos_rate"),
                "immediate_eos_count": round((ov.get("immediate_eos_rate") or 0) * ov["n"]),
                "non_empty_answer_rate": ov.get("non_empty_answer_rate"),
                "valid_utf8_non_empty_rate": ov.get("valid_utf8_non_empty_rate"),
                "correct_eos_rate": ov.get("correct_eos_rate"),
                "utf8_complete_rate": ov.get("utf8_complete_rate"),
                "free_acc": ov.get("free_acc"),
                "finish_reason_counts": ov.get("finish_reason_counts"),
            }

    # 3) EOS 监督位置核查（源码静态 + 动态掩码）
    sys.path.insert(0, str(V17))
    import train_v17 as T
    row = {"prompt": "用户：你好\n助手：", "answer": "你好！", "task": "A_dialogue"}
    x, mask, tw = T.make_batch(tok, [row], [0], enc, fd)
    seq = tok.encode(row["prompt"] + row["answer"], add_eos=True)
    eos_target_idx = len(seq) - 2   # x[:,1:] 中 EOS 的 target 位置
    eos_in_mask = bool(mask[0, eos_target_idx])
    eos_weight = float(tw[0, eos_target_idx])

    audit = {
        "purpose": "V171 P5 EOS degeneracy audit",
        "root_cause": ("V1.7 旧评测把 finish_reason=eos 当作 UTF-8 合法与成功结束, 但 592/600 "
                       "样本是首步即 EOS 的空串输出; 模型(实际=父模型)未学会在答案后结束, "
                       "而是把 EOS 当成对任意 prompt 的默认响应。"),
        "historical_v17_old_metric": {
            "n": len(od),
            "eos_count": old_eos,
            "empty_free_count": old_empty,
            "eos_and_empty_count": old_eos_empty,
            "immediate_eos_rate_old_criteria": round(old_eos_empty / len(od), 4) if od else None,
            "old_claimed_utf8_legal_rate": {k: v.get("utf8_legal_rate") for k, v in old_claim.items()},
            "old_claimed_eos_rate": {k: v.get("eos_rate") for k, v in old_claim.items()},
            "problem": "旧指标把立即EOS空串计为 utf8 合法 + eos 成功, 掩盖退化。",
        },
        "unified_protocol_results": new_results,
        "eos_supervision_check": {
            "eos_token_id": EOS,
            "eos_target_in_answer_mask": eos_in_mask,
            "eos_loss_weight": eos_weight,
            "mask_loop_upper_bound": "range(ans_start, len(s))  # V1.7 修复后含 EOS",
            "conclusion": ("EOS 已进入监督掩码(低权重 0.25)。但权重文件实际=父模型, "
                           "1536 步训练权重丢失, 无法验证训练是否学会正确 EOS 位置; "
                           "当前观察到的立即 EOS 是父模型本身的行为。"),
        },
        "decoder_min_token_control": {
            "main_evaluation_uses_min_tokens": False,
            "note": ("主评测未启用任何最小输出 Token 规则; 若需用最小 Token 抑制立即EOS, "
                     "只能作为单独对照配置, 不得伪装为模型能力提升。本轮不启用。"),
        },
        "rules_used_to_emit_correct_answers": False,
    }
    (V171 / "V171_EOS_AUDIT.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: audit[k] for k in ("historical_v17_old_metric", "eos_supervision_check")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
