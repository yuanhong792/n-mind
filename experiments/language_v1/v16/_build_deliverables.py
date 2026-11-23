"""V1.6：从冻结诊断结果生成正式交付 JSON。不训练、不更新权重。"""
from __future__ import annotations
import sys, json, hashlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

ROOT = Path(__file__).resolve().parents[3]; SRC = ROOT/"experiments/language_v1"; V15=SRC/"v15"; OUT=SRC/"v16"

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def dump(name, obj):
    (OUT/name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    print("written", name)

def main():
    raw = json.loads((OUT/"_diag_raw.json").read_text(encoding="utf-8"))
    shortcut = json.loads((OUT/"_diag_shortcut.json").read_text(encoding="utf-8"))
    router = json.loads((OUT/"_diag_router.json").read_text(encoding="utf-8"))

    # ===== V16_LOSS_BREAKDOWN.json =====
    def breakdown(rows):
        out = []
        for r in rows:
            sup = r["supervised_tokens"]
            key = [i for i in sup if i["target_id"] not in (2,)]  # 非EOS
            ans_first = sup[0] if sup else None
            out.append({
                "answer": r["answer"],
                "seq_len": r["seq_len"], "supervised_positions": r["supervised_positions"],
                "supervised_token_texts": r["supervised_texts"],
                "mean_loss_all_supervised": r["mean_loss"],
                "answer_first_token_ce": ans_first["ce"] if ans_first else None,
                "answer_first_token_text": ans_first["target_text"] if ans_first else None,
                "nons_eos_ce": [i["ce"] for i in key],
                "eos_ce": sup[-1]["ce"] if sup and sup[-1]["target_id"] == 2 else None,
                "per_token": [{"text": i["target_text"], "ce": i["ce"], "supervised": i["supervised"]} for i in r["all_tokens"]],
            })
        return out
    loss_doc = {
        "status": "FROZEN_EVAL_NO_OPTIMIZER_STEP",
        "model": raw["model"], "model_sha256": raw["model_sha256"],
        "tokenizer_hash": raw["tokenizer_hash"],
        "key_finding": "平均0.24 loss由答案首字(判别token, CE≈0.70≈ln2)与恒定的'色'/EOS(CE≈0.01)平均稀释而成; 0.24不代表学会颜色绑定",
        "train_set_breakdown": breakdown(raw["train_loss_breakdown"]),
        "sealed_set_breakdown": breakdown(raw["sealed_loss_breakdown"]),
        "category_loss": {
            "红色": [r["mean_loss"] for r in raw["train_loss_breakdown"] if r["answer"] == "红色"],
            "绿色": [r["mean_loss"] for r in raw["train_loss_breakdown"] if r["answer"] == "绿色"],
        },
    }
    dump("V16_LOSS_BREAKDOWN.json", loss_doc)

    # ===== V16_TEACHER_FORCING_AUDIT.json =====
    tf = {
        "status": "FROZEN_EVAL_NO_OPTIMIZER_STEP",
        "note": "三模式对比: A=Teacher-forced gold答案条件对数概率; B=自由生成; C=候选答案评分(完整答案条件对数概率)",
        "cases": raw["p2"],
        "finding": "5个case中红/绿候选logprob几乎恒定(红≈-0.36,绿≈-0.365), 实体/颜色/顺序交换均不反转排序; TF gold与候选评分一致说明评分口径正确; 模型本身对问题不敏感",
        "classification": "情形2(正确答案概率本身低/输出偏置)与情形4(模型对当前问题不敏感)同时成立; 非情形1(解码器选错), 非情形3(仅TF能预测)",
    }
    dump("V16_TEACHER_FORCING_AUDIT.json", tf)

    # ===== V16_FREE_GENERATION.jsonl =====
    lines = []
    for c in raw["p2"]:
        lines.append(json.dumps({"id": c["id"], "prompt": c["prompt"], "expected": c["expected"],
                                 "free_generation": c["free_generation"],
                                 "candidate_pred": c["candidate_pred"],
                                 "correct": c["candidate_correct"]}, ensure_ascii=False))
    (OUT/"V16_FREE_GENERATION.jsonl").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("written V16_FREE_GENERATION.jsonl")

    # ===== V16_ENTITY_BINDING_PROBE.json =====
    probe = {
        "status": "FROZEN_NO_PARAM_UPDATE",
        "state_trace": raw["p3"],
        "frozen_linear_probe": shortcut["frozen_probe"],
        "router_diagnostic": {
            "router_grad_abs_sum": router["router_grad_abs_sum"],
            "router_has_zero_grad_BEFORE_fix": router["router_has_zero_grad"],
            "h_mean_cosine_ming_vs_hong": router["h_mean_cosine_ming_vs_hong"],
            "h_mean_l2_diff": router["h_mean_l2_diff"],
            "final_logit_cosine": router["final_logit_cosine"],
            "final_logit_l2_diff": router["final_logit_l2_diff"],
        },
        "finding": "问小明vs小红: h_mean余弦0.999995、最终logit余弦0.999999, 实体信息未改变表示与读出; 修复前Router梯度为0(STE权重未接线, 见FIX_VALIDATION); 冻结线性探针仅50%, 表示的实体可解码性未被证明",
        "caveat": "线性探针成绩只证明表示可解码性, 不计入N-MIND自主语言能力",
    }
    dump("V16_ENTITY_BINDING_PROBE.json", probe)

    # ===== V16_DATA_SHORTCUT_AUDIT.json =====
    ds = {
        "status": "AUDIT",
        "train_answer_frequency": shortcut["shortcut_audit"]["answer_frequency"],
        "fixed_red_baseline_accuracy": shortcut["shortcut_audit"]["fixed_red_accuracy_on_train"],
        "fixed_green_baseline_accuracy": shortcut["shortcut_audit"]["fixed_green_accuracy_on_train"],
        "query_template_count": shortcut["shortcut_audit"]["query_template_count"],
        "answer_length_tokens": shortcut["shortcut_audit"]["answer_length_tokens"],
        "first_fact_color_vs_answer": shortcut["shortcut_audit"]["first_fact_color_vs_answer"],
        "checking_sha256": {
            "candidate_v15_baseline.pt": sha(V15/"candidate_v15_baseline.pt"),
            "candidate_v15_query_attention.pt": sha(V15/"candidate_v15_query_attention.pt"),
            "candidate_v14_main.pt": sha(SRC/"candidate_v14_main.pt"),
            "V15_TRAINING_RESULTS.json": sha(V15/"V15_TRAINING_RESULTS.json"),
        },
        "finding": "训练集红/绿完全平衡(2:2), 固定红/绿均只有50%基线; 模型固定输出红色是自身偏置而非数据偏斜; 但首字判别信号弱(CE≈ln2)+训练样本仅4条是根本限制",
    }
    dump("V16_DATA_SHORTCUT_AUDIT.json", ds)

if __name__ == "__main__":
    main()
