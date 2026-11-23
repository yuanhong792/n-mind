"""V1.6: audit TRAIN/SEALED/COUNTER independence + fixed answer baseline.
Read-only on historical data; creates a NEW independent counterfactual set (pre-registered).
ZERO optimizer steps."""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

# Verbatim from train_v15.py lines 12-14
TRAIN = [("用户：小明有红色苹果，小红有绿色苹果。\n用户：小明的苹果是什么颜色？\n助手：","红色"),("用户：小明有红色苹果，小红有绿色苹果。\n用户：小红的苹果是什么颜色？\n助手：","绿色"),("用户：小明有绿色苹果，小红有红色苹果。\n用户：小明的苹果是什么颜色？\n助手：","绿色"),("用户：小明有绿色苹果，小红有红色苹果。\n用户：小红的苹果是什么颜色？\n助手：","红色")]
SEALED = [("用户：小刚有蓝色书包，小丽有黄色书包。\n用户：小丽的书包是什么颜色？\n助手：","黄色"),("用户：小刚有黄色杯子，小丽有蓝色杯子。\n用户：小刚的杯子是什么颜色？\n助手：","黄色"),("用户：小丽有绿色苹果，小明有蓝色苹果。\n用户：小明的苹果是什么颜色？\n助手：","蓝色"),("用户：小红有黄色书包，小刚有绿色书包。\n用户：小红的书包是什么颜色？\n助手：","黄色")]
COUNTER = [("用户：小明有红色苹果，小红有绿色苹果。\n用户：小明的苹果是什么颜色？\n助手：","红色"),("用户：小明有红色苹果，小红有绿色苹果。\n用户：小红的苹果是什么颜色？\n助手：","绿色"),("用户：小明有绿色苹果，小红有红色苹果。\n用户：小明的苹果是什么颜色？\n助手：","绿色"),("用户：小明有绿色苹果，小红有红色苹果。\n用户：小红的苹果是什么颜色？\n助手：","红色")]

def fp(rows):
    canon = json.dumps([[p, a] for p, a in rows], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()

def prompts(rows): return [p for p, _ in rows]
def ans_counts(rows):
    d = {}
    for _, a in rows: d[a] = d.get(a, 0) + 1
    return d

r = {}
r["fingerprints"] = {"TRAIN": fp(TRAIN), "SEALED": fp(SEALED), "COUNTER": fp(COUNTER)}
r["sizes"] = {"TRAIN": len(TRAIN), "SEALED": len(SEALED), "COUNTER": len(COUNTER)}
# internal dedup
r["internal_duplicate_prompts"] = {
    "TRAIN": len(TRAIN) - len(set(prompts(TRAIN))),
    "SEALED": len(SEALED) - len(set(prompts(SEALED))),
    "COUNTER": len(COUNTER) - len(set(prompts(COUNTER))),
}
# cross-set prompt overlap
def overlap(a, b): return sorted(set(prompts(a)) & set(prompts(b)))
r["cross_overlap_count"] = {
    "TRAIN_vs_SEALED": len(overlap(TRAIN, SEALED)),
    "TRAIN_vs_COUNTER": len(overlap(TRAIN, COUNTER)),
    "SEALED_vs_COUNTER": len(overlap(SEALED, COUNTER)),
}
r["TRAIN_vs_COUNTER_identical"] = (TRAIN == COUNTER)
r["counter_is_independent"] = (len(overlap(TRAIN, COUNTER)) == 0 and TRAIN != COUNTER)
r["counter_independence_note"] = (
    "历史 V1.5 的 COUNTER 与 TRAIN 逐条完全相同（4/4 重合），"
    "不能作为独立未见反事实测试；已创建新的独立集 V16_COUNTERFACTUAL。"
)
# answer frequency
r["answer_frequency"] = {"TRAIN": ans_counts(TRAIN), "SEALED": ans_counts(SEALED), "COUNTER": ans_counts(COUNTER)}
all_eval = SEALED + COUNTER
r["answer_frequency_eval_combined"] = ans_counts(all_eval)
# fixed answer baselines on eval combined (8 rows)
n = len(all_eval)
fc = ans_counts(all_eval)
r["fixed_answer_baseline"] = {a: round(c / n, 4) for a, c in fc.items()}
r["best_fixed_answer"] = max(r["fixed_answer_baseline"], key=r["fixed_answer_baseline"].get)
r["best_fixed_answer_accuracy"] = max(r["fixed_answer_baseline"].values())

# NEW independent counterfactual set: unseen persons + unseen object + answer colors are seen;
# 4 rows forming 2 counterfactual PAIRS (swap colors, same question -> answer must flip).
CF = [
    ("用户：小强有红色笔，小美有黄色笔。\n用户：小强的笔是什么颜色？\n助手：", "红色"),
    ("用户：小强有黄色笔，小美有红色笔。\n用户：小强的笔是什么颜色？\n助手：", "黄色"),
    ("用户：小强有红色笔，小美有黄色笔。\n用户：小美的笔是什么颜色？\n助手：", "黄色"),
    ("用户：小强有黄色笔，小美有红色笔。\n用户：小美的笔是什么颜色？\n助手：", "红色"),
]
cf_rows = [{"prompt": p, "answer": a} for p, a in CF]
r["new_independent_counterfactual"] = {
    "file": "experiments/language_v1/v16/V16_COUNTERFACTUAL.json",
    "sha256": fp(CF),
    "rows": cf_rows,
    "n_pairs": 2,
    "design": "未见人物(小强/小美)+未见物品(笔)，交换两人物颜色后问同一人，答案必须翻转；答案色(红/黄)为训练已见词",
    "overlap_with_TRAIN_prompts": len(overlap(TRAIN, CF)),
    "overlap_with_SEALED_prompts": len(overlap(SEALED, CF)),
    "fixed_answer_baseline": {a: round(c/len(CF), 4) for a, c in ans_counts(CF).items()},
    "pre_registered_before_training": True,
    "mutable_after_freeze": False,
}
# write independent CF file (immutable pre-registration)
cf_path = ROOT / "experiments/language_v1/v16/V16_COUNTERFACTUAL.json"
cf_path.write_text(json.dumps({
    "status": "PRE_REGISTERED_FROZEN_BEFORE_TRAINING",
    "sha256": fp(CF),
    "created_at": "2026-11-23",
    "purpose": "独立反事实跟随评测；不复用历史 COUNTER(与TRAIN完全重合)",
    "rows": cf_rows,
}, ensure_ascii=False, indent=2), encoding="utf-8")
# verify written file hash matches declared
r["new_independent_counterfactual"]["written_file_sha256"] = fp([(x["prompt"], x["answer"]) for x in json.loads(cf_path.read_text(encoding="utf-8"))["rows"]])

r["verdict"] = {
    "historical_data_preserved": True,
    "counter_unusable_as_independent": True,
    "new_independent_cf_created": True,
    "fixed_answer_baseline_on_eval": r["best_fixed_answer_accuracy"],
}
out = ROOT / "experiments/language_v1/v16/V16_DATA_SPLIT_AUDIT.json"
out.write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({
    "fingerprints": r["fingerprints"],
    "TRAIN_vs_COUNTER_identical": r["TRAIN_vs_COUNTER_identical"],
    "cross_overlap_count": r["cross_overlap_count"],
    "best_fixed_answer": r["best_fixed_answer"],
    "best_fixed_answer_accuracy": r["best_fixed_answer_accuracy"],
    "new_CF_sha256": fp(CF),
    "new_CF_overlap_TRAIN": r["new_independent_counterfactual"]["overlap_with_TRAIN_prompts"],
    "new_CF_file_hash_match": r["new_independent_counterfactual"]["sha256"] == r["new_independent_counterfactual"]["written_file_sha256"],
}, ensure_ascii=False, indent=2))
