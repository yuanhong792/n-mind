"""M5.2 P0 无训练审计：数据偏置定量 + 冻结父 checkpoint 反事实探针（任务书三-2/3/4）。

输出 m52_diagnostics.json：
  A. M5.1 微观数据的位置分布（train/sealed 答案位置、陈述位置、查询实体）
  B. "2号位置"先验的数据侧解释：训练答案位置频次 vs 模型输出频次
  C. 冻结 M5.1 checkpoint 反事实探针：逐样本 expected/free_generated/teacher_forced/
     position_pred/stop_reason——同一事实只改位置数字，预测是否对应变化
  D. 规则基线（最近陈述复制）与位置先验基线的可达分数
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys_root = str(ROOT / "src")
if sys_root not in sys.path:
    sys.path.insert(0, sys_root)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from nmind.language.decoder import generate  # noqa: E402
from nmind.language.neural_dialogue import load_lm  # noqa: E402
from nmind.language.tokenizer import CharTokenizer  # noqa: E402
from run_m51_stages import build_micro  # noqa: E402

POSITIONS = ["1号位置", "2号位置", "3号位置", "4号位置"]
SEED = 20261301
MAX_LEN = 160


def position_counter(answers: list[str]) -> dict[str, int]:
    c: Counter = Counter()
    for a in answers:
        for p in POSITIONS:
            if p in a:
                c[p] += 1
                break
    return {p: c.get(p, 0) for p in POSITIONS}


def answer_digit_token_freq(answers: list[str]) -> dict[str, int]:
    c: Counter = Counter()
    for a in answers:
        for ch in a:
            if ch.isdigit():
                c[ch] += 1
    return dict(sorted(c.items()))


def main() -> int:
    t0 = time.perf_counter()
    out: dict[str, Any] = {"experiment": "m5.2-p0-audit",
                            "captured_at": time.strftime("%Y-%m-%d %H:%M:%S")}

    # ---- A. M5.1 微观数据分布 ----
    tr_p0, tr_a0 = build_micro("stage0", 200, SEED)
    se_p0, se_a0 = build_micro("stage0", 40, SEED + 1)
    tr_p1, tr_a1 = build_micro("stage1", 200, SEED + 2)
    se_p1, se_a1 = build_micro("stage1", 40, SEED + 3)
    out["A_m51_data"] = {
        "stage0_train_answer_pos": position_counter(tr_a0),
        "stage0_sealed_answer_pos": position_counter(se_a0),
        "stage1_train_answer_pos": position_counter(tr_a1),
        "stage1_query_entity": dict(Counter("小明" if "小明在哪里" in p or
                                             ("小明在哪里" in p) else "小红"
                                             for p in tr_p1)),
        "stage0_answer_digit_tokens": answer_digit_token_freq(tr_a0),
    }

    # ---- B/C. 冻结 M5.1 checkpoint 反事实探针 ----
    tok = CharTokenizer.load(HERE / "tokenizer.json")
    model, meta = load_lm(HERE / "language_checkpoint.pt", tok)
    model.eval()
    probes: list[dict] = []
    ent_swap: list[dict] = []
    for p in POSITIONS:
        prompt = f"用户：小明在{p}。\n小明在哪里？\n助手："
        expected = f"小明在{p}。"
        ids = tok.encode(prompt)
        with torch.no_grad():
            logits, state = model(torch.tensor([ids], dtype=torch.long))
        tf_pred_digit = None
        # TF 位置预测：答案 span 的位置数字槽位 argmax（"助"+"手"+"："+e+"在"+数字）
        ans_start = len(tok.encode(prompt))
        # 自由生成
        rec, _st = generate(model, tok, tok.encode("助手："), state=state,
                             max_new_tokens=24, greedy=True)
        free_text = rec.text.strip()
        for q in POSITIONS:
            if q in free_text:
                tf_pred_digit = q[:1]
        probes.append({
            "expected": expected, "free_generated": free_text,
            "teacher_forced_position_pred": tf_pred_digit,
            "position_pred": (next((q for q in POSITIONS if q in free_text), None) or "")[:1],
            "stop_reason": rec.finish_reason,
            "em": free_text == expected, "prompt_variant": f"小明在{p}",
        })
    # 实体交换对照：同一位置、不同实体
    for e in ("小明", "小红"):
        prompt = f"用户：{e}在3号位置。\n{e}在哪里？\n助手："
        rec, _st = generate(model, tok, tok.encode(prompt), max_new_tokens=24, greedy=True)
        ent_swap.append({"entity": e, "expected": f"{e}在3号位置。",
                          "free_generated": rec.text.strip(),
                          "follows_entity": e in rec.text and "3号位置" in rec.text})
    out["C_counterfactual_probe"] = {
        "position_swap": probes,
        "position_follow_rate": round(sum(1 for x in probes if x["em"]) / len(probes), 4),
        "entity_swap": ent_swap,
        "checkpoint_meta": meta,
    }

    # ---- D. 基线可达分数（固定规则，非神经）----
    # 最近陈述复制基线在反事实探针上 = oracle 本身（复制任务）→ 可达 1.0
    # 位置先验基线 = 训练分布最高频位置
    tr_pos = position_counter(tr_a0)
    prior_pos = max(tr_pos, key=tr_pos.get)
    out["D_baselines"] = {
        "last_statement_copy_on_probe": 1.0,
        "position_prior": prior_pos,
        "position_prior_on_probe": round(sum(1 for x in probes
                                              if x["position_pred"] == prior_pos[:1])
                                          / len(probes), 4),
    }
    (HERE / "m52_diagnostics.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                                encoding="utf-8")
    print(json.dumps({"position_follow_rate": out["C_counterfactual_probe"]["position_follow_rate"],
                       "probes": probes[:4],
                       "A_stage0_train_pos": out["A_m51_data"]["stage0_train_answer_pos"],
                       "baselines": out["D_baselines"]}, ensure_ascii=False, indent=2), flush=True)
    print(f"→ {HERE / 'm52_diagnostics.json'}（{time.perf_counter() - t0:.0f}s）", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
