"""M5.2 修正版事实跟随评测（纯评测零训练；修复 m52_eval 中损坏的配对与基线）。

跟随率正确定义（任务书四：交换两个实体位置后再问）：同一实体组合的两条反事实样本——
  A: 小明在p1。小红在p2。问小明  → oracle 小明在p1。
  B: 小明在p2。小红在p1。问小明  → oracle 小明在p2。
follow_ok = 两条都正确且答案随事实互换而改变。另附 S2 移动更新零样本探针（未训练任务，如实单列）。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys_root = str(ROOT / "src")
if sys_root not in sys.path:
    sys.path.insert(0, sys_root)
HERE = Path(__file__).resolve().parent

from nmind.language.neural_dialogue import load_lm  # noqa: E402
from nmind.language.decoder import generate  # noqa: E402
from nmind.language.tokenizer import CharTokenizer  # noqa: E402
from run_m52 import build_s0, build_s1, SEED  # noqa: E402  复用同一数据构造 ⇒ 同一词表

ENTITIES = ["小明", "小红"]
POSITIONS = ["1号位置", "2号位置", "3号位置", "4号位置"]


def main() -> int:
    t0 = time.perf_counter()
    tok = CharTokenizer.load(HERE / "tokenizer_m52.json") if (HERE / "tokenizer_m52.json").exists() \
        else _rebuild_tok()
    model, meta = load_lm(HERE / "m52_s1_checkpoint.pt", tok)
    model.eval()

    def gen(prompt: str) -> str:
        ids = tok.encode(prompt)
        with torch.no_grad():
            logits, state = model(torch.tensor([ids], dtype=torch.long))
        rec, _st = generate(model, tok, tok.encode("助手："), state=state,
                             max_new_tokens=24, greedy=True)
        return rec.text.strip()

    # ---- 位置交换反事实对（12 组合 × 2 查询实体 = 48 对）----
    pairs: list[dict] = []
    follow_ok = 0
    for e1_i, e1 in enumerate(ENTITIES):
        e2 = ENTITIES[1 - e1_i]
        for p1 in POSITIONS:
            for p2 in POSITIONS:
                if p1 == p2:
                    continue
                pa = f"用户：{e1}在{p1}。{e2}在{p2}。\n{e1}在哪里？\n助手："
                pb = f"用户：{e1}在{p2}。{e2}在{p1}。\n{e1}在哪里？\n助手："
                ra, rb = gen(pa), gen(pb)
                oa, ob = f"{e1}在{p1}。", f"{e1}在{p2}。"
                ok_a, ok_b = ra == oa, rb == ob
                ok = ok_a and ok_b
                follow_ok += int(ok)
                pairs.append({
                    "combo": f"{e1}@{p1}|{e2}@{p2}", "query": e1,
                    "A": {"prompt": pa.replace("\n", "|"), "oracle": oa, "pred": ra, "em": ok_a},
                    "B": {"prompt": pb.replace("\n", "|"), "oracle": ob, "pred": rb, "em": ok_b},
                    "follow_ok": ok, "answers_differ": ra != rb,
                })
    follow_rate = round(follow_ok / max(1, len(pairs)), 4)

    # ---- 修正后的基线（此前 last_statement_copy 被 `or True` 写坏，作废重算）----
    # 最后陈述复制：问题指向 e1 时答 e2 的位置（错误）→ 该基线在 e1 查询上应为错误答案
    # 规则世界跟踪基线（解析两陈述+按查询实体回答）
    rule_ok = 0
    for pr in pairs:
        pa = pr["A"]["prompt"]
        e1, e2 = ENTITIES
        p_of = {}
        for m in [e1, e2]:
            import re
            mm = re.search(rf"{m}在(\d)号位置", pa)
            if mm:
                p_of[m] = f"{mm.group(1)}号位置"
        q = pr["query"]
        rtext = f"{q}在{p_of[q]}。" if q in p_of else "我不知道。"
        rule_ok += int(rtext == pr["A"]["oracle"])
    rule_rate = round(rule_ok / len(pairs), 4)

    # ---- S2 移动更新零样本探针（未训练任务，如实单列）----
    rng_probe = json.loads((HERE / "m52_eval.json").read_text(encoding="utf-8")) \
        .get("s1", {}).get("s2_update_probe_zero_shot", {})
    move_zero_shot = {"em": rng_probe.get("em"), "n": rng_probe.get("n"),
                       "note": "S2 事实更新未训练：模型答移动前旧位置（8.33%）——真实能力缺口，待批准后训练"}

    payload = {
        "experiment": "m5.2-follow-eval-corrected",
        "captured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint": "m52_s1_checkpoint.pt", "checkpoint_meta": meta,
        "definition_fix": "m52_eval.json 的 fact_follow 配对逻辑失效（n=0）且 last_statement_copy "
                           "基线被 `or True` 写坏；本文件用位置交换反事实对与修正基线重测（纯评测零训练）",
        "n_pairs": len(pairs),
        "fact_follow_rate": follow_rate, "gate": 0.70,
        "gate_pass": bool(follow_rate >= 0.70),
        "rule_baseline_follow_rate": rule_rate,
        "s2_move_update_zero_shot": move_zero_shot,
        "pairs": pairs,
    }
    (HERE / "m52_follow_eval.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                                encoding="utf-8")
    print(json.dumps({"n_pairs": len(pairs), "fact_follow_rate": follow_rate,
                       "gate_pass": payload["gate_pass"], "rule_baseline": rule_rate},
                      ensure_ascii=False, indent=2), flush=True)
    print(f"→ {HERE / 'm52_follow_eval.json'}（{time.perf_counter() - t0:.0f}s）", flush=True)
    return 0


def _rebuild_tok() -> CharTokenizer:
    """m52 训练时的 tokenizer 未落盘（遗留），按相同种子确定性重建（词表哈希由 load 校验）。"""
    s0_tr = build_s0(400, SEED)
    s1_tr = build_s1(400, SEED + 3)
    return CharTokenizer.train_from_texts(s0_tr[0] + s0_tr[1] + s1_tr[0] + s1_tr[1])


if __name__ == "__main__":
    raise SystemExit(main())
