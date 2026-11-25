"""阶段 2.4 步骤 1 数据审计补报（2026-11-25，AutoClaw）：对**实际训练数据**补算事实对重叠。

20261124 的 fact_audit_20261124.json 用的是另一条生成种子（20270205 的 generate_tasks），
而训练实际使用 make_train_test(20000, 600)（内部 seed 20270203/20270204）。本脚本用与
20261124 完全相同的"事实/事实对"定义（fact_audit.py 的 facts()/pairs()）对实际训练数据
补算，并如实说明"留出"的实际含义。只读分析，不训练、不改数据。
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from nmind.language.large_binding_task import make_train_test  # noqa: E402

OUT = ROOT / "experiments" / "phase2_baselines" / "fact_audit_train_20261125.json"
TOK = ROOT / "experiments" / "m5_language" / "tokenizer_m52.json"

_FACT_RE = re.compile(r"([^：。\n]+)在([1-4]号位置)")


def facts(rs):
    return {(e, p) for r in rs for e, p in _FACT_RE.findall(r["prompt"])}


def pairs(rs):
    return {tuple(sorted(_FACT_RE.findall(r["prompt"])[:2])) for r in rs}


def head(rs):
    return {r["prompt"].split("\n")[0] for r in rs}


def main() -> int:
    t0 = time.time()
    tr, te = make_train_test(20000, 600, str(TOK))
    P = pairs(tr)
    fh, th = head(tr), head(te)
    fp = {(e, p) for r in te for e, p in _FACT_RE.findall(r["prompt"])}
    out = {
        "definition_note": ("事实=(实体,位置)；事实对=每条 prompt 头部前两条陈述的搭配"
                             "（与 fact_audit_20261124.py 同一定义）；本审计对象是"
                             " make_train_test(20000,600) 的实际训练/封存数据"),
        "n_train": len(tr), "n_test": len(te),
        "facts": {"train": len(facts(tr)), "test": len(facts(te))},
        "fact_pairs": {"train": len(P), "test": len(pairs(te))},
        "test_pair_in_train_ratio": round(sum(p in P for p in pairs(te))
                                           / max(1, len(pairs(te))), 6),
        "test_fact_in_train_ratio": round(sum(p in facts(tr) for p in fp)
                                           / max(1, len(fp)), 6),
        "test_unique_statement_heads": len(th),
        "test_heads_in_train": len(th & fh),
        "test_heads_in_train_ratio": round(len(th & fh) / max(1, len(th)), 6),
        "combo_overlap_by_construction": 0,
        "holdout_meaning": ("make_train_test 的留出是 combo 级留出；combo key 含查询实体/"
                             "问法/确认语/重复变体，因此组合不相交 ≈ 样本级不相交。"
                             "但底层事实对的重叠（见 test_pair_in_train_ratio）说明"
                             "这不是严格事实对留出，更不是新实体泛化（实体只有 4 个）"),
        "wall_s": round(time.time() - t0, 1),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
