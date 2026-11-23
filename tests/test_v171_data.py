"""V171 P4 regression: fact-level dedup + counterfactual pairing + quota control + V1.7 immutability."""
from __future__ import annotations
import sys, json, hashlib
from pathlib import Path
from collections import defaultdict, Counter

import pytest

ROOT = Path(__file__).resolve().parents[1]
V171_DIR = ROOT / "experiments/language_v1/v171"
V17_DIR = ROOT / "experiments/language_v1/v17"
sys.path.insert(0, str(V171_DIR))
import dataset_v171 as D


@pytest.fixture(scope="module")
def data():
    final, audit = D.build()
    return final, audit


def test_no_prompt_overlap_across_splits(data):
    final, audit = data
    names = list(final)
    for i, a in enumerate(names):
        for b in names[i+1:]:
            pa = {r["prompt"] for r in final[a]}
            pb = {r["prompt"] for r in final[b]}
            assert not (pa & pb), f"prompt overlap {a}/{b}: {len(pa & pb)}"


def test_fact_level_isolation(data):
    final, audit = data
    key_splits = defaultdict(set)
    for s, rows in final.items():
        for r in rows:
            key_splits[r["key"]].add(s)
    crossing = [k for k, v in key_splits.items() if len(v) > 1]
    assert crossing == [], f"facts crossing splits: {crossing[:5]}"


def test_counterfactual_true_pairs(data):
    final, audit = data
    for s in ("train", "val", "sealed", "counter"):
        cm = defaultdict(list)
        for r in final[s]:
            if r["task"] == "C_counter":
                cm[r["key"]].append(r)
        valid = 0
        for k, items in cm.items():
            if len(items) == 2 and items[0]["answer"] != items[1]["answer"]:
                roles = {x["meta"].get("pair_role") for x in items}
                assert roles == {"orig", "swap"}, f"{s}/{k} missing pair roles: {roles}"
                valid += 1
        assert valid > 0, f"{s} has no valid counterfactual pairs"
    # sealed 至少 50 对, counter 至少 100 对
    assert audit["counterfactual_pairs"]["sealed"]["valid_pairs"] >= 50
    assert audit["counterfactual_pairs"]["counter"]["valid_pairs"] >= 100


def test_task_quota_controllable(data):
    final, audit = data
    # 多轮上下文与反事实不再是 V1.7 的 16/300 极少分布
    assert audit["task_counts"]["train"]["E_context"] >= 500
    assert audit["task_counts"]["sealed"]["E_context"] >= 100
    assert audit["task_counts"]["train"]["C_counter"] >= 1000
    assert audit["task_counts"]["sealed"]["C_counter"] >= 100


def test_v17_original_data_immutable():
    """V1.7 数据文件 SHA 必须与冻结审计一致（本轮未覆盖）。"""
    audit = json.load(open(V17_DIR / "V17_DATA_AUDIT.json", encoding="utf-8"))
    for s in ("train", "val", "sealed", "counter"):
        p = V17_DIR / f"v17_{s}.jsonl"
        assert hashlib.sha256(p.read_bytes()).hexdigest() == audit[f"{s}_sha256"], \
            f"V1.7 {s} data was modified!"


def test_frozen_v171_sha():
    """V1.7.1 数据文件已冻结, SHA 与交付审计一致。"""
    audit = json.load(open(V171_DIR / "V171_DATA_AUDIT.json", encoding="utf-8"))
    ddir = V171_DIR / "data"
    for s in ("train", "val", "sealed", "counter"):
        p = ddir / f"v171_{s}.jsonl"
        assert p.exists()
        assert hashlib.sha256(p.read_bytes()).hexdigest() == audit[f"{s}_sha256"], \
            f"V171 {s} frozen SHA mismatch"


def test_distinct_person_first_chars():
    assert len({p[0] for p in D.PERSONS}) == len(D.PERSONS)
