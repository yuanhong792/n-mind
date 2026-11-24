"""M41F 反事实/距离评测集隔离生成器（零梯度）。

生成：
- m4_counterfactual_val.jsonl / m4_counterfactual_sealed.jsonl
- m4_distance_val.jsonl / m4_distance_sealed.jsonl

同一 pair_id 及其全部配对变体只进入一个集合；跨集合 canonical 重合必须为 0。
输出 M41F_EVAL_SPLIT_AUDIT.json；不覆盖历史 m4_counterfactual.jsonl / m4_distance.jsonl。
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def load(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def write(path, rows):
    text = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n"
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def assign(family):
    return "sealed" if int(hashlib.sha256(family.encode("utf-8")).hexdigest(), 16) % 5 == 0 \
        else "val"


def audit():
    cf = load(DATA / "m4_counterfactual.jsonl")
    dist = load(DATA / "m4_distance.jsonl")

    def dist_family(row):
        q = next((line.strip()[4:].strip() for line in row["prompt"].splitlines()
                  if line.strip().startswith("GET ")), "?")
        return f"{q}|{row['answer']}"

    cf_by = {"val": [], "sealed": []}
    dist_by = {"val": [], "sealed": []}
    for p in cf:
        cf_by[assign(p["pair_id"])].append(p)
    for r in dist:
        # 按目标事实族整体划分，保证所有距离变体同进同出。
        dist_by[assign("DIST_FAMILY|" + dist_family(r))].append(r)

    hashes = {}
    hashes["m4_counterfactual_val.jsonl"] = write(
        DATA / "m4_counterfactual_val.jsonl", cf_by["val"])
    hashes["m4_counterfactual_sealed.jsonl"] = write(
        DATA / "m4_counterfactual_sealed.jsonl", cf_by["sealed"])
    hashes["m4_distance_val.jsonl"] = write(
        DATA / "m4_distance_val.jsonl", dist_by["val"])
    hashes["m4_distance_sealed.jsonl"] = write(
        DATA / "m4_distance_sealed.jsonl", dist_by["sealed"])

    def cf_canons(rows):
        return {r["canonical"] for p in rows for key in ("base", "counterfactual", "query_switch")
                for r in p.get(key, [])}

    def row_canons(rows):
        return {r["canonical"] for r in rows}

    cf_val_ids = {p["pair_id"] for p in cf_by["val"]}
    cf_sealed_ids = {p["pair_id"] for p in cf_by["sealed"]}
    d_val_ids = {r["meta"]["pair_id"] for r in dist_by["val"]}
    d_sealed_ids = {r["meta"]["pair_id"] for r in dist_by["sealed"]}

    checks = {
        "CF_PAIR_ID_ISOLATION": not (cf_val_ids & cf_sealed_ids),
        "CF_CANONICAL_ISOLATION": not (cf_canons(cf_by["val"]) & cf_canons(cf_by["sealed"])),
        "DISTANCE_PAIR_ID_ISOLATION": not (d_val_ids & d_sealed_ids),
        "DISTANCE_CANONICAL_ISOLATION": not (
            row_canons(dist_by["val"]) & row_canons(dist_by["sealed"])),
        "CF_PAIR_COVERAGE": len(cf_by["val"]) + len(cf_by["sealed"]) == len(cf),
        "DISTANCE_COVERAGE": len(dist_by["val"]) + len(dist_by["sealed"]) == len(dist),
    }

    out = {
        "status": "PASS" if all(checks.values()) else "FAILED",
        "rule": "sha256(family) % 5 == 0 -> sealed; otherwise val; CF family=pair_id; distance family=target_fact(GET var|answer)",
        "counts": {
            "counterfactual_val_pairs": len(cf_by["val"]),
            "counterfactual_sealed_pairs": len(cf_by["sealed"]),
            "distance_val_rows": len(dist_by["val"]),
            "distance_sealed_rows": len(dist_by["sealed"]),
        },
        "checks": checks,
        "file_sha256": hashes,
        "usage": {
            "val": "允许参与能力门槛与模型选择",
            "sealed": "只能在所有训练与选模完成后评估",
        },
        "note": "历史 m4_counterfactual.jsonl / m4_distance.jsonl 保留作为全量证据。",
    }
    p = HERE / "M41F_EVAL_SPLIT_AUDIT.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[m41f-split] status={out['status']} counts={out['counts']} 写出 {p}")
    return out


if __name__ == "__main__":
    audit()
