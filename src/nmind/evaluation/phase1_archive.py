"""Large deterministic sealed-set generation and disjointness audit."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from ..language.binding_data import build_family, fingerprint_set, BindingTask
import itertools
import hashlib

def _expanded_position_archive(n, seed, split="sealed"):
    entities = ["小明", "小红", "小刚", "小丽", "小周", "小吴", "小郑", "小陈", "小林", "小赵"]
    positions = [f"{i}号位置" for i in range(1, 11)]
    asks = ["在哪里？", "的位置在哪里？", "现在什么地方？", "在哪个位置？"]
    rows = []
    for e, p, ask, d in itertools.islice(itertools.product(entities, positions, asks, ("", "好的。", "明白了。", "收到。")), n):
        stmt = f"{e}在{p}。"; prompt = f"用户：{stmt}{d}用户：{e}{ask}\n助手："; target = stmt
        fp = hashlib.sha256(f"{prompt}→{target}".encode()).hexdigest()[:16]
        rows.append(BindingTask("position_qa", split, prompt, target, f"用户：{stmt}", f"{e}{ask}",
                                tuple(sorted({"entity":e,"position":p,"query_entity":e,"ack":d}.items())), fp))
    return rows


def generate_archive(*, families, n_each=500, seed=20262401, forbidden=None):
    forbidden = set(forbidden or ())
    result, overlap = {}, 0
    for i, family in enumerate(families):
        # Builders are deterministic; distinct seed bands prevent learning-flow reuse.
        rows = (_expanded_position_archive(n_each, seed + i)
                if family == "position_qa" else build_family(family, "sealed", n=n_each, seed=seed + i))
        result[family] = rows
        overlap += len(fingerprint_set(rows) & forbidden)
    all_fp = sorted(t.fingerprint for rows in result.values() for t in rows)
    fingerprint = hashlib.sha256("\n".join(all_fp).encode()).hexdigest()[:16]
    return result, {"seed": seed, "n_each": n_each, "overlap_count": overlap,
                    "n_total": len(all_fp), "data_fingerprint": fingerprint}


def write_manifest(path: str | Path, audit: dict):
    Path(path).write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
