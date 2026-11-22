"""候选权重晋升：SEALED_EVALUATE → PASS 才 PROPOSED（首次人工批准）/ FAIL → ROLLBACK。

稳定 checkpoint 永不被原地覆盖：candidate 只写入独立文件；失败即删除候选并
记录审计（model_promotion_audit.jsonl）。rollback 前后都会核对 stable 哈希。
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

CANDIDATE_STATUSES = ("PROPOSED", "REJECTED", "INCONCLUSIVE", "APPROVED")


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_gates(matrix: dict[str, Any], *, b_ood_gain_threshold: float = 0.10,
                a_drop_max: float = 0.03) -> dict[str, Any]:
    """任务书第 7 节起始门槛（预登记，不得看结果后暗改）。

    matrix 需含：inheritance_ok、frozen_violations、neural B_OOD_gain（最佳神经臂）、
    frozen_baseline_B_OOD_gain、negative_B_OOD_gain、A_forgetting（最佳神经臂）、
    neural_beats_frozen。
    """
    gates = {
        "inheritance_and_frozen_ok": bool(matrix.get("inheritance_ok"))
        and int(matrix.get("frozen_violations", 1)) == 0,
        "b_ood_gain_ge_threshold": matrix.get("b_ood_gain") is not None
        and matrix["b_ood_gain"] >= b_ood_gain_threshold,
        "beats_negative_control": matrix.get("b_ood_gain") is not None
        and matrix.get("negative_b_ood_gain") is not None
        and matrix["b_ood_gain"] > matrix["negative_b_ood_gain"],
        "a_drop_within_limit": matrix.get("a_forgetting") is not None
        and matrix["a_forgetting"] <= a_drop_max,
        "neural_beats_frozen": bool(matrix.get("neural_beats_frozen")),
    }
    return {"gates": gates, "all_pass": all(gates.values()),
            "thresholds": {"b_ood_gain_pp": b_ood_gain_threshold * 100,
                            "a_drop_max_pp": a_drop_max * 100}}


def decide_status(gates_result: dict[str, Any], *, ran_all_arms: bool) -> str:
    if not ran_all_arms:
        return "INCONCLUSIVE"
    return "PROPOSED" if gates_result["all_pass"] else "REJECTED"


def append_audit(path: str | Path, record: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                             **record}, ensure_ascii=False) + "\n")


def rollback_candidate(candidate_path: str | Path, *, stable_path: str | Path,
                       stable_sha_before: str, reason: str) -> dict[str, Any]:
    """删除候选权重并核对稳定版本未被触碰。"""
    c = Path(candidate_path)
    existed = c.exists()
    if existed:
        c.unlink()
    stable_sha_after = file_sha256(stable_path)
    return {"action": "ROLLBACK", "reason": reason, "candidate_existed": existed,
            "candidate_deleted": existed,
            "stable_sha256_before": stable_sha_before,
            "stable_sha256_after": stable_sha_after,
            "stable_unchanged": stable_sha_after == stable_sha_before}
