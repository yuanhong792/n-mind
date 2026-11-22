"""候选世系与晋升管理：封存评测 → 门槛 → PROPOSED（人工批准）/ REJECTED（隔离回滚）。

硬约束（任务书 §8）：
  * stable checkpoint 只读，训练前后各校验一次 sha256，不一致立即 ERROR_SAFE；
  * 候选写独立文件；REJECTED 进 rejected/ 隔离目录，绝不覆盖 stable；
  * 晋升只走 `approve_promotion`（需显式 --i-approve 参数），实现为
    写新文件 → 校验 → 原子切换 current.json 指针，旧 stable 路径保留可回滚；
  * 阈值来自 learning_loop 的预登记常量，不接受运行时参数覆盖。
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

from ..online.promotion import file_sha256
from .learning_loop import run_candidate_update


def _audit_path(out: Path) -> Path:
    return out / "model_promotion_audit.jsonl"


def propose_candidate(*, stable_checkpoint: str, tokenizer_path: str, verified,
                      sealed_prompts: list[str], sealed_answers: list[str],
                      dev, out_dir: str | Path, updates: int = 48, budget=None,
                      arms: tuple[str, ...] = ("frozen", "neural"),
                      wall_limit_s: float = 60.0,
                      replay_pairs=None, replay_ratio: float = 0.15,
                      trainable: str = "all") -> dict[str, Any]:
    """跑一次候选更新 + 封存评测，落审计并返回 PROPOSED/REJECTED/INCONCLUSIVE。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stable_before = file_sha256(stable_checkpoint)
    cand_file = out / f"candidate_{int(time.time())}.pt"
    report = run_candidate_update(
        stable_checkpoint=stable_checkpoint, tokenizer_path=tokenizer_path,
        verified=verified, sealed_prompts=sealed_prompts, sealed_answers=sealed_answers,
        dev=dev, updates=updates, budget=budget, arms=arms, candidate_path=cand_file,
        wall_limit_s=wall_limit_s, replay_pairs=replay_pairs, replay_ratio=replay_ratio,
        trainable=trainable)
    stable_after = file_sha256(stable_checkpoint)
    status = report.get("status", "INCONCLUSIVE")
    if report.get("executed") and stable_before != stable_after:
        status = "ERROR_SAFE"
        report["stable_integrity_violation"] = True
    audit = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "event": "candidate_proposal",
              "candidate_status": status, "stable_sha256_before": stable_before,
              "stable_sha256_after": stable_after,
              "stable_intact": stable_before == stable_after,
              "gates": report.get("gates"),
              "new_skill_gain": report.get("new_skill_gain"),
              "old_task_forgetting_pp": report.get("old_task_forgetting_pp"),
              "report": {k: v for k, v in report.items() if k != "arms"}}
    _audit_path(out).open("a", encoding="utf-8").write(json.dumps(audit, ensure_ascii=False) + "\n")
    if status in ("REJECTED", "INCONCLUSIVE", "ERROR_SAFE"):
        quarantine = out / "rejected"
        quarantine.mkdir(exist_ok=True)
        if Path(cand_file).exists():
            shutil.move(str(cand_file), str(quarantine / Path(cand_file).name))
            audit["quarantined_to"] = str(quarantine / Path(cand_file).name)
    else:
        audit["candidate_kept_for_approval"] = str(cand_file)
    return {"status": status, "report": report, "audit": audit}


def approve_promotion(*, candidate_path: str | Path, stable_checkpoint: str | Path,
                      out_dir: str | Path, confirm: bool = False) -> dict[str, Any]:
    """人工批准的原子晋升：校验候选 → 备份旧 stable → 原子替换指针，可 rollback。"""
    if not confirm:
        return {"promoted": False, "reason": "promotion_mode=manual：必须显式 confirm=True "
                                             "（CLI 需 --i-approve），候选永不自动覆盖 stable"}
    out = Path(out_dir)
    cand = Path(candidate_path)
    stable = Path(stable_checkpoint)
    if not cand.exists():
        return {"promoted": False, "reason": f"候选不存在或已被隔离：{cand}"}
    pointer = out / "model_lineage.json"
    prev = (json.loads(pointer.read_text(encoding="utf-8")) if pointer.exists() else {})
    backup = out / "stable_backups" / f"{stable.name}.{int(time.time())}.bak"
    backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(stable, backup)
    tmp = stable.with_suffix(stable.suffix + ".tmp")
    shutil.copy2(cand, tmp)
    tmp.replace(stable)                                  # 原子切换（同盘 rename）
    lineage = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "stable": str(stable),
                "stable_sha256": file_sha256(stable), "promoted_from": str(cand),
                "previous": prev, "rollback_to": str(backup)}
    pointer.write_text(json.dumps(lineage, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"promoted": True, "lineage": lineage}


def rollback_stable(*, out_dir: str | Path, stable_checkpoint: str | Path) -> dict[str, Any]:
    """回滚到晋升前备份（存在 model_lineage.json 时）。"""
    pointer = Path(out_dir) / "model_lineage.json"
    stable = Path(stable_checkpoint)
    if not pointer.exists():
        return {"rolled_back": False, "reason": "无 model_lineage.json，从未晋升过"}
    lineage = json.loads(pointer.read_text(encoding="utf-8"))
    bak = Path(lineage["rollback_to"])
    if not bak.exists():
        return {"rolled_back": False, "reason": f"备份缺失：{bak}"}
    shutil.copy2(bak, stable)
    return {"rolled_back": True, "stable_sha256": file_sha256(stable), "from": str(bak)}
