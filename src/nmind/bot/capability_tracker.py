"""能力台账（M5 任务书 P1 的 Bot 版）：N-MIND 实际能完成什么，只认实测证据。

五级口径：UNTESTED / TESTED_FAIL / MEMORY_ASSISTED / TOOL_ASSISTED / NEURAL_GENERALIZED。
工具调用成功 ≠ 神经掌握；level 只能由 record_* 实测推进，无自报接口。
"""

from __future__ import annotations

import time
from typing import Any

LEVELS = ("UNTESTED", "TESTED_FAIL", "MEMORY_ASSISTED", "TOOL_ASSISTED",
           "NEURAL_TRAINED", "NEURAL_GENERALIZED")
NEURAL_HELDOUT_THRESHOLD = 0.75


class CapabilityTracker:
    def __init__(self, *, neural_threshold: float = NEURAL_HELDOUT_THRESHOLD) -> None:
        self.records: dict[str, dict[str, Any]] = {}
        self.neural_threshold = neural_threshold

    def record_attempt(self, *, capability_id: str, task_family: str, success: bool,
                       path: str, evidence_id: str = "", cost_s: float = 0.0,
                       tool_calls: int = 0, network_calls: int = 0,
                       model_sha: str = "") -> dict[str, Any]:
        if path not in ("neural", "memory", "tool", "none"):
            raise ValueError(f"非法 outcome_source {path!r}")
        rec = self.records.setdefault(capability_id, {
            "capability_id": capability_id, "task_family": task_family,
            "attempts": 0, "verified_successes": 0, "failures": 0,
            "heldout_score": None, "tool_invocations": 0, "network_invocations": 0,
            "average_cost_s": None, "last_verified_at": None, "model_sha": model_sha,
            "evidence_ids": [], "success_paths": [], "recent": [], "level": "UNTESTED",
        })
        rec["attempts"] += 1
        rec["tool_invocations"] += tool_calls
        rec["network_invocations"] += network_calls
        if success:
            rec["verified_successes"] += 1
            if path != "none":
                rec["success_paths"].append(path)
            if evidence_id:
                rec["evidence_ids"].append(evidence_id)
        else:
            rec["failures"] += 1
        rec["recent"].append(success)
        rec["recent"] = rec["recent"][-10:]
        if cost_s > 0:
            prev = rec["average_cost_s"] or 0.0
            rec["average_cost_s"] = round(
                (prev * (rec["attempts"] - 1) + cost_s) / rec["attempts"], 4)
        rec["last_verified_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        if model_sha:
            rec["model_sha"] = model_sha
        rec["level"] = self._level(rec)
        return dict(rec)

    def record_heldout(self, *, capability_id: str, task_family: str, score: float,
                       model_sha: str = "", evidence_id: str = "") -> dict[str, Any]:
        rec = self.records.setdefault(capability_id, {
            "capability_id": capability_id, "task_family": task_family,
            "attempts": 0, "verified_successes": 0, "failures": 0,
            "heldout_score": None, "tool_invocations": 0, "network_invocations": 0,
            "average_cost_s": None, "last_verified_at": None, "model_sha": "",
            "evidence_ids": [], "success_paths": [], "recent": [], "level": "UNTESTED",
        })
        rec["heldout_score"] = round(float(score), 4)
        rec["last_verified_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        if model_sha:
            rec["model_sha"] = model_sha
        if evidence_id:
            rec["evidence_ids"].append(evidence_id)
        rec["level"] = self._level(rec)
        return dict(rec)

    def _level(self, rec: dict[str, Any]) -> str:
        if rec["heldout_score"] is not None \
                and rec["heldout_score"] >= self.neural_threshold \
                and "neural" in rec["success_paths"]:
            return "NEURAL_GENERALIZED"
        if rec["attempts"] == 0:
            return "UNTESTED"
        if rec["verified_successes"] == 0:
            return "TESTED_FAIL"
        paths = set(rec["success_paths"])
        if "tool" in paths:
            return "TOOL_ASSISTED"
        if "memory" in paths:
            return "MEMORY_ASSISTED"
        return "NEURAL_TRAINED"          # 神经路径成功但无 holdout 泛化证据

    def family_summary(self) -> dict[str, Any]:
        """按 task_family 汇总（GoalManager 必须用这个，不能用 capability_id 键的 summary()）。

        历史 bug：summary() 的键是 "arithmetic:neural" 这类 capability_id，
        GoalManager 拿 family 名去查恒为 None ⇒ 目标永远停在第一个任务族，
        三类任务轮转从未发生。
        """
        out: dict[str, Any] = {}
        for rec in self.records.values():
            fam = rec["task_family"]
            agg = out.setdefault(fam, {"attempts": 0, "verified_successes": 0, "failures": 0,
                                        "heldout_score": None, "levels": {},
                                        "tool_invocations": 0, "network_invocations": 0})
            agg["attempts"] += rec["attempts"]
            agg["verified_successes"] += rec["verified_successes"]
            agg["failures"] += rec["failures"]
            agg["tool_invocations"] += rec["tool_invocations"]
            agg["network_invocations"] += rec["network_invocations"]
            if rec["heldout_score"] is not None:
                agg["heldout_score"] = (rec["heldout_score"] if agg["heldout_score"] is None
                                         else min(agg["heldout_score"], rec["heldout_score"]))
            agg["levels"][rec["capability_id"].split(":")[-1]] = rec["level"]
        return out

    def consecutive_failures(self, capability_id: str) -> int:
        rec = self.records.get(capability_id)
        if not rec:
            return 0
        n = 0
        for ok in reversed(rec["recent"]):
            if ok:
                break
            n += 1
        return n

    def summary(self) -> dict[str, Any]:
        """能力快照（写库 + CLI 展示用）。

        必须带 task_family 与 outcome_source：BotStore.upsert_capability 直接绑定这些列，
        历史 bug 是缺 task_family 触发 sqlite ProgrammingError 打断空闲循环。
        """
        out: dict[str, Any] = {}
        for k, v in self.records.items():
            paths = sorted(set(v["success_paths"]))
            source = ("tool" if "tool" in paths else
                      "memory" if "memory" in paths else
                      "neural" if "neural" in paths else "none")
            out[k] = {"capability_id": k, "task_family": v["task_family"],
                      "level": v["level"], "attempts": v["attempts"],
                      "verified_successes": v["verified_successes"],
                      "failures": v["failures"], "heldout_score": v["heldout_score"],
                      "outcome_source": source,
                      "tool_invocations": v["tool_invocations"],
                      "network_invocations": v["network_invocations"],
                      "average_cost_s": v["average_cost_s"]}
        return out
