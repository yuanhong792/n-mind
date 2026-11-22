"""能力台账（M5 任务书 P1 的 Bot 版）：N-MIND 实际能完成什么，只认实测证据。

六级口径：UNTESTED / TESTED_FAIL / MEMORY_ASSISTED / TOOL_ASSISTED / NEURAL_TRAINED /
NEURAL_GENERALIZED。工具调用成功 ≠ 神经掌握；level 只能由 record_* 实测推进，无自报接口。

R2 在这套口径上再补四列归因（response_source / feedback_source / learning_target /
capability_state），因为 R1 的失败复盘表明：**账本把不同来源的成功混在一起**，
才让"81 次练习、0 次神经成长"看起来像"练习不够"。现在一条经验必须说清
谁答的、怎么验的、打算拿来训什么，三件缺一就按最保守口径记账。
"""

from __future__ import annotations

import time
from typing import Any

LEVELS = ("UNTESTED", "TESTED_FAIL", "MEMORY_ASSISTED", "TOOL_ASSISTED",
           "NEURAL_TRAINED", "NEURAL_GENERALIZED")
NEURAL_HELDOUT_THRESHOLD = 0.75

# R2 归因词表（与 skill_registry / task_capability_bridge 用同一套名字，报告直接可比）
RESPONSE_SOURCES = ("neural", "tool", "retrieval", "rule", "abstain", "unknown")
FEEDBACK_SOURCES = ("deterministic_oracle", "externally_verified", "user_approved", "unverified")
LEARNING_TARGETS = ("neural_generation", "neural_tool_selection", "memory_only", "none")
CAPABILITY_STATES = ("NEURAL_MASTERED", "NEURAL_PARTIAL", "TOOL_VERIFIED", "MEMORY_ONLY",
                     "RULE_ONLY", "UNSUPPORTED", "UNTESTED", "ABSTAIN")


def _check_vocab(name: str, value: str, vocab: tuple[str, ...]) -> str:
    if value not in vocab:
        raise ValueError(f"{name}={value!r} 不在词表 {vocab}")
    return value


class CapabilityTracker:
    def __init__(self, *, neural_threshold: float = NEURAL_HELDOUT_THRESHOLD) -> None:
        self.records: dict[str, dict[str, Any]] = {}
        self.neural_threshold = neural_threshold

    def record_attempt(self, *, capability_id: str, task_family: str, success: bool,
                       path: str, evidence_id: str = "", cost_s: float = 0.0,
                       tool_calls: int = 0, network_calls: int = 0,
                       model_sha: str = "", skill_id: str = "",
                       response_source: str = "unknown",
                       feedback_source: str = "unverified",
                       learning_target: str = "none") -> dict[str, Any]:
        """记一次实测。R2 起必须同时说清：谁答的（response_source）、怎么验的
        （feedback_source）、这条经验打算用来干什么（learning_target）。
        缺任一项就落到最保守口径（unknown / unverified / none）。"""
        if path not in ("neural", "memory", "tool", "none"):
            raise ValueError(f"非法 outcome_source {path!r}")
        _check_vocab("response_source", response_source, RESPONSE_SOURCES)
        _check_vocab("feedback_source", feedback_source, FEEDBACK_SOURCES)
        _check_vocab("learning_target", learning_target, LEARNING_TARGETS)
        rec = self.records.setdefault(capability_id, self._new_rec(capability_id, task_family))
        # Backward-compatible normalization for tracker records created before
        # the V1.1 attribution fields existed.
        for table in ("response_sources", "feedback_sources", "learning_targets"):
            rec.setdefault(table, {})
        rec["attempts"] += 1
        rec["skill_id"] = skill_id or rec["skill_id"]
        # unverified 永远不计成功：§5 四态反馈里它和 verified 是两个账户
        counted = bool(success) and feedback_source != "unverified"
        for value, table in ((response_source, "response_sources"),
                             (feedback_source, "feedback_sources"),
                             (learning_target, "learning_targets")):
            bucket = rec[table]
            bucket[value] = bucket.get(value, 0) + 1
        rec["tool_invocations"] += tool_calls
        rec["network_invocations"] += network_calls
        if counted:
            rec["verified_successes"] += 1
            if path != "none":
                rec["success_paths"].append(path)
            if evidence_id:
                rec["evidence_ids"].append(evidence_id)
        else:
            rec["failures"] += 1
        rec["recent"].append(counted)
        rec["recent"] = rec["recent"][-10:]
        if cost_s > 0:
            prev = rec["average_cost_s"] or 0.0
            rec["average_cost_s"] = round(
                (prev * (rec["attempts"] - 1) + cost_s) / rec["attempts"], 4)
        rec["last_verified_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        if model_sha:
            rec["model_sha"] = model_sha
        rec["level"] = self._level(rec)
        rec["capability_state"] = self._state(rec, response_source=response_source,
                                              feedback_source=feedback_source)
        return dict(rec)

    def _state(self, rec: dict[str, Any], *, response_source: str = "",
               feedback_source: str = "") -> str:
        """把 6 级 level 收敛到任务书的 8 态 capability_state，并显式区分神经与非神经。"""
        srcs = set(rec.get("response_sources") or {})
        if response_source == "abstain" or "abstain" in srcs:
            return "ABSTAIN"
        if rec["heldout_score"] is not None \
                and rec["heldout_score"] >= self.neural_threshold \
                and "neural" in rec["success_paths"]:
            return "NEURAL_MASTERED"
        if "neural" in rec["success_paths"]:
            return "NEURAL_PARTIAL"
        if "tool" in rec["success_paths"] or "tool" in srcs:
            return "TOOL_VERIFIED"
        if "memory" in rec["success_paths"] or "retrieval" in srcs:
            return "MEMORY_ONLY"
        if "rule" in srcs:
            return "RULE_ONLY"
        if rec.get("attempts", 0) == 0:
            return "UNTESTED"
        # 测过但一次都没通过（含词表外 UNK 流）：当前能力做不到，如实记 UNSUPPORTED，
        # 不许留成 UNTESTED 假装"还没测"，也不许记成 RULE_ONLY 假装"规则能替代"。
        return "UNSUPPORTED" if rec.get("verified_successes", 0) == 0 else "RULE_ONLY"

    def _new_rec(self, capability_id: str, task_family: str) -> dict[str, Any]:
        """两条 record_* 路径共用同一个模板，避免"heldout 先建的记录缺归因列"。"""
        return {"capability_id": capability_id, "task_family": task_family,
                "attempts": 0, "verified_successes": 0, "failures": 0,
                "heldout_score": None, "tool_invocations": 0, "network_invocations": 0,
                "average_cost_s": None, "last_verified_at": None, "model_sha": "",
                "evidence_ids": [], "success_paths": [], "recent": [], "level": "UNTESTED",
                "skill_id": "", "response_sources": {}, "feedback_sources": {},
                "learning_targets": {}, "capability_state": "UNTESTED"}

    def record_heldout(self, *, capability_id: str, task_family: str, score: float,
                       model_sha: str = "", evidence_id: str = "",
                       skill_id: str = "") -> dict[str, Any]:
        """封存/holdout 实测分数——NEURAL_MASTERED 只认这条路径的证据，不认练习次数。"""
        rec = self.records.setdefault(capability_id, self._new_rec(capability_id, task_family))
        rec["heldout_score"] = round(float(score), 4)
        rec["last_verified_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        if model_sha:
            rec["model_sha"] = model_sha
        if skill_id:
            rec["skill_id"] = skill_id
        if evidence_id:
            rec["evidence_ids"].append(evidence_id)
        rec["level"] = self._level(rec)
        rec["capability_state"] = self._state(rec)
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
            agg.setdefault("capability_states", {})[rec["capability_id"].split(":")[-1]] = \
                rec.get("capability_state", "UNTESTED")
            for table in ("response_sources", "feedback_sources"):
                merged = agg.setdefault(table, {})
                for name, n in (rec.get(table) or {}).items():
                    merged[name] = merged.get(name, 0) + n
        return out

    def skill_summary(self) -> dict[str, Any]:
        """按 skill_id 汇总（R2 技能注册表口径）。没带 skill_id 的历史记录归入 unattributed。

        这张表就是"哪些成功是神经的、哪些是工具/记忆/规则的"唯一权威视图，
        CLI `bot skills` 与 BOT_R2_REPORT 都读它，不再从 attempts 计数反推。
        """
        out: dict[str, Any] = {}
        for rec in self.records.values():
            sid = rec.get("skill_id") or "unattributed"
            agg = out.setdefault(sid, {"task_family": rec["task_family"], "attempts": 0,
                                        "verified_successes": 0, "failures": 0,
                                        "heldout_score": None,
                                        "capability_state": "UNTESTED", "levels": [],
                                        "response_sources": {}, "feedback_sources": {},
                                        "learning_targets": {}})
            agg["attempts"] += rec["attempts"]
            agg["verified_successes"] += rec["verified_successes"]
            agg["failures"] += rec["failures"]
            agg["capability_state"] = rec.get("capability_state", "UNTESTED")
            if rec["level"] not in agg["levels"]:
                agg["levels"].append(rec["level"])
            if rec["heldout_score"] is not None:
                agg["heldout_score"] = (rec["heldout_score"] if agg["heldout_score"] is None
                                        else min(agg["heldout_score"], rec["heldout_score"]))
            for table in ("response_sources", "feedback_sources", "learning_targets"):
                bucket = agg[table]
                for name, n in (rec.get(table) or {}).items():
                    bucket[name] = bucket.get(name, 0) + n
        return out

    @staticmethod
    def _dominant(table: dict[str, int], fallback: str) -> str:
        """写库用的单值归因：取出现最多的来源（明细仍完整保存在 *_sources 计数里）。"""
        if not table:
            return fallback
        return max(sorted(table.items()), key=lambda kv: (kv[1], kv[0]))[0]

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
                      "skill_id": v.get("skill_id", ""),
                      "response_source": self._dominant(v.get("response_sources") or {}, "unknown"),
                      "feedback_source": self._dominant(v.get("feedback_sources") or {},
                                                         "unverified"),
                      "learning_target": self._dominant(v.get("learning_targets") or {}, "none"),
                      "capability_state": v.get("capability_state", "UNTESTED"),
                      "tool_invocations": v["tool_invocations"],
                      "network_invocations": v["network_invocations"],
                      "average_cost_s": v["average_cost_s"]}
        return out
