"""BotRuntime（M5 Agent Runtime R1）：独立本地 Bot 的生命周期、状态机与自主练习循环。

状态机（任务书 §2）：STOPPED→STARTING→READY→(SERVING|IDLE_EXPLORE→IDLE_PRACTICE→
CANDIDATE_TRAINING→SEALED_EVALUATION→AWAITING_APPROVAL)→COOLDOWN→READY；PAUSED/ERROR_SAFE
可从多数状态进入；STOP 由任意运行态进入并退出进程。

硬约束（代码强制，不靠提示词）：
  * 单实例锁（lock.json 含 PID；第二实例直接拒绝启动）；
  * 停止/暂停经控制文件（另一终端可写），运行循环在**安全边界**响应，绝不忙等；
  * stable checkpoint 只读（只校验 sha，从不写回）；
  * 封存评测集用独立随机流生成、按内容指纹持久化，永不进入学习流；
  * 练习实例按指纹去重：重启不会重复执行同一实例（老实现 completed_tasks 写了却从不查）；
  * 默认 auto_candidate_training=false 且授权预算默认 0：无显式授权一律 MEMORY_ONLY；
  * 外部 LLM / 预训练权重不可参与（response_source 区分 neural/tool/retrieval/none）。

本版修复的既有缺陷（P0 审计结论，逐条对应代码）：
  1. AdaptationBudget/torch 在文件末尾 import、类内引用 → 提到顶部；
  2. tokenizer 与模型各加载两次 → 只加载一次；
  3. `propose_candidate(...) if False else None` 死分支引用未定义 dev → 删除，
     授权路径真正走 evolution_manager（含 stable sha 前后校验与隔离）；
  4. `random.Random()` 无种子 → 固定种子，循环可复现；
  5. 封存实例来自同一随机流（i % 4 == 3）→ 独立种子 + 持久化 + 指纹排除；
  6. budget_remaining 直接报 policy.candidate_max_steps（假值）→ 真实累计扣减；
  7. 无驻留循环/心跳 → run_forever + 心跳 + 可中断 sleep；
  8. 资源上限只写在配置里 → 每轮实测 CPU/RSS 并记录 throttle 事件。
"""

from __future__ import annotations

import enum
import hashlib
import json
import os
import random
import time
import traceback
from types import SimpleNamespace
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from ..language.decoder import generate
from ..language.neural_dialogue import load_lm
from ..language.tokenizer import CharTokenizer
from ..learning.online_adaptation import AdaptationBudget, state_hash
from ..online.scheduler import SchedulerConfig
from . import resources
from .capability_tracker import CapabilityTracker
from .experience_bridge import VerifiedExperience, build_experience
from .curriculum import CurriculumConfig, SkillSelector, dataset_stats
from .goal_manager import GoalManager
from .policy import BotPolicy
from .practice import (execute_tool, feedback_state, make_sealed_set, make_task,
                     program_oracle, verify)
from .scheduler import BotScheduler
from .skill_registry import default_registry
from .status import build_snapshot
from .store import BotStore

LOCK_FILE = "lock.json"
PAUSE_FLAG = "pause.flag"
STOP_FLAG = "stop.flag"
STATUS_FILE = "status.json"
INBOX_FILE = "inbox.jsonl"
EXPERIENCE_FILE = "experiences.jsonl"
SEED = 20261601                                # 练习流固定种子（可复现）
SEALED_SIZE = 8                               # 每族封存实例数（独立随机流）
MAX_TASK_RETRY = 24                           # 指纹去重后最多尝试几次出题（空间穷尽保护）
LOG_MAX_BYTES = 64 * 1024                     # 单个 jsonl 日志上限（§9.12 容量上限）
LOG_KEEP_LINES = 500                          # 超限后保留的最近行数
EVENT_KEEP = 2000                             # SQLite events 表保留上限


class BotState(enum.Enum):
    STOPPED = enum.auto()
    STARTING = enum.auto()
    READY = enum.auto()
    SERVING = enum.auto()
    IDLE_EXPLORE = enum.auto()
    IDLE_PRACTICE = enum.auto()
    CANDIDATE_TRAINING = enum.auto()
    SEALED_EVALUATION = enum.auto()
    AWAITING_APPROVAL = enum.auto()
    COOLDOWN = enum.auto()
    PAUSED = enum.auto()
    ERROR_SAFE = enum.auto()


_LEGAL = {
    "STOPPED": {"STARTING"},
    "STARTING": {"READY", "ERROR_SAFE"},
    "READY": {"SERVING", "IDLE_EXPLORE", "PAUSED", "STOPPED", "COOLDOWN", "ERROR_SAFE",
              # `nmind bot candidate --i-authorize` 是人工显式授权的独立入口：
              # 它不经过空闲循环，直接从 READY 进候选训练（预算与门槛完全一样）。
              "CANDIDATE_TRAINING"},
    "SERVING": {"READY", "ERROR_SAFE", "PAUSED", "STOPPED"},
    "IDLE_EXPLORE": {"IDLE_PRACTICE", "READY", "PAUSED", "STOPPED", "ERROR_SAFE", "SERVING"},
    "IDLE_PRACTICE": {"READY", "CANDIDATE_TRAINING", "COOLDOWN", "PAUSED", "STOPPED",
                      "ERROR_SAFE"},
    "CANDIDATE_TRAINING": {"SEALED_EVALUATION", "READY", "ERROR_SAFE", "PAUSED"},
    "SEALED_EVALUATION": {"AWAITING_APPROVAL", "READY", "COOLDOWN", "ERROR_SAFE"},
    "AWAITING_APPROVAL": {"READY", "COOLDOWN", "PAUSED", "ERROR_SAFE"},
    "COOLDOWN": {"READY", "PAUSED", "STOPPED", "ERROR_SAFE", "SERVING"},
    "PAUSED": {"READY", "STOPPED", "ERROR_SAFE"},
    "ERROR_SAFE": {"STOPPED", "READY"},
}


def can_transition(cur: str, new: str) -> bool:
    return new in _LEGAL.get(cur, set()) or cur == new


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


class BotRuntime:
    """独立本地 Bot：状态机 + 空闲自主练习循环 + 用户任务优先 + 审计留痕。"""

    def __init__(self, *, root_dir: str | Path, policy: BotPolicy | None = None,
                 stable_checkpoint: str | Path | None = None,
                 tokenizer_path: str | Path | None = None,
                 run_id: str | None = None, clock=time.monotonic,
                 device: torch.device | None = None, seed: int = SEED,
                 authorized_optimizer_steps: int = 0,
                 sealed_per_family: int | None = None) -> None:
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.policy = policy or BotPolicy()
        self.monotonic = clock                            # 可插拔时钟（测试用）
        self.run_id = run_id or f"run-{int(time.time())}-{os.getpid()}"
        self.state = BotState.STOPPED
        self.device = device or torch.device("cpu")       # 默认 CPU（任务书 §11 CPU 必须可跑）
        self.seed = int(seed)
        self.sealed_per_family = int(self.policy.heldout_per_family if sealed_per_family is None
                                     else sealed_per_family)
        self.store = BotStore(self.root / "bot_state.db")
        self.tracker = CapabilityTracker()
        self.goals = GoalManager(allowlist=self.policy.practice_task_allowlist,
                                 validator_ids={f: f"oracle-{f}"
                                                for f in self.policy.practice_task_allowlist})
        self.scheduler = BotScheduler(
            SchedulerConfig(verified_threshold=self.policy.scheduler_verified_threshold,
                            cooldown_s=self.policy.min_cooldown_seconds,
                            max_updates_per_batch=self.policy.scheduler_max_updates_per_batch),
            cooldown_s=self.policy.min_cooldown_seconds, clock=clock)
        self.user_queue: list[str] = []
        self._rng = random.Random(self.seed)              # 练习流（可复现）
        self._started = False
        self.res_window = resources.Window()
        self.resource_totals: dict[str, float] = {**{"wall_s": 0.0, "cpu_s": 0.0,
                                                    "rss_peak_mb": 0.0, "cycles": 0},
                                                  **self.store.get_state("resource_totals", {})}
        self.exhausted_families: set[str] = set()
        # ---- R2：技能注册表 + 能力匹配选择器（Gate-A 产物只读加载，绝不在循环里重算）----
        self.registry = default_registry()
        allow = tuple(self.policy.practice_skill_allowlist) or tuple(
            sk.skill_id for sk in self.registry.trainable())
        self.selector = SkillSelector(
            CurriculumConfig(exploration_fraction=self.policy.idle_exploration_fraction),
            allowed_skill_ids=allow)
        self._gate_a: dict[str, Any] | None = None
        self._gate_a_note: str = ""
        self._dataset_stats_cache: dict[str, Any] | None = None
        self._recent_skills: list[str] = list(self.store.get_state("recent_skills", []))
        self._skill_row_cursor = int(self.store.get_state("skill_row_cursor", 0))
        self.skill_cycles = int(self.store.get_state("skill_cycles", 0))
        self.dormant_cycles = int(self.store.get_state("dormant_cycles", 0))
        self.cpu_cap: dict[str, Any] = {}
        self._heldout_done: set[str] = set(self.store.get_state("heldout_done", []))

        # ---- 语言侧：stable 权重只读加载一次 ----
        self._model = None
        self._tokenizer = None
        self._model_meta: dict[str, Any] = {}
        self._lang_paths: tuple[str, str] | None = None
        self.stable_sha = "no-language-checkpoint"
        self.stable_file_sha = "no-language-checkpoint"
        if stable_checkpoint and tokenizer_path and Path(stable_checkpoint).exists() \
                and Path(tokenizer_path).exists():
            from ..online.promotion import file_sha256
            self._tokenizer = CharTokenizer.load(tokenizer_path)
            self._model, self._model_meta = load_lm(Path(stable_checkpoint), self._tokenizer)
            self._model.eval()
            self.stable_sha = state_hash(self._model)
            self.stable_file_sha = file_sha256(stable_checkpoint)   # 与 evolution_manager 同口径
            self._lang_paths = (str(stable_checkpoint), str(tokenizer_path))

        # ---- 跨重启的累计状态（老实现每次重启都从 0 开始、预算报假值）----
        self.cycles_done = int(self.store.get_state("cycles_done", 0))
        self._recent_families: list[str] = list(self.store.get_state("recent_families", []))
        self.candidate_status = str(self.store.get_state("candidate_status", "NONE"))
        self.last_result: str | None = self.store.get_state("last_result")
        self.steps_used = int(self.store.get_state("candidate_steps_used", 0))
        self.authorized_steps_total = int(self.store.get_state(
            "candidate_steps_authorized", int(authorized_optimizer_steps)))
        if authorized_optimizer_steps:
            self.store.set_state("candidate_steps_authorized", self.authorized_steps_total)
        if self.store.get_state("tracker_records"):
            self.tracker.records = self.store.get_state("tracker_records")

    # ---- 生命周期 ----
    def start(self) -> None:
        if self._started:
            return
        self._transition("STARTING", "启动：加载稳定权重与记录")
        if self.policy.single_instance:
            lock = self.root / LOCK_FILE
            if lock.exists():
                try:
                    prev = json.loads(lock.read_text(encoding="utf-8"))
                except Exception:                        # noqa: BLE001 - 锁文件损坏也照样拒绝
                    prev = {"pid": "unknown"}
                if not self._pid_alive(int(prev.get("pid") or -1)):
                    lock.unlink()                        # 陈旧锁（进程已死）可回收
                    self.store.event(self.run_id, "stale_lock_reclaimed", {"prev": prev})
                else:
                    raise RuntimeError(f"另一 Bot 实例正在运行（{lock} · pid={prev.get('pid')}）；"
                                       "single_instance 硬约束")
            lock.write_text(json.dumps({"pid": os.getpid(), "run_id": self.run_id,
                                        "ts": _now_iso()}, ensure_ascii=False), encoding="utf-8")
        self._started = True
        (self.root / STOP_FLAG).unlink(missing_ok=True)   # 上次遗留的停止请求不该卡住新会话
        self._apply_cpu_cap()
        self._transition("READY", "就绪")
        self.write_status()

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        if pid <= 0:
            return True                                  # 未知 → 保守认为仍在运行
        try:
            import psutil
            return psutil.pid_exists(pid)
        except Exception:                                # noqa: BLE001 - 无 psutil 时保守
            return True

    def stop(self, reason: str = "用户停止") -> None:
        if self.state == BotState.STOPPED:
            return
        self._transition("STOPPED", reason)
        lock = self.root / LOCK_FILE
        if lock.exists():
            lock.unlink()
        self._persist()
        self.write_status()
        self._started = False
        self.store.close()

    def pause(self, reason: str = "用户暂停") -> None:
        (self.root / PAUSE_FLAG).write_text(reason, encoding="utf-8")
        if self.state not in (BotState.PAUSED, BotState.STOPPED):
            self._transition("PAUSED", reason)

    def resume(self) -> None:
        flag = self.root / PAUSE_FLAG
        if flag.exists():
            flag.unlink()
        if self.state == BotState.PAUSED:
            self._transition("READY", "用户恢复")
        elif self.state == BotState.ERROR_SAFE:
            if self._lang_paths:
                from ..online.promotion import file_sha256
                now = file_sha256(self._lang_paths[0])
                if self.stable_file_sha != "no-language-checkpoint" and now != self.stable_file_sha:
                    self.store.event(self.run_id, "resume_refused",
                                      {"reason": "stable 文件 sha 与启动记录不一致",
                                       "expected": self.stable_file_sha, "actual": now})
                    raise RuntimeError("stable checkpoint 与启动时不一致，拒绝从 ERROR_SAFE 恢复；"
                                       "请先 nmind bot rollback")
            self._transition("READY", "操作员确认 ERROR_SAFE 后恢复（stable 完整性已复核）")

    def _check_control(self) -> str | None:
        if (self.root / STOP_FLAG).exists():
            return "stop"
        if (self.root / PAUSE_FLAG).exists():
            return "pause"
        return None

    def error_safe(self, reason: str, detail: str = "") -> None:
        """异常安全态：停下一切写训练/训练类动作，保留 stable 与记录。"""
        if self.state == BotState.ERROR_SAFE:
            return
        self._transition("ERROR_SAFE", reason)
        self.store.event(self.run_id, "error_safe", {"reason": reason, "detail": detail[:2000]})
        self.candidate_status = "ERROR_SAFE"
        self._persist()
        self.write_status()

    # ---- 状态与审计 ----
    def _transition(self, new: str, reason: str) -> None:
        cur = self.state.name
        if new not in _LEGAL.get(cur, set()) and cur != new:
            raise RuntimeError(f"非法状态转移 {cur} → {new}（{reason}）")
        self.state = BotState[new]
        self.store.event(self.run_id, "transition", {"from": cur, "to": new, "reason": reason})
        self._append(self.root / "runtime_transitions.jsonl",
                     {"ts": _now_iso(), "run_id": self.run_id, "from": cur, "to": new,
                      "reason": reason})

    def _append(self, path: Path, record: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._rotate_log(path)

    @staticmethod
    def _log_needs_rotation(path: Path) -> bool:
        try:
            return path.stat().st_size > LOG_MAX_BYTES
        except OSError:
            return False

    def _rotate_log(self, path: Path) -> None:
        """日志容量上限（任务书 §9.12）：单个 jsonl 超 LOG_MAX_BYTES 只留最近的行。

        老实现没有任何上限：常驻 Bot 每循环写 4 类日志，磁盘会无限涨。
        截到"低水位"（上限的一半）而不是刚好到上限，否则超限后每次 append 都要重写一遍。
        审计数据库同步 prune（见 _trim_store）。
        """
        if not self._log_needs_rotation(path):
            return
        from collections import deque
        low_water = LOG_MAX_BYTES // 2
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            data = deque(fh)
        while len(data) > LOG_KEEP_LINES or (data and sum(
                len(l.encode("utf-8")) for l in data) > low_water):
            if len(data) <= 1:
                break
            data.popleft()                       # 丢最旧，保最近
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            fh.writelines(data)
        tmp.replace(path)
        self.store.event(self.run_id, "log_rotated",
                          {"path": path.name, "kept_lines": len(data),
                           "cap_bytes": LOG_MAX_BYTES})

    def _trim_store(self) -> None:
        """事件表容量上限：超过 EVENT_KEEP 的旧事件丢弃（完整历史已落 jsonl）。"""
        self.store.prune_events(keep=EVENT_KEEP)

    def _persist(self) -> None:
        self.store.set_state("cycles_done", self.cycles_done)
        self.store.set_state("recent_families", self._recent_families[-8:])
        self.store.set_state("candidate_status", self.candidate_status)
        self.store.set_state("last_result", self.last_result)
        self.store.set_state("candidate_steps_used", self.steps_used)
        self.store.set_state("tracker_records", self.tracker.records)
        self.store.set_state("resource_totals", self.resource_totals)
        self.store.set_state("heldout_done", sorted(self._heldout_done))
        for cid, rec in self.tracker.summary().items():
            self.store.upsert_capability({**rec, "capability_id": cid,
                                          "evidence_path": "autonomous_cycles.jsonl"})

    def write_status(self) -> None:
        """status.json 落盘：另一终端的 `nmind bot status` 直接读它（不需要 IPC）。"""
        snap = self.status()
        # Readers in another process must never observe a half-written JSON file.
        # This is the same-directory atomic-replace pattern used by log rotation.
        status_path = self.root / STATUS_FILE
        tmp_path = status_path.with_suffix(status_path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(snap, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        tmp_path.replace(status_path)

    # ---- 用户任务（优先于空闲学习）----
    def submit_user_task(self, text: str) -> dict[str, Any]:
        self.user_queue.append(text)
        return {"queued": True, "queue_len": len(self.user_queue)}

    def inbox_len(self) -> int:
        box = self.root / INBOX_FILE
        if not box.exists():
            return 0
        return sum(1 for line in box.read_text(encoding="utf-8").splitlines() if line.strip())

    def _drain_inbox(self) -> list[str]:
        box = self.root / INBOX_FILE
        if not box.exists():
            return []
        texts = [json.loads(line)["text"] for line in box.read_text(encoding="utf-8").splitlines()
                 if line.strip()]
        box.unlink()
        return texts

    def process_user_tasks(self) -> list[dict[str, Any]]:
        """用户消息优先：先处理队列（含跨进程 inbox），空闲练习让路（任务书 §4.1）。"""
        if self.state in (BotState.PAUSED, BotState.ERROR_SAFE, BotState.STOPPED):
            return []                    # 暂停/故障态不动工作队列：消息留在 inbox 里等恢复
        texts = self._drain_inbox() + self.user_queue
        self.user_queue = []
        if not texts:
            return []
        self._transition("SERVING", f"{len(texts)} 条用户消息到达，暂停低优先级练习")
        replies = []
        for text in texts:
            if self._model is not None:
                from ..language.session import ChatSession
                if getattr(self, "_chat", None) is None:
                    self._chat = ChatSession(self._model, self._tokenizer,
                                             session_id=f"bot-{self.run_id}")
                reply = self._chat.send(text)
                replies.append({"input": text, "text": reply.text,
                                "response_source": reply.response_source})
            else:
                replies.append({"input": text,
                                "text": "（当前无本地语言权重可用，无法作答；"
                                        "N-MIND 不调用外部大模型兜底）",
                                "response_source": "none"})
            self.store.event(self.run_id, "user_task", replies[-1])
        self._transition("READY", "用户任务处理完毕")
        return replies

    # ---- 神经尝试（稳定权重自由生成，不读标签）----
    def neural_attempt(self, question: str, *, max_new_tokens: int = 24) -> str:
        if self._model is None:
            return ""
        ids = self._tokenizer.encode(f"用户：{question}\n助手：")
        with torch.no_grad():
            _logits, state = self._model(torch.tensor([ids], dtype=torch.long))
        rec, _st = generate(self._model, self._tokenizer, self._tokenizer.encode("助手："),
                            state=state, max_new_tokens=max_new_tokens, greedy=True)
        return rec.text.strip()

    # ---- 封存评测集（独立随机流，持久化，永不进学习流）----
    def sealed_set(self, family: str) -> list[dict[str, Any]]:
        key = f"sealed:{family}"
        cached = self.store.get_state(key)
        if cached:
            return cached
        # 稳定的族后缀：Python 的 str hash 受 PYTHONHASHSEED 随机化，跨进程会给出不同封存集
        suffix = int(hashlib.sha256(family.encode("utf-8")).hexdigest()[:6], 16)
        pool = make_sealed_set(family, self.sealed_per_family, seed=self.seed + suffix)
        self.store.set_state(key, pool)
        self.store.event(self.run_id, "sealed_pool_created",
                          {"family": family, "n": len(pool),
                           "note": "独立随机流生成，永不进入学习流"})
        return pool

    def sealed_fingerprints(self, family: str) -> set[str]:
        return {t["fingerprint"] for t in self.sealed_set(family)}

    def sealed_questions(self, family: str) -> set[str]:
        """封存题面集合：练习流的排除口径必须是**题面**而不是指纹。

        指纹含答案，"同题面不同答案"的实例（json_extract 早期版本）会绕过指纹检查，
        让同一句话既进训练流又进评测流——既是矛盾监督又是评测污染。
        """
        return {t["question"] for t in self.sealed_set(family)}

    def sealed_as_rows(self, family: str) -> tuple[list[str], list[str]]:
        pool = self.sealed_set(family)
        prompts = [f"用户：{t['question']}\n助手：" for t in pool]
        answers = [str(t["answer"]) for t in pool]
        return prompts, answers

    def measure_heldout(self, family: str, *, model=None, sha: str | None = None,
                        tag: str = "stable") -> dict[str, Any]:
        """封存集上的自由生成 EM → 写入能力台账（NEURAL_GENERALIZED 的唯一依据）。

        没有本地权重时如实返回 NOT_RUN，绝不拿工具正确率冒充神经泛化。
        """
        prompts, answers = self.sealed_as_rows(family)
        if self._model is None and model is None:
            return {"status": "NOT_RUN", "family": family,
                    "reason": "无本地语言权重：神经侧未测试（UNTESTED），不冒充"}
        from .learning_loop import free_gen_em
        em = free_gen_em(model or self._model, self._tokenizer, prompts, answers)
        score = 0.0 if em is None else float(em)
        self.tracker.record_heldout(capability_id=f"{family}:neural", task_family=family,
                                    score=score, model_sha=sha or self.stable_sha,
                                    evidence_id=f"sealed:{family}:{tag}")
        self._persist()
        return {"status": "MEASURED", "family": family, "sealed_em": em, "n": len(prompts),
                 "arm": tag, "model_sha": sha or self.stable_sha}


    # ---- R2 技能级练习：Gate-A 产物 → 选择器 → 学习流轮换 ----
    def _capability_matrix(self) -> dict[str, Any] | None:
        """只读加载 Gate-A 产物。缺失/权重不符就如实返回 None 并说明原因，绝不自行重测基线。"""
        if self._model is None or self._tokenizer is None:
            self._gate_a_note = "无 stable 语言权重：技能级练习不可用（只有工具/规则练习）"
            return None
        if self._gate_a is not None:
            return self._gate_a
        from ..online.promotion import file_sha256
        repo = Path(__file__).resolve().parents[3]
        cand = ([Path(self.policy.capability_matrix_path)] if self.policy.capability_matrix_path
                else [self.root / "baseline_capability_matrix.json",
                      repo / "results" / "bot_r2" / "baseline_capability_matrix.json"])
        for path in cand:
            if not path.exists():
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                self._gate_a_note = f"Gate-A 产物不可解析：{type(exc).__name__}: {exc}"
                return None
            if data.get("gate") != "GATE_A":
                self._gate_a_note = f"{path.name} 不是 Gate-A 产物（gate={data.get('gate')!r}）"
                return None
            sha = data.get("stable_sha256_state")
            if self.stable_sha and sha and sha != self.stable_sha:
                self._gate_a_note = (f"Gate-A 基线已过期：产物 state_hash={sha}，"
                                     f"当前 stable={self.stable_sha}，必须重跑 Gate-A")
                return None
            data["_source_path"] = str(path)
            data["_source_file_sha256"] = file_sha256(path)[:16]
            self._gate_a = data
            self._gate_a_note = f"loaded:{path.name}"
            return data
        self._gate_a_note = ("Gate-A 产物不存在：先运行 experiments/bot_runtime_r2/run_gate_a.py"
                             "（本轮不在 Bot 内重测基线）")
        return None

    def _select_practice_target(self) -> dict[str, Any] | None:
        """问选择器"现在该练什么"。返回 None 表示问都没法问（缺权重或缺基线产物）。"""
        gate = self._capability_matrix()
        if gate is None:
            return None
        if self._dataset_stats_cache is None:
            per_item = float(gate.get("wall_s") or 0.0) / max(1, int(gate.get("n_measured") or 1))
            self._dataset_stats_cache = dataset_stats(
                self._tokenizer, gate, per_item_s=per_item or 0.06)
        assessments = self.selector.assess(
            gate_a=gate, registry=self.registry, dataset_stats=self._dataset_stats_cache,
            recent_families=tuple(self._recent_skills[-3:]))
        return {"gate_a_verdict": gate.get("verdict"), "assessments": assessments,
                "pick": self.selector.pick(assessments, cycle_index=self.cycles_done),
                "matrix_source": self._gate_a_note}

    def _skill_cycle(self, target: dict[str, Any], *, dry_run: bool = False
                     ) -> dict[str, Any]:
        """练 Gate-A 认定"独立 oracle + 真实缺口"的相邻技能：只喂学习流行，绝不喂评测集。"""
        pick = target["pick"]
        if dry_run:
            return {"ran": False, "dry_run": True, "practice_kind": "neural_skill",
                    "skill_plan": {
                        "status": pick["status"], "selected": pick.get("selected"),
                        "reason": pick["reason"], "exploration": pick.get("exploration"),
                        "gate_a_verdict": target["gate_a_verdict"],
                        "matrix_source": target["matrix_source"],
                        "rows_per_cycle": self.policy.skill_tasks_per_idle_cycle,
                        "candidate": self._candidate_plan()}}
        from ..language import binding_data as bd
        from . import task_capability_bridge as bridge
        self._transition("IDLE_PRACTICE", f"技能练习 {pick['selected']['family']}")
        fam, sid = pick["selected"]["family"], pick["selected"]["skill_id"]
        rows = bd.build_family(fam, "train")          # 预登记规模；n 覆盖会制造模板饱和假象
        if not rows:
            self.store.event(self.run_id, "skill_stream_empty", {"family": fam})
            return {"ran": False, "early_exit": True, "reason": f"{fam} 学习流为空"}
        want = max(1, int(self.policy.skill_tasks_per_idle_cycle))
        picked = [rows[(self._skill_row_cursor + i) % len(rows)] for i in range(want)]
        specs = {t.fingerprint: t for t in
                 bridge.binding_specs(self._tokenizer, splits=("train",)) if t.family == fam}
        attempts: list[dict[str, Any]] = []
        verified: list[VerifiedExperience] = []
        for i, task in enumerate(picked):
            ctrl = self._check_control()
            if ctrl in ("stop", "pause"):
                return {**self._control_exit(ctrl, attempts, goal=f"skill:{sid}"),
                        "early_exit": True}
            if self.inbox_len():
                out = self._preempt_for_user(
                    SimpleNamespace(goal_id=f"skill:{sid}", task_family=fam),
                    i, attempts, want)
                out["early_exit"] = True
                return out
            spec = specs.get(task.fingerprint)
            if spec is None:
                attempts.append({"family": fam, "fingerprint": task.fingerprint,
                                  "verification": "uncertain", "feedback_source": "unverified",
                                  "reason": "学习流行缺少对应 TaskSpec（输入桥接漏洞，如实记录）"})
                continue
            rec = bridge.attempt(self._model, self._tokenizer, spec)
            verdict = "verified" if rec["em"] else "rejected"
            fstate = feedback_state({"family": fam}, verdict, source="deterministic_oracle")
            exp = build_experience(
                {"question": task.prompt, "answer": task.target, "family": fam},
                neural_attempt=rec["predicted"], tool_result=None, verification=verdict,
                validator_id=bd.ORACLE_ID, model_sha=self.stable_sha,
                exp_id=f"exp-{self.run_id}-{self.cycles_done}-s{i}",
                skill_id=sid, response_source="neural", feedback_source=fstate,
                learning_target="neural_generation")
            attempts.append({"task_id": task.key(), "fingerprint": task.fingerprint,
                              "family": fam, "skill_id": sid, "tier": task.tier,
                              "question": task.prompt, "oracle": bd.oracle_for(task),
                              "neural_attempt": rec["predicted"], "em": rec["em"],
                              "verification": verdict, "feedback_source": fstate,
                              "response_source": "neural",
                              "learning_target": "neural_generation",
                              "format": rec["format"], "stop_reason": rec["stop_reason"],
                              "prompt_unk_rate": rec["prompt_unk_rate"], "sealed": False})
            verified.append(exp)
            self._append(self.root / EXPERIENCE_FILE, asdict(exp))
            self.store.mark_task_done(task.key(), fam, verdict, fingerprint=task.fingerprint)
            # 逐行记账：一次练习轮里有 4 条题面就该留 4 条实测，
            # 不能一轮只记 1 次再靠 all() 取与——那样"练了 4 条、台账说测了 1 条"。
            self.tracker.record_attempt(
                capability_id=f"{fam}:neural", task_family=fam,
                success=bool(rec["em"]), path="neural",
                evidence_id=f"{self.run_id}:c{self.cycles_done}:s{i}",
                model_sha=self.stable_sha, skill_id=sid,
                response_source="neural", feedback_source=fstate,
                learning_target="neural_generation")
        n_ok = sum(1 for a in attempts if a.get("verification") == "verified")
        self.cycles_done += 1
        self.skill_cycles += 1
        self._recent_skills.append(fam)
        usage = self.res_window.stop("skill_practice")
        self._accumulate_resource(usage)
        result = {"ran": True, "practice_kind": "neural_skill", "goal": f"skill:{sid}",
                  "task_family": fam, "skill_id": sid, "tier": picked[0].tier,
                  "gate_a_verdict": target["gate_a_verdict"],
                  "selection": {"status": pick["status"], "reason": pick["reason"],
                                 "selection_source": pick["selection_source"],
                                 "exploration": pick.get("exploration")},
                  "attempts": attempts, "verified_count": n_ok,
                  "rejected_count": len(attempts) - n_ok,
                  "neural_growth_claim": "none（本轮零 optimizer.step，练习只产生候选经验）",
                  "practice_resource": usage, "capabilities": self.tracker.summary()}
        self.last_result = f"skill:{sid} verified={n_ok}/{len(attempts)}"
        return self._finish_cycle(verified, result)

    def _finish_cycle(self, verified: list[VerifiedExperience],
                      result: dict[str, Any]) -> dict[str, Any]:
        """技能练习与工具练习共用的收尾：候选计划→日志→资源闸→冷却→落库。"""
        result["candidate"] = self._maybe_candidate(verified, result)
        self._append(self.root / "autonomous_cycles.jsonl",
                     {**{k: v for k, v in result.items() if k != "capabilities"},
                      "run_id": self.run_id})
        throttled = self._resource_guard(result.get("practice_resource") or {})
        if self.state == BotState.AWAITING_APPROVAL:
            self._transition("COOLDOWN", "候选已进人工审批队列（stable 未变），Bot 继续空闲练习")
        if self.state == BotState.IDLE_PRACTICE:
            self._transition("COOLDOWN", "练习完成进入冷却" +
                             ("（资源超限已加长冷却）" if throttled else ""))
        self.scheduler.mark_idle_done(now=self.monotonic())
        self._trim_store()
        self._persist()
        self.write_status()
        result["state"] = self.state.name
        return result

    def _control_exit(self, ctrl: str, attempts: list[dict[str, Any]], *,
                      goal: str) -> dict[str, Any]:
        if ctrl == "stop":
            (self.root / STOP_FLAG).unlink(missing_ok=True)
            self.stop("控制文件请求停止")
            return {"ran": True, "stopped_early": True, "goal": goal, "attempts": attempts}
        self._transition("PAUSED", "控制文件请求暂停")
        return {"ran": True, "paused": True, "goal": goal, "attempts": attempts}

    def _rotation_exclude(self) -> tuple[str, ...]:
        """只排除最近 rotation_exclude 个族：排除过多会把目标池排空、轮转失效。"""
        k = max(1, int(getattr(self.goals, "rotation_exclude", 2)))
        return tuple(self._recent_families[-k:])

    def _next_practice_task(self, family: str, slot: int) -> dict[str, Any] | None:
        """出一个"没练过、且不在封存集里"的实例；空间穷尽返回 None（如实记录）。"""
        blocked_q = self.sealed_questions(family)
        blocked_fp = self.sealed_fingerprints(family)
        for _ in range(MAX_TASK_RETRY):
            task = make_task(family, self._rng, index=slot)
            fp = task["fingerprint"]
            if task["question"] in blocked_q or fp in blocked_fp or self.store.task_seen(fp):
                continue
            return task
        return None

    # ---- 空闲自主练习循环 ----
    def run_idle_cycle(self, *, dry_run: bool = False) -> dict[str, Any]:
        """一次完整空闲学习机会（发现缺口→练习→oracle 验证→分层记账→[授权时]候选更新）。"""
        if self.state == BotState.ERROR_SAFE:
            return {"ran": False, "reason": "ERROR_SAFE：需显式 nmind bot resume（或 stop）"
                                            "才继续，绝不自动重试可能损坏权重的动作"}
        target = self._select_practice_target() if self.policy.skill_practice_enabled else None
        if dry_run:
            goal = self.goals.next_goal(self.tracker, exclude=self._rotation_exclude())
            if target is not None and target["pick"]["status"] == "SELECTED":
                preview = self._skill_cycle(target, dry_run=True)
                preview["fallback_goal"] = {"goal": goal.goal_id, "family": goal.task_family,
                                            "reason": goal.selection_reason}
                return preview
            return {"ran": False, "dry_run": True,
                    "skill_target": ({"status": target["pick"]["status"],
                                       "reason": target["pick"]["reason"],
                                       "rejected_evidence":
                                       target["pick"]["rejected_evidence"]}
                                      if target else {"status": "GATE_A_UNAVAILABLE",
                                                      "reason": self._gate_a_note}),
                    "plan": {"goal": goal.goal_id, "family": goal.task_family,
                             "reason": goal.selection_reason,
                             "tasks_per_cycle": self.policy.max_tasks_per_idle_cycle,
                             "sealed_per_family": self.sealed_per_family,
                             "candidate": self._candidate_plan(),
                             "network_enabled": self.policy.network_enabled}}
        if not self.policy.idle_learning_enabled:
            self._transition("COOLDOWN", "idle_learning_enabled=false：不练习")
            return {"ran": False, "reason": "idle_learning_enabled=false"}
        if self.state == BotState.PAUSED:
            return {"ran": False, "paused": True,
                    "reason": "pause.flag 仍在：暂停态不得开始练习循环（历史 bug：PAUSED → IDLE_EXPLORE 非法转移直接崩进程）"}
        self.res_window.start()
        if self.state == BotState.COOLDOWN:
            self._transition("READY", "冷却结束，回到就绪")
        self._transition("IDLE_EXPLORE", "空闲：自主选择练习目标")
        # R2：先问"有没有与当前能力匹配、且验证器独立的新技能"。没有就如实休眠
        # （NO_ELIGIBLE_TASK），再退回工具/规则练习——那条路 learning_target=memory_only，
        # 永不冒充神经成长；这是 R1"练了 81 次却把工具成功记在神经账上"的正解。
        if target is not None and target["pick"]["status"] == "SELECTED":
            return self._skill_cycle(target)
        dormancy: dict[str, Any] | None = None
        if target is not None:
            self.dormant_cycles += 1
            dormancy = {"gate_a_verdict": target["gate_a_verdict"],
                        "reason": target["pick"]["reason"],
                        "rejected_evidence": target["pick"]["rejected_evidence"],
                        "note": "NO_ELIGIBLE_TASK 休眠：对齐没问题，只是暂时没有可学的"
                                "相邻技能，这不等于神经学习失败"}
            self.store.event(self.run_id, "no_eligible_task", dormancy)
        elif self.policy.skill_practice_enabled:
            self.store.event(self.run_id, "skill_practice_unavailable",
                              {"reason": self._gate_a_note})
        goal = self.goals.next_goal(self.tracker, exclude=self._rotation_exclude())
        self.store.event(self.run_id, "goal_selected",
                          {"goal_id": goal.goal_id, "family": goal.task_family,
                           "reason": goal.selection_reason,
                           "selection_source": "rule（确定性缺口排序，非学习型探索）"})
        self._transition("IDLE_PRACTICE", f"练习 {goal.task_family}")
        heldout: dict[str, Any] | None = None
        hk = f"{goal.task_family}:{self.stable_sha[:12]}"
        if hk not in self._heldout_done:                  # 每族每权重只测一次基线（省算力）
            heldout = self.measure_heldout(goal.task_family)
            self._heldout_done.add(hk)
        verified: list[VerifiedExperience] = []
        attempts: list[dict[str, Any]] = []
        exhausted = False
        n_tasks = max(1, self.policy.max_tasks_per_idle_cycle)
        for i in range(n_tasks):
            ctrl = self._check_control()
            if ctrl == "stop":
                (self.root / STOP_FLAG).unlink(missing_ok=True)
                self.stop("控制文件请求停止")
                return {"ran": True, "stopped_early": True, "goal": goal.goal_id,
                        "attempts": attempts}
            if ctrl == "pause":
                self._transition("PAUSED", "控制文件请求暂停")
                return {"ran": True, "paused": True, "goal": goal.goal_id,
                        "attempts": attempts}
            if self.inbox_len():
                return self._preempt_for_user(goal, i, attempts, n_tasks)
            task = self._next_practice_task(goal.task_family, self.cycles_done * n_tasks + i)
            if task is None:
                exhausted = True
                self.exhausted_families.add(goal.task_family)
                self.store.event(self.run_id, "task_space_exhausted",
                                  {"family": goal.task_family,
                                   "note": "该族可枚举实例已练完或全部命中封存集"})
                break
            neural_attempt = self.neural_attempt(task["question"])
            tool_result: str | None = None
            tool_calls = 0
            if task.get("tool") and task["tool"] in _TOOL_ALLOWLIST:
                tool_result = execute_tool(task)
                tool_calls = 1
            v_neural = verify(task, neural_attempt)
            v_tool = verify(task, tool_result) if tool_result is not None else None
            # 拿不到独立基准（program_oracle=None）一律 uncertain：不许写进任何成功账
            if program_oracle(task) is None:
                verification, fstate = "uncertain", "unverified"
            elif "verified" in (v_neural, v_tool):
                verification, fstate = "verified", "deterministic_oracle"
            else:
                verification, fstate = "rejected", "deterministic_oracle"
            source = ("neural" if v_neural == "verified" else
                      "rule" if task.get("tool") == "grid_rules" else
                      "tool" if v_tool == "verified" else
                      "abstain" if not neural_attempt else "neural")
            exp = build_experience(task, neural_attempt=neural_attempt, tool_result=tool_result,
                                   verification=verification,
                                   validator_id=f"oracle-{goal.task_family}",
                                   model_sha=self.stable_sha,
                                   exp_id=f"exp-{self.run_id}-{self.cycles_done}-{i}",
                                   skill_id=goal.task_family, response_source=source,
                                   feedback_source=fstate, learning_target="memory_only")
            verified.append(exp)
            self._append(self.root / EXPERIENCE_FILE, asdict(exp))
            self.store.mark_task_done(task["task_id"], goal.task_family, verification,
                                      fingerprint=task["fingerprint"])
            attempts.append({"task_id": task["task_id"], "fingerprint": task["fingerprint"],
                              "family": goal.task_family, "question": task["question"],
                              "oracle": task["answer"], "neural_attempt": neural_attempt,
                              "tool_result": tool_result, "verified_neural": v_neural,
                              "verified_tool": v_tool, "verification": verification,
                              "sealed": False, "feedback_source": fstate,
                              "response_source": source, "skill_id": goal.task_family,
                              "learning_target": "memory_only",
                              "outcome_source": ("neural" if v_neural == "verified" else
                                                 "tool" if v_tool == "verified" else "none")})
            self.tracker.record_attempt(
                capability_id=f"{goal.task_family}:neural", task_family=goal.task_family,
                success=(v_neural == "verified"), path="neural",
                evidence_id=exp.experience_id, cost_s=0.0, model_sha=self.stable_sha,
                skill_id=goal.task_family, response_source="neural",
                feedback_source=fstate, learning_target="memory_only")
            if tool_result is not None:
                self.tracker.record_attempt(
                    capability_id=f"{goal.task_family}:tool", task_family=goal.task_family,
                    success=(v_tool == "verified"), path="tool",
                    evidence_id=exp.experience_id, cost_s=0.0, tool_calls=tool_calls,
                    model_sha=self.stable_sha, skill_id=goal.task_family,
                    response_source=("rule" if task.get("tool") == "grid_rules" else "tool"),
                    feedback_source=fstate, learning_target="memory_only")
        self.cycles_done += 1
        self._recent_families.append(goal.task_family)
        usage = self.res_window.stop("idle_practice")
        self._accumulate_resource(usage)
        result = {"ran": True, "goal": goal.goal_id, "goal_reason": goal.selection_reason,
                  "task_family": goal.task_family, "attempts": attempts,
                  "verified_count": sum(1 for a in attempts if a["verification"] == "verified"),
                  "rejected_count": sum(1 for a in attempts if a["verification"] == "rejected"),
                  "task_space_exhausted": exhausted,
                  "sealed_pool_size": len(self.sealed_set(goal.task_family)),
                  "heldout": heldout,
                  "skill_dormancy": dormancy,
                  "practice_resource": usage,
                  "capabilities": self.tracker.summary()}
        self.last_result = (f"{goal.task_family} verified="
                            f"{result['verified_count']}/{len(attempts)}")
        # ---- 候选更新（只在显式授权 + 硬预算内，走 evolution_manager）----
        candidate = self._maybe_candidate(verified, result)
        result["candidate"] = candidate
        self._append(self.root / "autonomous_cycles.jsonl",
                     {**{k: v for k, v in result.items() if k != "capabilities"},
                      "run_id": self.run_id})
        throttled = self._resource_guard(usage)
        if self.state == BotState.AWAITING_APPROVAL:
            self._transition("COOLDOWN", "候选已进人工审批队列（stable 未变），Bot 继续空闲练习")
        if self.state == BotState.IDLE_PRACTICE:
            self._transition("COOLDOWN", "练习完成进入冷却" +
                             ("（资源超限已加长冷却）" if throttled else ""))
        self.scheduler.mark_idle_done(now=self.monotonic())
        self._trim_store()                     # §9.12 事件表容量上限（每循环一次，成本可忽略）
        self._persist()
        self.write_status()
        result["state"] = self.state.name
        return result

    def _preempt_for_user(self, goal, interrupted_at: int, attempts: list[dict[str, Any]],
                          n_tasks: int) -> dict[str, Any]:
        """§4.1/§9.2 用户消息抢占：在练习实例边界收尾，先答题再回来。

        已完成的实例保留（各自已 mark_task_done），但本轮不计 cycles_done——
        半轮不是"一轮自主练习"，虚报会让日志失真。
        """
        self._transition("READY", f"用户消息到达，中断自主练习（第 {interrupted_at}/{n_tasks} 实例）")
        replies = self.process_user_tasks()
        usage = self.res_window.stop("preempted_by_user")
        self._accumulate_resource(usage)
        self.store.event(self.run_id, "practice_preempted",
                          {"goal": goal.goal_id, "family": goal.task_family,
                           "interrupted_at": interrupted_at, "planned": n_tasks,
                           "answered": len(replies),
                           "practice_attempts_before_preempt": len(attempts)})
        self._persist()
        self.write_status()
        return {"ran": True, "preempted": True, "goal": goal.goal_id,
                "task_family": goal.task_family, "attempts": attempts,
                "user_replies": replies, "state": self.state.name,
                "practice_resource": usage}

    def resource_summary(self) -> dict[str, Any]:
        """整轮累计：CPU 秒、墙钟秒、机器口径占用率、RSS 峰值（交付的 CPU 消耗证据）。"""
        t = dict(self.resource_totals)
        cores = resources.cpu_count()
        wall = float(t.get("wall_s") or 0.0)
        cpu = float(t.get("cpu_s") or 0.0)
        t["cpu_cores"] = cores
        t["cpu_percent_of_machine"] = round(100.0 * cpu / (wall * cores), 1) if wall > 0 else None
        t["limit_cpu_percent"] = self.policy.max_idle_cpu_percent
        t["limit_rss_mb"] = self.policy.max_rss_mb
        t["cpu_cap"] = self.cpu_cap
        return t

    def _accumulate_resource(self, usage: dict[str, Any]) -> None:
        self.resource_totals["wall_s"] = round(self.resource_totals["wall_s"] + usage["wall_s"], 3)
        self.resource_totals["cpu_s"] = round(self.resource_totals["cpu_s"] + usage["cpu_s"], 3)
        self.resource_totals["cycles"] += 1
        peak = usage.get("rss_peak_mb")
        if peak:
            self.resource_totals["rss_peak_mb"] = round(max(self.resource_totals["rss_peak_mb"],
                                                            float(peak)), 1)

    def _apply_cpu_cap(self) -> dict[str, Any]:
        """把 torch 线程数压到策略上限内（"空闲学习不许打扰用户"的唯一硬手段）。

        只设线程数不改代码路径；上限按 max_idle_cpu_percent ÷ 逻辑核数折算。
        """
        cores = resources.cpu_count()
        before = torch.get_num_threads()
        # 留 1 核余量：python 主线程/IO 线程也要 CPU，按精确折算会轻微超限然后触发无谓的 throttle
        want = max(1, int(cores * float(self.policy.max_idle_cpu_percent) / 100.0) - 1)
        torch.set_num_threads(min(before, want))
        info = {"cores": cores, "threads_before": before, "threads_after": torch.get_num_threads(),
                "requested_by_policy": want,
                "max_idle_cpu_percent": self.policy.max_idle_cpu_percent}
        self.store.event(self.run_id, "cpu_cap_applied", info)
        self.cpu_cap = info
        return info

    def _resource_guard(self, usage: dict[str, Any]) -> bool:
        """超上限（机器口径 CPU% / RSS）→ 如实记录并延长冷却，不谎称"在限额内"。"""
        breach: list[str] = []
        pct = usage.get("cpu_percent_of_machine")
        if pct is not None and pct > self.policy.max_idle_cpu_percent:
            breach.append(f"cpu_percent_of_machine {pct} > {self.policy.max_idle_cpu_percent}")
        rss = usage.get("rss_peak_mb")
        if rss and rss > self.policy.max_rss_mb:
            breach.append(f"rss_peak_mb {rss} > {self.policy.max_rss_mb}")
        if not breach:
            return False
        self.store.event(self.run_id, "resource_throttled",
                          {"breach": breach, "usage": usage,
                           "action": "冷却加倍后继续（不静默放宽上限）"})
        self.scheduler.cooldown_s = min(
            self.scheduler.cooldown_s * 2.0,
            max(self.policy.min_cooldown_seconds, 600.0))
        return True

    # ---- 候选更新：授权 + 预算 ----
    @property
    def budget_remaining_steps(self) -> int:
        return max(0, self.authorized_steps_total - self.steps_used)

    def _candidate_plan(self) -> dict[str, Any]:
        return {"authorized": bool(self.policy.candidate_training_authorized
                                   and self.authorized_steps_total > 0),
                "authorized_total_steps": self.authorized_steps_total,
                "steps_used": self.steps_used,
                "steps_remaining": self.budget_remaining_steps,
                "promotion_mode": self.policy.promotion_mode,
                "stable_checkpoint": (self._lang_paths or [None])[0]}

    def maybe_candidate(self, verified: list[VerifiedExperience]) -> dict[str, Any]:
        """公开入口：供 `bot candidate` 显式触发（测试与演示用）。"""
        return self._maybe_candidate(verified, {})

    def _maybe_candidate(self, verified: list[VerifiedExperience],
                         result: dict[str, Any]) -> dict[str, Any]:
        if not self.policy.candidate_training_authorized or not self._lang_paths:
            self._transition("COOLDOWN",
                              "MEMORY_ONLY：未授权神经更新（或无本地权重），经验已入档")
            return {"executed": False,
                    "reason": ("auto_candidate_training=false（默认关闭）"
                               if not self.policy.candidate_training_authorized
                               else "无 stable 语言权重，神经侧 NOT_RUN"),
                    "decision": "MEMORY_ONLY"}
        if self.budget_remaining_steps <= 0:
            self._transition("COOLDOWN", "MEMORY_ONLY：授权预算已用尽")
            return {"executed": False, "decision": "MEMORY_ONLY",
                    "reason": f"授权预算已用尽（累计 {self.steps_used}/{self.authorized_steps_total} 步）",
                    "budget": self._candidate_plan()}
        if not verified:
            self._transition("COOLDOWN", "MEMORY_ONLY：本轮无 verified 经验")
            return {"executed": False, "decision": "MEMORY_ONLY",
                    "reason": "本轮没有 verified 经验（rejected 不触发更新，任务书 §9.7）"}
        pool = self._verified_pool()
        decision = self.scheduler.decide(verified_count=len(pool),
                                         budget_remaining=self.budget_remaining_steps,
                                         now=self.monotonic())
        if decision["decision"] != "CANDIDATE_WEIGHT_UPDATE":
            self._transition("COOLDOWN", f"MEMORY_ONLY：{decision['reasons']}")
            return {"executed": False, "decision": "MEMORY_ONLY",
                    "reasons": decision["reasons"], "pool_size": len(pool)}
        from .evolution_manager import propose_candidate
        self._transition("CANDIDATE_TRAINING",
                          f"{len(pool)} 条 verified 经验触发候选更新（已授权，"
                          f"剩余预算 {self.budget_remaining_steps} 步）")
        family = result.get("task_family") or self.policy.practice_task_allowlist[0]
        prompts, answers = self.sealed_as_rows(family)
        rows_pool = pool
        arms = tuple(self.policy.candidate_arms)
        # 授权预算是跨臂硬总闸：neural 与 negative（标签置换对照）共用同一个
        # AdaptationBudget，因此**本次候选更新执行的 optimizer.step 总数 ≤ 三重上限的最小值**：
        # 剩余授权步数 / 单轮上限 / 调度器单批上限。
        cap = min(self.budget_remaining_steps,
                  int(self.policy.candidate_max_steps),
                  int(self.scheduler.cfg.max_updates_per_batch))
        budget = AdaptationBudget(cap, label=f"bot-candidate-{self.run_id}")
        replay = self._replay_pairs(pool, exclude_family=family)
        try:
            proposal = propose_candidate(
                stable_checkpoint=self._lang_paths[0], tokenizer_path=self._lang_paths[1],
                verified=rows_pool, sealed_prompts=prompts, sealed_answers=answers,
                dev=self.device, out_dir=self.root, updates=cap, arms=arms, budget=budget,
                wall_limit_s=self.policy.candidate_max_wall_s, replay_pairs=replay,
                replay_ratio=self.policy.lambda_replay)
        except Exception as exc:                          # noqa: BLE001 - 任何异常都要落 ERROR_SAFE
            self.error_safe(f"候选更新异常：{type(exc).__name__}: {exc}",
                            traceback.format_exc())
            return {"executed": False, "status": "ERROR_SAFE", "error": str(exc)}
        report = proposal["report"]
        used = int(report.get("budget_used_steps") or 0)
        if used > cap:
            self.error_safe(f"预算越界：本轮实际 {used} 步 > 授权 {cap} 步", "")
        self.steps_used += used
        self.store.set_state("candidate_steps_used", self.steps_used)
        self.candidate_status = proposal["status"]
        self.store.event(self.run_id, "candidate_proposal",
                          {"status": proposal["status"], "steps_used_total": self.steps_used,
                           "gates": report.get("gates"),
                           "new_skill_gain": report.get("new_skill_gain"),
                           "old_task_forgetting_pp": report.get("old_task_forgetting_pp")})
        self._transition("SEALED_EVALUATION", "封存自由生成评测（propose_candidate 内完成）")
        if proposal["status"] == "PROPOSED":
            self._transition("AWAITING_APPROVAL",
                              "门槛通过 → 等待人工批准（nmind bot approve），绝不自动晋升")
        elif proposal["status"] == "ERROR_SAFE":
            self.error_safe("stable checkpoint 训练前后 sha256 不一致",
                            json.dumps(report.get("stable_integrity_violation"), ensure_ascii=False))
        else:
            self._transition("COOLDOWN", f"候选 {proposal['status']}：已隔离，stable 未变")
        return {"executed": True, "status": proposal["status"], "report": report,
                "steps_this_cycle": used, "steps_used_total": self.steps_used,
                "budget": self._candidate_plan(),
                "promotion": "manual（需 nmind bot approve --i-approve）"}

    def _verified_pool(self) -> list[VerifiedExperience]:
        """全部历史 verified 经验（按 content_sha256 去重，保持时间顺序）。"""
        path = self.root / EXPERIENCE_FILE
        if not path.exists():
            return []
        out: list[VerifiedExperience] = []
        seen: set[str] = set()
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("verification_status") != "verified":
                continue
            if rec["content_sha256"] in seen:
                continue
            seen.add(rec["content_sha256"])
            out.append(VerifiedExperience(**rec))
        return out

    def _replay_pairs(self, pool: list[VerifiedExperience], *, exclude_family: str
                      ) -> tuple[list[str], list[str]] | None:
        """旧任务重放：从**其他任务族**的 verified 经验里取，用于测遗忘（任务书 §8）。"""
        from .experience_bridge import to_learning_rows
        # 重放行只取"当初就是拿来训神经"的经验；工具/规则经验进重放=用非神经数据冒充持续训练
        old = [e for e in pool if e.task_family != exclude_family
               and e.learning_target == "neural_generation"]
        if not old:
            return None
        prompts, answers, _skipped = to_learning_rows(old[:8])
        return (prompts, answers) if prompts else None

    # ---- 驻留循环（真实 Windows 后台运行）----
    def run_forever(self, *, max_cycles: int | None = None, deadline_s: float | None = None,
                    poll_s: float = 1.0) -> dict[str, Any]:
        """常驻：用户消息优先 → 空闲则练习 → 否则可中断睡眠（无忙等）。"""
        self.start()
        t0 = self.monotonic()
        cycles = 0
        replies_done = 0
        stop_reason = "未停止（本次调用未被中断）"
        while True:
            ctrl = self._check_control()
            if ctrl is None and self.state == BotState.PAUSED:
                self.resume()                            # 另一个终端 `bot resume` 删掉了 pause.flag
                self.write_status()
            if ctrl == "stop":
                (self.root / STOP_FLAG).unlink(missing_ok=True)
                stop_reason = "控制文件 stop.flag"
                self.stop(stop_reason)
                break
            if ctrl == "pause":
                if self.state != BotState.PAUSED and can_transition(self.state.name, "PAUSED"):
                    self._transition("PAUSED", "控制文件 pause.flag")
                self.write_status()                      # 让另一个终端的 bot status 看到 PAUSED
                if self._sleep(0.5) == "stop":
                    stop_reason = "控制文件 stop.flag"
                    self.resume()
                    self.stop(stop_reason)
                    break
                continue
            replies = self.process_user_tasks()
            replies_done += len(replies)
            if replies:
                continue                                   # 用户任务优先，处理完再看
            if max_cycles is not None and cycles >= max_cycles:
                stop_reason = f"达到 max_cycles={max_cycles}"
                break
            if deadline_s is not None and self.monotonic() - t0 >= deadline_s:
                stop_reason = f"达到 deadline {deadline_s}s"
                break
            if self.scheduler.should_idle_learn(now=self.monotonic(),
                                                user_queue_len=self.inbox_len()):
                out = self.run_idle_cycle()
                cycles += 1
                if out.get("stopped_early"):
                    stop_reason = "循环内响应控制文件（stop.flag）"
                    break
                if out.get("paused"):
                    continue            # 暂停≠退出：回到顶部 pause 分支挂起等待恢复
                if self.state == BotState.ERROR_SAFE:
                    stop_reason = "进入 ERROR_SAFE，停止自主循环"
                    break
            else:
                wait = min(poll_s, max(0.05, self.scheduler.cooldown_remaining(
                    now=self.monotonic())))
                if self._sleep(wait) == "stop":
                    stop_reason = "控制文件 stop.flag"
                    continue
            self.heartbeat()
        return {"cycles": cycles, "user_replies": replies_done, "reason": stop_reason,
                "state": self.state.name, "resource_totals": self.resource_summary()}

    def _sleep(self, seconds: float) -> str | None:
        """可中断睡眠：每 ≤0.5s 检查一次控制文件，绝不 busy-wait。"""
        end = self.monotonic() + max(0.0, seconds)
        while True:
            ctrl = self._check_control()
            if ctrl:
                return ctrl
            left = end - self.monotonic()
            if left <= 0:
                return None
            time.sleep(min(0.5, left))

    def heartbeat(self) -> None:
        self.store.set_state("heartbeat_ts", _now_iso())
        self.write_status()

    # ---- 状态快照 ----
    def status(self) -> dict[str, Any]:
        snap = build_snapshot(state=self.state.name,
                              pid=None if self.state == BotState.STOPPED else os.getpid(),
                              run_id=self.run_id, stable_sha=self.stable_sha,
                              candidate_status=self.candidate_status,
                              cycles_done=self.cycles_done,
                              queue_len=len(self.user_queue) + self.inbox_len(),
                              budget_remaining=self.budget_remaining_steps,
                              last_result=self.last_result,
                              capabilities=self.tracker.summary(),
                              heartbeat_ts=self.store.get_state("heartbeat_ts"))
        snap.update({
            "root": str(self.root),
            "cooldown_remaining_s": self.scheduler.cooldown_remaining(now=self.monotonic()),
            "authorized_steps_total": self.authorized_steps_total,
            "steps_used": self.steps_used,
            "candidate_plan": self._candidate_plan(),
            "resource_totals": self.resource_summary(),
            "cpu_cap": self.cpu_cap,
            "task_instances_done": self.store.completed_count(),
            "sealed_pool_sizes": {f: len(self.sealed_set(f))
                                  for f in self.policy.practice_task_allowlist},
            "exhausted_families": sorted(self.exhausted_families),
            "language_checkpoint": (self._lang_paths or [None])[0],
            "model_meta": self._model_meta,
            "policy": self.policy.to_dict(),
            "note": "capabilities.level 只认实测：tool 成功≠神经掌握；"
                    "无语言权重时 neural 侧一律记空尝试（rejected），不冒充学习"})
        return snap


# 工具白名单：只有这里登记的实现可被空闲循环调用（模型文本永不拼入 shell）。
_TOOL_ALLOWLIST = {"calculator", "json_extractor"}


def read_status(root_dir: str | Path) -> dict[str, Any]:
    """跨进程读状态（`nmind bot status` 用）：status.json + lock.json，并判断是否真在运行。"""
    root = Path(root_dir)
    status_file = root / STATUS_FILE
    if not status_file.exists():
        return {"running": False, "reason": f"{status_file} 不存在：Bot 从未运行过"}
    snap = json.loads(status_file.read_text(encoding="utf-8"))
    lock = root / LOCK_FILE
    pid = int(snap.get("pid") or 0)
    running = lock.exists() and pid > 0 and BotRuntime._pid_alive(pid)
    snap["running"] = running
    snap["lock_present"] = lock.exists()
    return snap


def submit_user_message(root_dir: str | Path, text: str) -> dict[str, Any]:
    """另一终端向运行中的 Bot 投递用户消息（追加 inbox.jsonl；驻留循环在下个安全边界处理）。"""
    root = Path(root_dir)
    root.mkdir(parents=True, exist_ok=True)
    rec = {"ts": _now_iso(), "text": text}
    with (root / INBOX_FILE).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return {"submitted": True, "inbox": str(root / INBOX_FILE)}


def request_control(root_dir: str | Path, action: str) -> dict[str, Any]:
    """另一终端的 pause/resume/stop 控制（写控制文件，运行循环在边界响应）。"""
    root = Path(root_dir)
    root.mkdir(parents=True, exist_ok=True)
    if action == "pause":
        (root / PAUSE_FLAG).write_text(_now_iso(), encoding="utf-8")
    elif action == "resume":
        (root / PAUSE_FLAG).unlink(missing_ok=True)
    elif action == "stop":
        (root / STOP_FLAG).write_text(_now_iso(), encoding="utf-8")
    else:
        raise ValueError(f"未知控制动作 {action!r}")
    return {"action": action, "root": str(root),
            "note": "控制文件在下一个安全边界被响应（练习实例之间），不抢占正在跑的 optimizer step"}
