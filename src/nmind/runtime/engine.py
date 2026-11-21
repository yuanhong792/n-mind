"""NMindEngine：指令第 6 节运行逻辑的唯一实现处。

    observe -> encode -> update local workspace -> select blocks ->
    retrieve memory -> local decision -> validate -> (optional cloud) ->
    record episode -> consolidation

层次说清楚，避免"数据库记住了"被说成"神经网络学会了"：
  * 神经网络给出的 = 动作 logits / 置信度 / 预测误差 / 路由与预算统计；
  * SQLite 给出的 = 可精确召回的历史与事实（检索命中数、命中标题）；
  * 两者对最终 action 的贡献是**分开记录**的（decision_source 字段），
    消融开关 use_memory=False 时记忆通道彻底置零，用来量出记忆到底值多少。

云端默认关闭（cfg.cloud.mode=off），--offline 是硬开关，见 config.effective_cloud_mode。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

import numpy as np
import torch

from ..bridge.policy import Auditor, CloudPolicy
from ..bridge.privacy import build_payload
from ..bridge.providers import CloudResult, CloudRequest, build_provider
from ..config import Config, default_config
from ..core.architecture import NModel
from ..core.encoder import EventEncoder
from ..language.interface import describe_event, parse_rule, verbalize_decision
from ..memory import Consolidator, MemoryRetriever, MemoryStore
from .governor import ComputeGovernor
from .hardware import HardwareReport, detect

GRID = 6                      # 与 learning.datasets 的 6x6 网格一致


@dataclass
class Outcome:
    tick: int
    action: int
    action_name: str
    probabilities: list[float]
    confidence: float
    margin: float
    error: float
    k: int
    blocks_touched: int
    step_ms: float
    validation: dict[str, Any]
    memory_hits: int
    memory_titles: list[str]
    decision_source: str
    cloud_used: bool
    cloud_summary: dict[str, Any]
    text: str
    episode_id: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in self.__dict__.items()}


@dataclass
class SequenceResult:
    """一次多步运行后留给训练与评测的张量（不含任何云端内容）。"""

    obs: np.ndarray                       # [T, D]
    actions: np.ndarray                   # [T] 模型选出的动作
    targets: np.ndarray | None            # [T] oracle 动作（有标签时）
    goal: tuple[int, int]
    start: tuple[int, int]
    outcomes: list[Outcome] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        if self.targets is None:
            raise ValueError("无标签序列不计算准确率")
        return float(np.mean(self.actions == self.targets))

    @property
    def occluded_accuracy(self) -> float:
        """遮挡期间的准确率：这一段才真正考"记忆"，输入里已经没有目标了。"""
        if self.targets is None:
            raise ValueError("无标签序列不计算准确率")
        occ = self.obs[:, -2] > 0.5
        if not occ.any():
            return float("nan")
        return float(np.mean(self.actions[occ] == self.targets[occ]))


ACTION_NAMES = ("up", "right", "down", "left")


class NMindEngine:
    def __init__(self, cfg: Config | None = None, *, model: NModel | None = None,
                 store: MemoryStore | None = None, use_memory: bool = True,
                 session_id: str = "default", seed: int | None = None,
                 text_source: str | None = None, verbose: bool = False) -> None:
        self.cfg = cfg or default_config()
        self.use_memory = bool(use_memory and self.cfg.memory.episodic_enabled)
        self.session_id = session_id
        self.verbose = verbose
        self.seed = int(seed if seed is not None else self.cfg.runtime.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed % (2**32))

        self.device, self.hardware = detect(self.cfg.runtime.device,
                                            torch_threads=self.cfg.runtime.torch_threads)
        self.model = model or NModel(self.cfg.model, n_actions=4, mem_dim=16)
        self.model.to(self.device)
        self.model.eval()
        # Governor 的上下限来自配置与模型形状：模型自己不能决定"多算一点"的上限。
        self.governor = ComputeGovernor(
            k_default=self.cfg.model.top_k,
            k_min=1,
            k_max=self.cfg.model.n_blocks,
            max_step_ms=self.cfg.runtime.max_step_ms,
            max_rss_mb=self.cfg.runtime.max_rss_mb,
            max_depth=self.cfg.runtime.max_recurrence_depth,
        )
        self.text_source = text_source or self.cfg.language.source
        self._text_proj: Callable[[str], torch.Tensor] | None = self._build_text_projector()

        self.store = MemoryStore(self.cfg.memory.db_path) if self.use_memory else None
        if self.store is None and store is not None:
            self.store = store                     # 测试/训练可注入同一个库
        self.retriever = MemoryRetriever(self.store, top_k=self.cfg.memory.retrieval_top_k) \
            if self.use_memory and self.store is not None else None
        self.consolidator = Consolidator(self.store) if self.use_memory and self.store is not None else None

        self.policy = CloudPolicy(self.cfg)
        self.auditor = Auditor(self.cfg.resolve(self.cfg.cloud.audit_log),
                               mask=self.cfg.logging.mask_secrets)
        self.provider = build_provider(self.cfg.effective_cloud_mode(), self.cfg)
        self.cloud_calls = 0
        self.cloud_results: list[dict[str, Any]] = []
        self.state = self.model.initial_state(1, self.device)
        self._last_goal: tuple[int, int] | None = None
        self._pos: tuple[int, int] | None = None
        self._size = 6

    # ------------------------------------------------------------------ 文本特征

    def _build_text_projector(self) -> Callable[[str], torch.Tensor] | None:
        """文本通道：默认哈希特征；配置成本地适配器且可用时才换来源，否则自动降级。"""
        dim = self.cfg.model.event_text_dim
        if self.text_source != "local_adapter":
            return lambda s: EventEncoder.text_features(s, dim).to(self.device)
        from ..language.local_adapter import LocalTextAdapter

        adapter = LocalTextAdapter(self.cfg)
        if not adapter.status.available:
            self.hardware.notes.append(f"语言适配器不可用（{adapter.status.reason}），退回哈希文本特征")
            self.text_source = "hash"
            return lambda s: EventEncoder.text_features(s, dim).to(self.device)

        def _via_adapter(s: str) -> torch.Tensor:
            feat = adapter.encode([s])
            if feat is None:
                return EventEncoder.text_features(s, dim).to(self.device)
            # 外部模型维度未知且随模型变化，用一个延迟定型的线性层投到 event_text_dim。
            # 这是**只读投影**（不在推理时更新），随机初始化后固定，别把它说成学到的表征。
            proj = getattr(self, "_text_proj_layer", None)
            if proj is None or proj.in_features != int(feat.shape[-1]):
                proj = torch.nn.Linear(int(feat.shape[-1]), dim)
                with torch.no_grad():
                    proj.weight.mul_(0.02).add_(torch.randn_like(proj.weight) * 0.02)
                proj = proj.to(self.device).eval()
                self._text_proj_layer = proj
            with torch.no_grad():
                out = proj(torch.tensor(feat, dtype=torch.float32, device=self.device))
            return out

        self._adapter = adapter
        return _via_adapter

    # ------------------------------------------------------------------ 记忆通道

    def _mem_vector(self, query: str) -> tuple[torch.Tensor, list[Any]]:
        if self.retriever is None:
            return torch.zeros(1, 16, device=self.device), []
        vec = self.retriever.vector_context(query, dim=16, session_id=self.session_id)
        hits = self.retriever.search(query, limit=self.cfg.memory.retrieval_top_k,
                                     session_id=self.session_id)
        return torch.tensor(vec, dtype=torch.float32, device=self.device).unsqueeze(0), hits

    def _remember(self, obs_text: str, action_name: str, outcome: str, *,
                  claims: list[list[str]], confidence: float, privacy: str = "normal") -> int:
        if self.store is None:
            return 0
        return self.store.add_episode(
            session_id=self.session_id, task="grid", observation=obs_text,
            action=action_name, outcome=outcome, feedback=0.0, source="local",
            confidence=confidence, privacy_level=privacy,
            observation_json={"claims": claims})

    def set_text(self, text: str) -> bool:
        """把一句中文规则变成主张，等下一步写进记忆。**不直接改网络权重。**

        是否成为"事实"由巩固的重复次数与反馈门槛决定（memory/consolidation.py）。
        """
        rule = parse_rule(text)
        if not rule.matched:
            if self.verbose:
                print(f"  [语言] 没解析出结构化含义：未识别片段 {rule.unknown_spans}")
            return False
        claims = rule.as_claims()
        if not claims:
            return False
        self._pending_claims = claims
        self._pending_text = text
        if self.verbose:
            print(f"  [语言] 记为待巩固主张：{claims}")
        return True

    # ------------------------------------------------------------------ 主循环

    def step(self, numeric: Any, *, action: int | None = None, text: str | None = None,
             goal_visible: bool | None = None, cloud_threshold: float = 0.34,
             record: bool = True) -> Outcome:
        """纵向闭环一步：预测误差 → 路由 → 检索记忆 → 决策 → 校验 → （可选）云端 → 落库。"""
        v = np.asarray(numeric, dtype=np.float32).reshape(1, -1)
        if v.shape[1] != self.cfg.model.event_numeric_dim:
            raise ValueError(f"观测维度应为 {self.cfg.model.event_numeric_dim}，实际 {v.shape[1]}")
        occluded = bool(v[0, 24] > 0.5) if v.shape[1] >= 25 else False
        self._pos = _argmax_pair(v[0], 0, 6)
        if not occluded:
            # 观测布局（learning/datasets.py）：[12:18] 目标 x、[18:24] 目标 y。
            # 曾误写成 (18,24)：那会把"目标 y"当成 x、把遮挡/碰撞标志当成 y，
            # 校验层于是拿着一个不存在的目标坐标算距离——结论看着合理其实全错。
            g = _argmax_pair(v[0], 12, 18)
            if g is not None:
                self._last_goal = g

        obs_text = describe_event(v[0], size=self._size)
        query = (text or obs_text)
        mem, hits = self._mem_vector(query)
        tf = None if self._text_proj is None else self._text_proj(query)
        err_scalar = float(getattr(self, "_last_err", 0.5))     # 上一步的预测误差，本步的 k 依据它

        with torch.no_grad():
            budget = self.governor.allocate(err_scalar)
            action_prev = torch.tensor([0 if action is None else int(action)],
                                       dtype=torch.long, device=self.device)
            # 计时交给 Governor：它维护滑动窗口，"最近变慢就降 k"这条规则
            # 只有从这里拿到真实耗时才可能生效（自己另测一份数字等于两份口径）。
            self.governor.start_step()
            out, self.state = self.model.step(torch.tensor(v, dtype=torch.float32, device=self.device),
                                              self.state, action_prev,
                                              text_feat=tf, mem=mem,
                                              k=budget.k, depth=budget.depth)
            step_ms = self.governor.end_step()
            probs = out.decision["probabilities"].reshape(-1).tolist()
            idx = int(out.decision["action_index"][0])
            err = float(out.error.reshape(-1).mean())
        self._last_err = err

        validation = _validate(idx, self._pos, self._last_goal, occluded)
        cloud_used = False
        cloud_summary: dict[str, Any] = {"used": False}
        margin = float(out.decision["margin"])
        if 0.0 <= margin < cloud_threshold and self.cfg.effective_cloud_mode() != "off":
            cloud_summary = self._ask_cloud(query, obs_text, hits, margin)
            cloud_used = bool(cloud_summary.get("used"))

        text_out = verbalize_decision(
            {"action_index": idx, "confidence": float(out.decision["confidence"]), "margin": margin},
            event_text=obs_text, retrieved=len(hits), cloud_used=cloud_used,
            degraded=budget.degraded)

        episode_id = None
        if record and self.store is not None:
            claims = getattr(self, "_pending_claims", []) or _claims_from(v[0], idx, occluded)
            episode_id = self._remember(obs_text, ACTION_NAMES[idx],
                                         "validated" if validation["ok"] else "invalid",
                                         claims=claims,
                                         confidence=float(out.decision["confidence"]))
            if getattr(self, "_pending_claims", None):
                self._pending_claims = []
                self._pending_text = ""

        oc = Outcome(tick=int(self.state.tick), action=idx, action_name=ACTION_NAMES[idx],
                     probabilities=[round(float(p), 4) for p in probs],
                     confidence=round(float(out.decision["confidence"]), 4), margin=round(margin, 4),
                     error=round(err, 4), k=out.k, blocks_touched=out.blocks_touched,
                     step_ms=round(step_ms, 3), validation=validation,
                     memory_hits=len(hits), memory_titles=[h.text[:36] for h in hits],
                     decision_source=("memory+network" if hits else "network"),
                     cloud_used=cloud_used, cloud_summary=cloud_summary,
                     text=text_out, episode_id=episode_id)
        if self.verbose:
            print(f"  [t{oc.tick}] {oc.text} (k={oc.k} err={oc.error:.3f} "
                  f"命中记忆={oc.memory_hits} 校验={validation['ok']})")
        return oc

    def run_sequence(self, traj: Any, *, cloud_threshold: float = 0.34,
                     record: bool = False) -> SequenceResult:
        """跑一条已生成轨迹：动作由模型选，oracle 标签只用于打分，绝不喂给模型。"""
        obs = np.asarray(traj.obs, dtype=np.float32)
        targets = np.asarray(traj.actions, dtype=np.int64)
        chosen = np.zeros(len(targets), dtype=np.int64)
        outcomes: list[Outcome] = []
        for t in range(obs.shape[0]):
            prev = int(chosen[t - 1]) if t else None
            oc = self.step(obs[t], action=prev, cloud_threshold=cloud_threshold, record=record)
            chosen[t] = oc.action
            outcomes.append(oc)
        return SequenceResult(obs=obs, actions=chosen, targets=targets,
                              goal=tuple(traj.goal), start=tuple(traj.start), outcomes=outcomes)

    def rollout(self, *, T: int, seed: int = 0, cloud_threshold: float = 0.34) -> SequenceResult:
        """环境无关的交互 rollout（M1 CLI 演示用）：模型选动作，环境给下一观测。"""
        from ..learning.datasets import GridTask, DIRS, _encode, oracle_action

        rng = np.random.default_rng(seed)
        task = GridTask(goal_visible_steps=3, seed=seed)
        obstacles = set(task.obstacles)
        pos = (int(rng.integers(0, 6)), int(rng.integers(0, 6)))
        goal = (int(rng.integers(0, 6)), int(rng.integers(0, 6)))
        obs_list: list[np.ndarray] = []
        acts: list[int] = []
        outcomes: list[Outcome] = []
        collided = False
        chosen_prev: int | None = None
        for _t in range(T):
            occluded = len(obs_list) >= task.goal_visible_steps
            v = _encode(pos[0], pos[1], goal[0], goal[1], occluded, collided)
            obs_list.append(v)
            oc = self.step(v, action=chosen_prev, cloud_threshold=cloud_threshold, record=True)
            acts.append(oc.action)
            outcomes.append(oc)
            chosen_prev = oc.action
            dx, dy = DIRS[oc.action]
            nx, ny = pos[0] + dx, pos[1] + dy
            if 0 <= nx < 6 and 0 <= ny < 6 and (nx, ny) not in obstacles:
                pos, collided = (nx, ny), False
            else:
                collided = True
        return SequenceResult(obs=np.stack(obs_list), actions=np.array(acts, dtype=np.int64),
                              targets=None, goal=goal, start=pos, outcomes=outcomes)

    def evaluate_trajs(self, trajs: Iterable[Any], *, occluded_only: bool = False) -> dict[str, float]:
        """一组轨迹上的准确率；occluded_only 只看遮挡段（记忆真实贡献）。"""
        tot = hit = 0
        from ..learning.datasets import oracle_action

        for tr in trajs:
            obs, targets = np.asarray(tr.obs), np.asarray(tr.actions)
            mask = (obs[:, -2] > 0.5) if occluded_only else np.ones(len(targets), dtype=bool)
            if not mask.any():
                continue
            for t in np.flatnonzero(mask):
                prev = int(targets[t - 1]) if t else None
                step_out = self.step(obs[t], action=prev, record=False)
                tot += 1
                hit += int(step_out.action == int(targets[t]))
        return {"n": float(tot), "accuracy": (hit / tot) if tot else float("nan")}

    # ------------------------------------------------------------------ 云端

    def _ask_cloud(self, query: str, obs_text: str, hits: list[Any], margin: float) -> dict[str, Any]:
        levels = self.retriever.privacy_snapshot(hits)["levels"] if self.retriever else {}
        payload = build_payload(category="public_web_query", question=obs_text,
                                context=[h.text for h in hits],
                                numeric={"margin": round(margin, 4), "goal_known": self._last_goal},
                                privacy_levels=levels,
                                max_chars=self.cfg.cloud.max_payload_chars)
        decision = self.policy.evaluate(payload)
        self.auditor.record(session_id=self.session_id, decision=decision,
                            question_preview=obs_text)
        if not decision.allowed:
            return {"used": False, "reason": decision.reason, "blocked": decision.blocked}
        req = CloudRequest(category=payload.category, question=payload.question,
                           context_summary=payload.context_summary,
                           numeric_summary=payload.numeric_summary,
                           timeout_s=self.cfg.cloud.timeout_s)
        result: CloudResult = self.provider.ask(req)
        self.cloud_calls += 1
        self.policy.commit(decision)
        summary = result.summary()
        summary.update({"used": result.ok, "reason": "allowed", "margin": round(margin, 4)})
        self.cloud_results.append(summary)
        if not result.ok:
            return {"used": False, "reason": f"provider_error:{result.error}", "degraded": True}
        return summary

    # ------------------------------------------------------------------ 巩固与统计

    def consolidate(self, **kw: Any) -> dict[str, Any]:
        if self.consolidator is None:
            return {"skipped": "记忆层未启用（use_memory=False 或 episodic_enabled=False）"}
        return self.consolidator.run(**kw)

    def close(self) -> None:
        """释放记忆库连接。长驻进程里反复建引擎时必须调用，否则留一堆 WAL 句柄。"""
        if self.store is not None:
            self.store.close()
            self.store = None
            self.retriever = None
            self.consolidator = None

    def __enter__(self) -> "NMindEngine":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def stats(self) -> dict[str, Any]:
        rep = self.governor.report          # 是 dataclass 实例，不是方法：别加括号
        return {"governor": rep.summary(), "model": self.model.active_block_stats(),
                "params": self.model.n_params()["total"],
                "memory": self.store.stats() if self.store else {"disabled": True},
                "cloud": {**self.policy.status(), "calls": self.cloud_calls},
                "text_source": self.text_source, "use_memory": self.use_memory,
                "device": str(self.device), "session": self.session_id}


def _argmax_pair(v: np.ndarray, a: int, b: int) -> tuple[int, int] | None:
    xs, ys = v[a:a + 6], v[b:b + 6]
    if xs.max() < 0.5 or ys.max() < 0.5:
        return None
    return (int(np.argmax(xs)), int(np.argmax(ys)))


def _claims_from(v: np.ndarray, action: int, occluded: bool) -> list[list[str]]:
    """从当前观测里能**直接观测到**的主张（遮挡期不猜目标，避免编造）。"""
    pos = _argmax_pair(v, 0, 6)
    if pos is None:
        return []
    claims = [[f"cell:{pos[0]}:{pos[1]}", "moves", ACTION_NAMES[int(action)]]]
    goal = _argmax_pair(v, 12, 18)
    if goal is not None:
        claims.append([f"cell:{pos[0]}:{pos[1]}", "goal_visible_at", f"{goal[0]}:{goal[1]}"])
    return claims


def _validate(idx: int, pos: tuple[int, int] | None, goal: tuple[int, int] | None,
              occluded: bool) -> dict[str, Any]:
    """规则校验层：动作是否让到目标的曼哈顿距离变小。

    遮挡期没有可见目标，只能用"记忆中最后可见的目标"，因此这里同时报告
    校验依据（observed / remembered / unknown）—— 让审计能分清证据强度。
    """
    from ..learning.datasets import DIRS

    if pos is None:
        return {"ok": False, "reason": "position_unreadable", "basis": "unknown"}
    dx, dy = DIRS[idx]
    nxt = (pos[0] + dx, pos[1] + dy)
    in_bounds = 0 <= nxt[0] < GRID and 0 <= nxt[1] < GRID
    if not in_bounds:
        return {"ok": False, "reason": "out_of_bounds", "basis": "observed"}
    if goal is None:
        return {"ok": True, "reason": "no_goal_yet", "basis": "unknown"}
    d_now = abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
    d_next = abs(nxt[0] - goal[0]) + abs(nxt[1] - goal[1])
    basis = "remembered" if occluded else "observed"
    return {"ok": d_next <= d_now, "reason": "distance_improved" if d_next < d_now
            else ("distance_stayed" if d_next == d_now else "distance_increased"),
            "basis": basis, "dist_now": d_now, "dist_next": d_next}
