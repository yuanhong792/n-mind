"""Windows 命令行入口：`nmind bot ...`（任务书 §6 要求的启动/停止/暂停/恢复/状态/日志）。

设计要点：
  * 进程间控制用文件（lock/pause/stop/status/inbox），不引入服务框架，双击 .cmd 即可用；
  * stable 权重默认**复制**进 bot 根目录（results/bot_r1/stable_model.pt）再当稳定模型用，
    晋升/回滚只动这份副本，绝不原地改写 experiments/ 里的 M4/M5 产物；
  * --i-approve / --i-authorize 是显式人工授权参数，缺一个就拒绝执行对应动作。
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any

from .policy import BotPolicy
from .runtime import (
    EXPERIENCE_FILE,
    BotRuntime,
    BotState,
    read_status,
    request_control,
    submit_user_message,
)
from .store import BotStore

DEFAULT_ROOT = "results/bot_r1"
MIRROR_HINT = "（镜像配置；代码侧 src/nmind/bot/policy.py 才是权威）"


def _repo_root() -> Path:
    from ..config import REPO_ROOT
    return Path(REPO_ROOT)


def _policy(args: argparse.Namespace) -> BotPolicy:
    cfg = Path(args.config_file) if getattr(args, "config_file", None) else None
    if cfg is None and getattr(args, "profile", None):
        cfg = _repo_root() / "configs" / f"bot_{args.profile}.yaml"
        if not cfg.exists():
            raise SystemExit(f"失败：找不到 profile 配置 {cfg}")
    if cfg is None:
        cfg = _repo_root() / "configs" / "bot_runtime.yaml"
    if cfg.exists():
        policy, unknown = BotPolicy.from_yaml(cfg)
        if unknown:
            print(f"提示：配置里有未识别键 {unknown}，已忽略 {MIRROR_HINT}", file=__import__("sys").stderr)
        policy.source_config = str(cfg)
        return policy
    return BotPolicy()


def _root(args: argparse.Namespace) -> Path:
    return Path(args.root or _repo_root() / DEFAULT_ROOT).resolve()


def _stable_pair(args: argparse.Namespace, root: Path) -> tuple[Path | None, Path | None]:
    """稳定权重解析：优先 bot 根目录副本；--import-stable 时把外部 checkpoint 复制进来。"""
    stable = root / "stable_model.pt"
    tok = root / "tokenizer.json"
    if args.import_stable and args.import_tokenizer:
        root.mkdir(parents=True, exist_ok=True)
        src_s, src_t = Path(args.import_stable), Path(args.import_tokenizer)
        if not src_s.exists() or not src_t.exists():
            raise SystemExit(f"失败：导入源不存在 {src_s} / {src_t}")
        shutil.copy2(src_s, stable)
        shutil.copy2(src_t, tok)
        print(f"已把稳定权重复制到 {stable}（原文件保持只读，晋升只动这份副本）")
    if args.stable and args.tokenizer:
        return Path(args.stable), Path(args.tokenizer)
    if stable.exists() and tok.exists():
        return stable, tok
    return None, None


def _build(args: argparse.Namespace, *, authorized_steps: int = 0) -> BotRuntime:
    root = _root(args)
    policy = _policy(args)
    if getattr(args, "offline", False):
        policy.network_enabled = False
    if getattr(args, "no_idle", False):
        policy.idle_learning_enabled = False
    stable, tok = _stable_pair(args, root)
    return BotRuntime(root_dir=root, policy=policy, stable_checkpoint=stable,
                      tokenizer_path=tok,
                      authorized_optimizer_steps=authorized_steps)


def _print_json(payload: Any) -> int:
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return 0


# ---------------------------------------------------------------------- 子命令
def cmd_bot_doctor(args: argparse.Namespace) -> int:
    root = _root(args)
    policy = _policy(args)
    stable, tok = _stable_pair(args, root)
    import torch
    root.mkdir(parents=True, exist_ok=True)
    probe = root / ".write_test"
    try:
        probe.write_text("ok", encoding="utf-8")
        writable = True
    except OSError as exc:
        writable = f"不可写：{exc}"
    finally:
        probe.unlink(missing_ok=True)
    checks: dict[str, Any] = {
        "python": __import__("sys").version.split()[0],
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "cpu_logical": __import__("os").cpu_count(),
        "root": str(root),
        "root_writable": writable,
        "stable_checkpoint": str(stable) if stable else None,
        "tokenizer": str(tok) if tok else None,
        "language_ready": bool(stable and tok),
        "network_enabled": policy.network_enabled,
        "auto_candidate_training": policy.candidate_training_authorized,
        "promotion_mode": policy.promotion_mode,
        "allow_remote_model_api": policy.allow_remote_model_api,
        "allow_external_code_execution": policy.allow_external_code_execution,
        "practice_allowlist": list(policy.practice_task_allowlist),
        "gates_pre_registered": {"new_skill_gain_min": 0.10,
                                 "old_task_forgetting_max_pp": 3.0},
        "single_instance": policy.single_instance,
        "lock_present": (root / "lock.json").exists(),
    }
    problems = []
    if not checks["language_ready"]:
        problems.append("没有本地语言权重：神经侧将全部记 NOT_RUN/rejected。"
                        "用 `nmind bot run --import-stable <pt> --import-tokenizer <json>` 导入")
    if checks["lock_present"]:
        snap = read_status(root)
        if snap.get("running"):
            problems.append(f"已有实例在运行 pid={snap.get('pid')}：先 `nmind bot stop`")
    if checks["network_enabled"]:
        problems.append("network_enabled=true：首次运行必须为 false（任务书 §7）")
    checks["problems"] = problems
    checks["verdict"] = "PASS" if not problems else "WARN"
    return _print_json(checks)


def cmd_bot_run(args: argparse.Namespace) -> int:
    bot = _build(args, authorized_steps=int(args.authorize_steps or 0))
    if args.dry_run:
        return _print_json(bot.run_idle_cycle(dry_run=True))
    try:
        out = bot.run_forever(max_cycles=args.max_cycles, deadline_s=args.deadline,
                              poll_s=args.poll)
    except KeyboardInterrupt:
        (bot.root / "stop.flag").write_text("ctrl-c", encoding="utf-8")
        out = {"cycles": bot.cycles_done, "reason": "Ctrl-C → 写 stop.flag 后退出",
               "state": bot.state.name, "resource_totals": bot.resource_summary()}
        bot.stop("Ctrl-C")
    except Exception as exc:
        try:
            bot.error_safe(f"驻留循环异常：{type(exc).__name__}: {exc}",
                           __import__("traceback").format_exc())
        except Exception as state_err:                  # noqa: BLE001 - 状态机本身坏了也要继续解锁
            print(f"[bot] error_safe 本身失败（仍继续解锁）：{type(state_err).__name__}: "
                  f"{state_err}", file=__import__("sys").stderr)
        (bot.root / "lock.json").unlink(missing_ok=True)  # 陈旧锁会让下次启动被拒
        raise
    if bot.state != BotState.STOPPED:
        bot.stop(out.get("reason") or "命令结束（max-cycles/deadline）")
    return _print_json(out)


def cmd_bot_once(args: argparse.Namespace) -> int:
    bot = _build(args, authorized_steps=int(args.authorize_steps or 0))
    bot.start()
    outs = []
    try:
        for _ in range(max(1, args.cycles)):
            outs.append(bot.run_idle_cycle())
    finally:
        bot.stop("once 模式：跑完即退")
    if args.compact:
        for o in outs:
            print(f"[{o.get('task_family')}] verified={o.get('verified_count')} "
                  f"rejected={o.get('rejected_count')} "
                  f"candidate={(o.get('candidate') or {}).get('decision') or (o.get('candidate') or {}).get('status')} "
                  f"cpu={((o.get('practice_resource') or {}).get('cpu_s'))}s")
        return 0
    return _print_json(outs)


def cmd_bot_status(args: argparse.Namespace) -> int:
    root = _root(args)
    snap = read_status(root)
    if args.watch:
        for _ in range(args.watch):
            print(json.dumps(snap, ensure_ascii=False, default=str))
            print("---")
            time.sleep(args.interval)
            snap = read_status(root)
        return 0
    if not snap:
        return _print_json({"running": False, "reason": f"{root} 下没有 status.json"})
    return _print_json(snap)


def cmd_bot_goals(args: argparse.Namespace) -> int:
    bot = _build(args)
    goals = bot.goals.candidates(bot.tracker)
    return _print_json([{"goal_id": g.goal_id, "family": g.task_family,
                         "gap": g.expected_gap, "failure_rate": g.failure_rate,
                         "validator": g.validator_id, "reason": g.selection_reason}
                        for g in goals])


def cmd_bot_pause(args: argparse.Namespace) -> int:
    return _print_json(request_control(_root(args), "pause"))


def cmd_bot_resume(args: argparse.Namespace) -> int:
    root = _root(args)
    if (root / "status.json").exists() and not (root / "lock.json").exists():
        (root / "pause.flag").unlink(missing_ok=True)
        return _print_json({"action": "resume", "note": "Bot 未在运行：已清掉 pause.flag"})
    return _print_json(request_control(root, "resume"))


def cmd_bot_stop(args: argparse.Namespace) -> int:
    return _print_json(request_control(_root(args), "stop"))


def cmd_bot_ask(args: argparse.Namespace) -> int:
    return _print_json(submit_user_message(_root(args), args.text))


def cmd_bot_logs(args: argparse.Namespace) -> int:
    root = _root(args)
    if args.cycles:
        path = root / "autonomous_cycles.jsonl"
        if not path.exists():
            return _print_json({"lines": [], "reason": f"{path} 还不存在（Bot 未跑过空闲循环）"})
        lines = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
        return _print_json(lines[-args.limit:])
    store = BotStore(root / "bot_state.db")
    ev = store.list_events(limit=args.limit, kind=args.kind)
    store.close()
    return _print_json(ev)


def cmd_bot_audit(args: argparse.Namespace) -> int:
    root = _root(args)
    out: dict[str, Any] = {"root": str(root)}
    for name, key in (("model_promotion_audit.jsonl", "promotion_audit"),
                      ("model_lineage.json", "lineage"),
                      ("runtime_transitions.jsonl", "transitions")):
        p = root / name
        if not p.exists():
            out[key] = "NOT_CREATED_YET"
            continue
        if name.endswith(".jsonl"):
            lines = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
            out[key] = lines[-args.limit:]
            out[f"{key}_count"] = len(lines)
        else:
            out[key] = json.loads(p.read_text(encoding="utf-8"))
    rej = root / "rejected"
    out["quarantined_candidates"] = sorted(p.name for p in rej.glob("*.pt")) if rej.exists() else []
    exp = root / EXPERIENCE_FILE
    if exp.exists():
        rows = [json.loads(x) for x in exp.read_text(encoding="utf-8").splitlines() if x.strip()]
        out["experiences"] = {"total": len(rows),
                              "verified": sum(1 for r in rows
                                              if r["verification_status"] == "verified"),
                              "rejected": sum(1 for r in rows
                                              if r["verification_status"] == "rejected")}
    return _print_json(out)


def cmd_bot_candidate(args: argparse.Namespace) -> int:
    """显式授权一次候选更新（不改变"默认关闭"这件事本身）。"""
    if not args.i_authorize:
        return _print_json({"executed": False,
                            "reason": "神经候选更新需要显式 --i-authorize（任务书 §7/§8）",
                            "hint": "例：nmind bot candidate --i-authorize --steps 48"})
    bot = _build(args, authorized_steps=int(args.steps))
    bot.policy.candidate_training_authorized = True
    bot.scheduler.cfg.verified_threshold = min(bot.scheduler.cfg.verified_threshold,
                                               args.verified_threshold)
    bot.start()
    try:
        pool = bot._verified_pool()
        out = bot.maybe_candidate(pool)
    finally:
        bot.stop("candidate 命令结束")
    return _print_json(out)


def cmd_bot_approve(args: argparse.Namespace) -> int:
    """人工批准的原子晋升：只有 --i-approve 才会替换 stable 副本。"""
    from .evolution_manager import approve_promotion
    root = _root(args)
    stable, _tok = _stable_pair(args, root)
    if stable is None:
        return _print_json({"promoted": False,
                            "reason": "没有 stable 副本可晋升：先 `bot run --import-stable ...`"})
    cand = Path(args.candidate) if args.candidate else None
    if cand is None:
        kept = sorted((root / "rejected").glob("*.pt")) if (root / "rejected").exists() else []
        cands = sorted(root.glob("candidate_*.pt"))
        if not cands:
            return _print_json({"promoted": False,
                                 "reason": "没有可晋升候选（REJECTED 候选已被隔离，不自动回捞）",
                                 "quarantined": [p.name for p in kept]})
        cand = cands[-1]
    return _print_json(approve_promotion(candidate_path=cand, stable_checkpoint=stable,
                                         out_dir=root, confirm=bool(args.i_approve)))


def cmd_bot_rollback(args: argparse.Namespace) -> int:
    from .evolution_manager import rollback_stable
    root = _root(args)
    stable, _tok = _stable_pair(args, root)
    return _print_json(rollback_stable(out_dir=root, stable_checkpoint=stable or
                                       root / "stable_model.pt"))


def cmd_bot_eval(args: argparse.Namespace) -> int:
    """封存集上的稳定模型 EM（不训练、不写权重）。"""
    bot = _build(args)
    fams = [args.family] if args.family else list(bot.policy.practice_task_allowlist)
    return _print_json({f: bot.measure_heldout(f) for f in fams})


# ---------------------------------------------------------------------- Bot V1
def _kb_store(args: argparse.Namespace):
    from .knowledge_store import KnowledgeStore
    return KnowledgeStore(_root(args) / "knowledge.db")


def _kb_search(args: argparse.Namespace):
    from .knowledge_search import KnowledgeSearch
    return KnowledgeSearch(_kb_store(args))


def _registry(args: argparse.Namespace):
    from .source_registry import SourceRegistry
    return SourceRegistry(_root(args) / "sources.json")


def _explorer(args: argparse.Namespace):
    """构建与 CLI/GUI 一致的 IdleWebExplorer（offline 或配置授权决定模式）。"""
    from .idle_explorer import IdleWebExplorer
    policy = _policy(args)
    offline = bool(getattr(args, "offline", False)) or not policy.network_enabled
    return IdleWebExplorer(registry=_registry(args), knowledge_store=_kb_store(args),
                           policy=policy, offline=offline), policy, offline


def cmd_bot_ui(args: argparse.Namespace) -> int:
    """Bot V1 桌面界面：默认本地 Web 工作台（用户选定的新 UI 设计）；--tk 走 Tkinter。"""
    if getattr(args, "tk", False):
        from .ui.desktop import launch_desktop
        return launch_desktop(args)
    from .web_server import launch_web_ui
    return launch_web_ui(args)


def cmd_bot_knowledge(args: argparse.Namespace) -> int:
    action = args.kb_cmd
    if action == "search":
        res = _kb_search(args).search(args.query, limit=int(args.limit))
        return _print_json({"query": args.query, "hits": len(res), "results": res})
    if action == "stats":
        return _print_json(_kb_search(args).stats())
    if action == "import-fixture":
        from .knowledge_ingest import import_fixture_pages
        out = import_fixture_pages(_kb_store(args))
        return _print_json({**out, "note": "三篇中文样例均标注 fixture 来源，重复导入不增长计数"})
    if action == "export":
        import json as _json
        store = _kb_store(args)
        rows = []
        for r in store.conn.execute(
                "SELECT * FROM documents WHERE latest=1 ORDER BY fetched_at_utc"):
            doc = dict(r)
            doc["chunks"] = store.chunks_of(doc["doc_id"])
            rows.append(doc)
        Path(args.export_to).write_text(_json.dumps(rows, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
        return _print_json({"exported": len(rows), "out": str(args.export_to)})
    raise SystemExit(f"未知 knowledge 子命令 {action!r}")


def cmd_bot_web(args: argparse.Namespace) -> int:
    action = args.web_cmd
    registry = _registry(args)
    if action == "status":
        explorer, policy, offline = _explorer(args)
        return _print_json({"registry": {"sites": registry.list(),
                                         "exploring_enabled": registry.exploring_enabled,
                                         "policy_hash": registry.policy_hash()},
                            "limits": registry.limits,
                            "offline": offline,
                            "network_enabled": policy.network_enabled,
                            "quota": explorer.quota_snapshot(),
                            "recent_fetches": _kb_store(args).fetch_events(limit=10)})
    if action == "allow":
        rec = registry.add(args.host, path_scope=args.path or "",
                           created_by="cli-user")
        return _print_json({"added": True, "site": rec,
                            "note": "只写授权列表；真实 HTTP 仍需配置 network_enabled 且非 --offline"})
    if action == "deny":
        return _print_json({"removed": registry.remove(args.host)})
    if action == "fetch":
        explorer, policy, offline = _explorer(args)
        rec = explorer.fetch_for_user(args.url)
        return _print_json(rec)
    if action == "explore":
        """CLI 手动触发限量探索（受白名单/配额/fixture 或真实授权约束）。"""
        from .store import BotStore  # noqa: F401 - 与 GUI 同源；运行时经 runtime 驱动
        explorer, policy, offline = _explorer(args)
        explorer.set_enabled(True)
        root = _root(args)
        runtime = BotRuntime(root_dir=root, policy=policy,
                             stable_checkpoint=None, tokenizer_path=None)
        runtime.start()
        outs = []
        try:
            for _i in range(max(1, args.cycles)):
                out = explorer.run_idle_exploration(runtime)
                outs.append(out)
                if out.get("stopped_early") or out.get("preempted"):
                    break
        finally:
            runtime.stop("web explore 命令结束")
        return _print_json({"cycles": len(outs), "results": outs})
    if action == "test-security":
        from .web_security_probe import run_probes
        return _print_json(run_probes())
    raise SystemExit(f"未知 web 子命令 {action!r}")


def cmd_bot_sessions(args: argparse.Namespace) -> int:
    from .client import BotClient
    client = BotClient(_root(args))
    try:
        if args.sess_cmd == "list":
            return _print_json(client.sessions())
        if args.sess_cmd == "delete":
            n = client.delete_session(args.session_id)
            return _print_json({"deleted": args.session_id, "messages_removed": n})
        if args.sess_cmd == "history":
            return _print_json(client.history(args.session_id, limit=int(args.limit)))
        raise SystemExit(f"未知 sessions 子命令 {args.sess_cmd!r}")
    finally:
        client.close()


# ---------------------------------------------------------------------- 注册
def register(sub: argparse.Action, parents: list[argparse.ArgumentParser]) -> None:
    bp = sub.add_parser("bot", help="Agent Runtime R1：独立 Bot（自主练习/受控候选学习）",
                        parents=parents)
    bsub = bp.add_subparsers(dest="bot_cmd", required=True)

    def add(name: str, fn, help_text: str) -> argparse.ArgumentParser:
        sp = bsub.add_parser(name, help=help_text)
        # 全局开关也要能写在叶子子命令后面：项目其余命令都是"前后都收"的口径
        # （_add_common 用 SUPPRESS 默认值，所以后写不会覆盖前写）。
        # 缺陷实例：文档里 `bot once --offline --cycles 20` 照敲会 argparse error。
        from ..cli import _add_common
        _add_common(sp, suppress=True)
        sp.add_argument("--root", default=None, help=f"Bot 状态目录（默认 {_repo_root() / DEFAULT_ROOT}）")
        sp.add_argument("--config-file", default=None, help="configs/bot_runtime.yaml 路径")
        sp.add_argument("--profile", default=None, choices=("safe-local", "demo", "candidate-demo"),
                        help="按名字选 configs/bot_<profile>.yaml（safe-local=默认安全档）")
        sp.add_argument("--stable", default=None, help="直接指定 stable checkpoint（跳过副本）")
        sp.add_argument("--tokenizer", default=None, help="配套 tokenizer JSON")
        sp.add_argument("--import-stable", default=None, help="把外部 checkpoint 复制为 stable 副本")
        sp.add_argument("--import-tokenizer", default=None, help="配套 tokenizer 复制源")
        sp.add_argument("--no-idle", action="store_true", help="关掉空闲学习（只服务用户）")
        sp.set_defaults(fn=fn)
        return sp

    add("doctor", cmd_bot_doctor, "体检：依赖/线程/权重/策略/锁")
    sp = add("run", cmd_bot_run, "常驻运行（用户优先，空闲自动练习）")
    sp.add_argument("--max-cycles", type=int, default=None)
    sp.add_argument("--deadline", type=float, default=None, help="墙钟上限秒")
    sp.add_argument("--poll", type=float, default=1.0)
    sp.add_argument("--authorize-steps", type=int, default=0,
                    help="本次会话授权的候选更新总步数硬预算（0=不授权）")
    sp.add_argument("--dry-run", action="store_true", help="只打印计划，不写任何文件")
    sp = add("once", cmd_bot_once, "跑 N 轮空闲循环后退出")
    sp.add_argument("--cycles", type=int, default=1)
    sp.add_argument("--compact", action="store_true", help="每轮一行摘要")
    sp.add_argument("--authorize-steps", type=int, default=0)
    sp = add("status", cmd_bot_status, "查看状态（跨进程读 status.json）")
    sp.add_argument("--watch", type=int, default=0, help="连续打印 N 次")
    sp.add_argument("--interval", type=float, default=2.0)
    add("goals", cmd_bot_goals, "当前目标池与选择理由（selection_source=rule）")
    add("pause", cmd_bot_pause, "暂停（写 pause.flag）")
    add("resume", cmd_bot_resume, "恢复（清 pause.flag）")
    add("stop", cmd_bot_stop, "停止（写 stop.flag，运行循环在安全边界退出）")
    sp = add("ask", cmd_bot_ask, "给运行中的 Bot 发一条用户消息（优先处理）")
    sp.add_argument("text")
    sp = add("logs", cmd_bot_logs, "学习日志（事件或自主循环记录）")
    sp.add_argument("--limit", type=int, default=20)
    sp.add_argument("--kind", default=None, help="只看某类事件，如 transition")
    sp.add_argument("--cycles", action="store_true", help="看 autonomous_cycles.jsonl")
    sp = add("audit", cmd_bot_audit, "晋升审计、世系、隔离区、经验计数")
    sp.add_argument("--limit", type=int, default=10)
    sp = add("eval", cmd_bot_eval, "封存集上的自由生成 EM（不训练）")
    sp.add_argument("--family", default=None)
    sp = add("candidate", cmd_bot_candidate, "显式授权一次候选更新（需 --i-authorize）")
    sp.add_argument("--i-authorize", action="store_true", dest="i_authorize")
    sp.add_argument("--steps", type=int, default=48)
    sp.add_argument("--verified-threshold", type=int, default=32, dest="verified_threshold",
                    help="本次触发阈值（默认沿用预登记的 32；调低要如实写进报告）")
    sp = add("approve", cmd_bot_approve, "人工批准晋升（需 --i-approve，只动 stable 副本）")
    sp.add_argument("--candidate", default=None)
    sp.add_argument("--i-approve", action="store_true", dest="i_approve")
    add("rollback", cmd_bot_rollback, "回滚 stable 副本到上一次晋升前备份")

    # ---- Bot V1：桌面窗口 / 知识库 / 受控联网 / 会话 ----
    # 项目风格：公共开关加在叶子子命令上（否则 `bot web fetch --root ...` 无法解析）
    def add_v1(name: str, fn, help_text: str) -> argparse.ArgumentParser:
        sp = bsub.add_parser(name, help=help_text, parents=parents)
        sp.set_defaults(fn=fn)
        return sp

    def v1_common(sp: argparse.ArgumentParser) -> argparse.ArgumentParser:
        from ..cli import _add_common
        _add_common(sp, suppress=True)
        sp.add_argument("--root", default=None,
                        help=f"Bot 状态目录（默认 {_repo_root() / DEFAULT_ROOT}）")
        sp.add_argument("--config-file", default=None)
        sp.add_argument("--profile", default=None)
        return sp

    sp = add_v1("ui", cmd_bot_ui, "打开 Windows 中文聊天桌面窗口（默认本地 Web 工作台，完全离线）")
    sp.add_argument("--port", type=int, default=8765)
    sp.add_argument("--root", default=None)
    sp.add_argument("--tk", action="store_true",
                    help="使用 Tkinter 原生窗口（默认使用新设计的 Web 工作台）")
    kb = add_v1("knowledge", None, "中文知识库（离线检索/统计/夹具导入/导出）")
    ksub = kb.add_subparsers(dest="kb_cmd", required=True)
    sp = v1_common(ksub.add_parser("search", parents=parents,
                                   help="中文关键词检索（返回摘录+来源+分数）"))
    sp.add_argument("query")
    sp.add_argument("--limit", type=int, default=5)
    sp.set_defaults(fn=cmd_bot_knowledge)
    sp = v1_common(ksub.add_parser("stats", parents=parents,
                                   help="文档/片段/去重统计"))
    sp.set_defaults(fn=cmd_bot_knowledge)
    sp = v1_common(ksub.add_parser("import-fixture", parents=parents,
                                   help="导入三篇中文测试资料（标注 fixture）"))
    sp.set_defaults(fn=cmd_bot_knowledge)
    sp = v1_common(ksub.add_parser("export", parents=parents,
                                   help="导出全部最新版文档与片段为 JSON"))
    sp.add_argument("--export-to", default="knowledge_export.json")
    sp.set_defaults(fn=cmd_bot_knowledge)
    web = add_v1("web", None, "受控联网（默认关；白名单+限额+审计）")
    wsub = web.add_subparsers(dest="web_cmd", required=True)
    sp = v1_common(wsub.add_parser("status", parents=parents,
                                   help="网络授权状态/配额/最近抓取"))
    sp.set_defaults(fn=cmd_bot_web)
    sp = v1_common(wsub.add_parser("allow", parents=parents,
                                   help="添加允许站点（精确域名，可选 /path 范围）"))
    sp.add_argument("host")
    sp.add_argument("--path", default="")
    sp.set_defaults(fn=cmd_bot_web)
    sp = v1_common(wsub.add_parser("deny", parents=parents, help="移除允许站点"))
    sp.add_argument("host")
    sp.set_defaults(fn=cmd_bot_web)
    sp = v1_common(wsub.add_parser("fetch", parents=parents,
                                   help="用户主动只读读取一个 URL（先检查白名单）"))
    sp.add_argument("url")
    sp.set_defaults(fn=cmd_bot_web)
    sp = v1_common(wsub.add_parser("explore", parents=parents,
                                   help="手动跑 N 轮限量探索（fixture/受控源）"))
    sp.add_argument("--cycles", type=int, default=1)
    sp.set_defaults(fn=cmd_bot_web)
    sp = v1_common(wsub.add_parser("test-security", parents=parents,
                                   help="运行内置网络安全探针（纯本地，零真实请求）"))
    sp.set_defaults(fn=cmd_bot_web)
    sess = add_v1("sessions", None, "聊天会话管理（列表/删除/历史）")
    ssub = sess.add_subparsers(dest="sess_cmd", required=True)
    sp = v1_common(ssub.add_parser("list", parents=parents))
    sp.set_defaults(fn=cmd_bot_sessions)
    sp = v1_common(ssub.add_parser("delete", parents=parents))
    sp.add_argument("session_id")
    sp.set_defaults(fn=cmd_bot_sessions)
    sp = v1_common(ssub.add_parser("history", parents=parents))
    sp.add_argument("session_id")
    sp.add_argument("--limit", type=int, default=50)
    sp.set_defaults(fn=cmd_bot_sessions)
