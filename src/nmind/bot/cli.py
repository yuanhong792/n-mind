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
from .runtime import (EXPERIENCE_FILE, BotRuntime, BotState, read_status,
                      request_control, submit_user_message)
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
    except Exception as exc:                              # noqa: BLE001 - 崩溃必须留下证据并解锁
        try:
            bot.error_safe(f"驻留循环异常：{type(exc).__name__}: {exc}",
                           __import__("traceback").format_exc())
        except Exception:                                 # noqa: BLE001 - 状态机本身坏了也要继续解锁
            pass
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
