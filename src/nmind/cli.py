"""N-MIND 2.0 命令行：本机 CPU 上的单一入口。

约定（与 pyproject 的 [project.scripts] 一致）：
    python -m nmind.cli <子命令> ...      # 免安装直接跑
    nmind <子命令> ...                    # pip install -e . 之后

全局开关：
    --config NAME      读取 configs/NAME.yaml（缺省用内置默认，零配置可跑）
    --offline          硬断网：即便配置里有云端密钥也不发起任何请求
    --device auto|cpu|cuda
    --seed N           覆盖随机种子（同 seed 同权重，用于复现验收）
    --json             机器可读输出（写 results/*.json 并用 stdout 打印路径）
    --out PATH         覆盖输出文件位置

诚实性说明：这里打印的每个数字都来自本次实测；文档里的结论必须能在 results/ 找到对应 JSON。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import Config, load_config
from .core.architecture import NModel
from .learning import datasets as D
from .learning.checkpoint import load_checkpoint, save_checkpoint
from .learning.trainer import evaluate, make_data, train, train_continual
from .runtime.hardware import detect


# ---------------------------------------------------------------------- 工具

def _cfg(args: argparse.Namespace) -> Config:
    cfg = load_config(args.config)
    if args.offline:
        cfg = cfg.model_copy(update={"offline_forced": True})
    if args.device:
        cfg = cfg.model_copy(update={"runtime": cfg.runtime.model_copy(update={"device": args.device})})
    if args.seed is not None:
        cfg = cfg.model_copy(update={"runtime": cfg.runtime.model_copy(update={"seed": int(args.seed)})})
    if getattr(args, "traj_len", None):
        cfg = cfg.model_copy(update={"learning": cfg.learning.model_copy(
            update={"traj_len": int(args.traj_len)})})
    return cfg


def _emit(payload: dict[str, Any], args: argparse.Namespace, *, default_name: str,
          human: str | None = None) -> int:
    """人类可读输出走 stdout；--json 时只输出文件路径，便于脚本消费。"""
    if args.json:
        out = Path(args.out) if args.out else _cfg_root(args) / "results" / default_name
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                       encoding="utf-8")
        print(out)
        return 0
    print(human if human is not None else json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return 0


def _cfg_root(args: argparse.Namespace) -> Path:
    from .config import REPO_ROOT

    return REPO_ROOT


def _kv(d: dict[str, Any], keys: tuple[str, ...]) -> str:
    return "  ".join(f"{k}={d[k]}" for k in keys if k in d)


# ---------------------------------------------------------------------- doctor

def cmd_doctor(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    device, report = detect(cfg.runtime.device, torch_threads=cfg.runtime.torch_threads)
    model = NModel(cfg.model, n_actions=4, mem_dim=16)
    params = model.n_params()
    payload = {"hardware": report.as_dict(), "params": params,
               "cloud_mode": cfg.effective_cloud_mode(),
               "cloud_key_present": cfg.cloud_key_present(),
               "language_source": cfg.language.source,
               "offline_forced": cfg.offline_forced}
    human = (f"平台      : {report.platform} ({report.machine})\n"
             f"Python    : {report.python}   torch: {report.torch}\n"
             f"设备      : {device}   线程: {report.torch_threads}\n"
             f"CPU       : 逻辑 {report.cpu_count_logical} 核 / 物理 {report.cpu_count_physical}\n"
             f"指令集    : {_kv(report.cpu_features, ('avx2', 'avx512', 'amx_tile', 'vnni'))}\n"
             f"内存      : 总 {report.ram_total_mb} MB / 可用 {report.ram_available_mb} MB\n"
             f"参数量    : {params['total']:,}（blocks {params['blocks']:,}）\n"
             f"云端模式  : {cfg.effective_cloud_mode()}"
             f"{'（--offline 已强制关闭）' if cfg.offline_forced else ''}\n"
             f"文本来源  : {cfg.language.source}（auto_download={cfg.language.auto_download}）\n"
             + "\n".join(f"备注      : {n}" for n in report.notes))
    return _emit(payload, args, default_name="doctor.json", human=human)


# ---------------------------------------------------------------------- demo (M1)

def cmd_demo(args: argparse.Namespace) -> int:
    """CPU 训练 demo：训练 → 保存 → 重新加载 → 预测，四步一条命令走完。"""
    cfg = _cfg(args)
    epochs = int(args.epochs or cfg.learning.epochs)
    t0 = time.perf_counter()
    res = train(cfg, epochs=epochs, seed=args.seed, device=cfg.runtime.device,
                checkpoint_path=None, quiet=args.quiet)
    ckpt = _cfg_root(args) / "results" / "demo.ckpt.pt"
    save_checkpoint(res.model, cfg=cfg, path=ckpt, probe=res.probe,
                    meta={"seed": res.final["seed"], "dataset_hash": res.final["dataset_hash"],
                          "task": res.final["task"], "eval": res.final["eval"]})
    reloaded, meta = load_checkpoint(ckpt, cfg=cfg, device="cpu")
    row = make_data(cfg, D.GridTask(goal_visible_steps=4, seed=res.final["seed"]),
                    seed=res.final["seed"])["test"][0]
    a_orig, b_orig = _predict(res.model, row)
    a_re, b_re = _predict(reloaded, row)
    ident = bool((a_orig == a_re).all() and np.allclose(b_orig, b_re, atol=0, rtol=0))
    ev = res.final["eval"]
    payload = {"wall_s": round(time.perf_counter() - t0, 2), "epochs": epochs,
               "params": res.final["params"]["total"], "seed": res.final["seed"],
               "loss_first": round(res.curve["total"][0], 4),
               "loss_last": round(res.curve["total"][-1], 4),
               "eval_occluded": ev, "eval_all": res.final["eval_all"],
               "route_load": res.final["route"], "dataset_hash": res.final["dataset_hash"],
               "config_hash": res.final["config_hash"], "git_revision": res.final["git_revision"],
               "checkpoint": str(ckpt),
               "save_load_bitwise_identical": ident,
               "predicted_actions_first_row": a_orig.tolist(),
               "oracle_actions_first_row": [int(x) for x in row["actions"][:8]]}
    human = (f"[demo] {epochs} epoch 用时 {payload['wall_s']}s（CPU）  参数 {payload['params']:,}"
             f"  seed={payload['seed']}\n"
             f"  损失 首 {payload['loss_first']:.4f} → 末 {payload['loss_last']:.4f}\n"
             f"  遮挡段准确率 {ev.get('accuracy_occluded', float('nan')):.3f}"
             f"（随机猜 0.250）  记忆探针 {ev['probe_accuracy']:.3f}（随机猜 0.028）\n"
             f"  全段准确率   {res.final['eval_all']['accuracy']:.3f}\n"
             f"  块负载       {_kv(res.final['route'], ('top1_block_share', 'dead_blocks'))}\n"
             f"  保存/加载    {ckpt}  逐位一致={ident}\n"
             f"  第一行预测   {payload['predicted_actions_first_row'][:8]}\n"
             f"  oracle 标签  {payload['oracle_actions_first_row']}\n"
             f"  数据哈希     {payload['dataset_hash']}  配置哈希 {payload['config_hash']}\n"
             f"  git          {payload['git_revision']}\n"
             f"说明：准确率是网络在遮挡期选对动作的比例，不代表语言理解；"
             f"SQLite 命中不计入这里。")
    return _emit(payload, args, default_name="demo.json", human=human)


def _predict(model: NModel, row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        out = model.forward_sequence(torch.tensor(row["obs"][None]),
                                     torch.tensor(row["actions"][None]))
        probs = out["action_probs"][0].cpu().numpy()
    return probs.argmax(-1), probs.max(-1)


# ---------------------------------------------------------------------- train

def cmd_train(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    kwargs: dict[str, Any] = {"epochs": args.epochs, "seed": args.seed,
                              "device": cfg.runtime.device, "quiet": args.quiet}
    if args.ablation:
        kwargs.update(_ablation_kwargs(args.ablation))
    if args.continual:
        res = train_continual(cfg, seed=args.seed, device=cfg.runtime.device,
                              use_replay=not args.no_replay, quiet=args.quiet)
        payload = {k: v for k, v in res.items() if k not in ("model", "probe", "data_A", "data_B")}
        ckpt = args.ckpt or str(_cfg_root(args) / "results" / "continual.ckpt.pt")
        save_checkpoint(res["model"], cfg=cfg, path=ckpt, probe=res["probe"],
                        meta={"kind": "continual", "seed": args.seed or cfg.runtime.seed})
        payload["checkpoint"] = ckpt
        human = (f"[continual] A(遮挡) {payload['A_before'].get('accuracy_occluded'):.3f} → "
                 f"{payload['A_after'].get('accuracy_occluded'):.3f}"
                 f"   遗忘 {payload['forgetting_occluded_accuracy']:+.3f}"
                 f"   (回放={'开' if not args.no_replay else '关'})\n"
                 f"  B(遮挡) {payload['B_after'].get('accuracy_occluded'):.3f}\n"
                 f"  探针遗忘 {payload['forgetting_probe_accuracy']:+.3f}\n"
                 f"  缓冲区 {payload['buffer_size']} 条  指纹 {payload['replay_fingerprint']}\n"
                 f"  checkpoint → {ckpt}\n"
                 f"注意：这里只说明'同 seed、只差回放'下的遗忘量，不推断到真实语言任务。")
        return _emit(payload, args, default_name="continual.json", human=human)
    res = train(cfg, **kwargs)
    ckpt = args.ckpt or str(_cfg_root(args) / "results" / "train.ckpt.pt")
    res.checkpoint = save_checkpoint(res.model, cfg=cfg, path=ckpt, probe=res.probe,
                                     meta={"seed": res.final["seed"],
                                           "dataset_hash": res.final["dataset_hash"],
                                           "task": res.final["task"], "eval": res.final["eval"],
                                           "curve_tail": res.curve["total"][-5:]})
    payload = {"summary": res.summary(), "final": res.final,
               "curve": {k: [round(x, 4) for x in v] for k, v in res.curve.items()},
               "evals": res.evals, "checkpoint": res.checkpoint}
    ev = res.final["eval"]
    human = (f"[train] epochs={args.epochs or cfg.learning.epochs} 用时 {res.wall_s:.1f}s "
             f"步数 {res.steps}  跳过非有限损失 {res.final['non_finite_steps_skipped']} 次\n"
             f"  遮挡准确率 {ev.get('accuracy_occluded', float('nan')):.3f} "
             f"探针 {ev['probe_accuracy']:.3f} 参数 {res.final['params']['total']:,}\n"
             f"  checkpoint → {res.checkpoint}")
    return _emit(payload, args, default_name="train.json", human=human)


_ABLATION_FLAGS = {
    "no_router": {"execution": "dense", "use_router": False},
    "no_error_gating": {"use_error_gating": False},
    "no_workspace": {"use_workspace": False},
}


def _ablation_kwargs(name: str) -> dict[str, Any]:
    if name not in _ABLATION_FLAGS:
        raise SystemExit(f"未知消融项 {name!r}；可选：{', '.join(_ABLATION_FLAGS)}"
                         "（记忆/回放/云端消融见 ablate 子命令）")
    return dict(_ABLATION_FLAGS[name])


# ---------------------------------------------------------------------- predict / run

def cmd_predict(args: argparse.Namespace) -> int:
    """加载 checkpoint，对给定观测（文件或随机轨迹）输出动作、置信度、预测误差与校验依据。"""
    cfg = _cfg(args)
    ckpt = args.ckpt or str(_cfg_root(args) / "results" / "demo.ckpt.pt")
    model, meta = load_checkpoint(ckpt, cfg=cfg, device=cfg.runtime.device,
                                  allow_config_mismatch=args.allow_config_mismatch)
    if args.observe:
        rows = [_row_from_text(line) for line in Path(args.observe).read_text(encoding="utf-8").splitlines()
                if line.strip()]
    else:
        task = D.GridTask(goal_visible_steps=args.visible_steps, seed=args.seed or cfg.runtime.seed)
        rows = make_data(cfg, task, seed=args.seed or cfg.runtime.seed)["test"][:args.n]
    from .runtime.engine import NMindEngine

    eng = NMindEngine(cfg, model=model, use_memory=not args.no_memory, seed=args.seed,
                      session_id="predict")
    steps: list[dict[str, Any]] = []
    for r in rows:
        prev = None
        for t in range(min(len(r["obs"]), args.max_steps)):
            oc = eng.step(r["obs"][t], action=prev, record=False)
            prev = oc.action
            steps.append({"row": r.get("row", 0), "t": t, **oc.as_dict()})
    acc = float(np.mean([int(s["validation"]["ok"]) for s in steps])) if steps else float("nan")
    payload = {"checkpoint": ckpt, "meta": meta, "n_steps": len(steps),
               "validation_ok_rate": round(acc, 4),
               "cloud": eng.stats()["cloud"], "memory": eng.stats()["memory"],
               "steps": steps[:40], "note": "validation_ok_rate 是规则校验通过率，不是准确率；"
                                            "真实准确率需要 oracle 标签，见 evaluate 子命令。"}
    human = (f"[predict] {ckpt}\n"
             f"  保存于 {meta.get('saved_at')}  config_hash={meta.get('config_hash')} "
             f"git={meta.get('git_revision')}\n"
             f"  前 {len(steps)} 步，规则校验通过率 {payload['validation_ok_rate']:.3f}\n"
             f"  云端：{_kv(payload['cloud'], ('mode', 'requested', 'allowed'))}\n"
             + "\n".join(f"  t{s['t']:02d} {s['text']}" for s in steps[:12])
             + ("\n  …（--json 输出全部步）" if len(steps) > 12 else ""))
    eng.close()
    return _emit(payload, args, default_name="predict.json", human=human)


def _row_from_text(line: str) -> dict[str, Any]:
    """从 CSV 行（26 个浮点，逗号分隔）读一条观测序列的单步；纯本地解析，不信任内容。"""
    vals = np.array([float(x) for x in line.replace(";", ",").split(",")], dtype=np.float32)
    if vals.shape[0] != D.OBS_DIM:
        raise SystemExit(f"观测长度应为 {D.OBS_DIM}，实际 {vals.shape[0]}：{line[:40]}…")
    return {"obs": vals[None], "actions": np.zeros(1, dtype=np.int64), "len": 1, "goal_code": 0}


def cmd_run(args: argparse.Namespace) -> int:
    """交互 rollout：模型选动作、环境给下一观测，逐步打印（M1 验收用的可看闭环）。"""
    cfg = _cfg(args)
    from .runtime.engine import NMindEngine

    ckpt = args.ckpt
    model = None
    if ckpt:
        model, _meta = load_checkpoint(ckpt, cfg=cfg, device=cfg.runtime.device,
                                       allow_config_mismatch=True)
    eng = NMindEngine(cfg, model=model, use_memory=not args.no_memory,
                      seed=args.seed or cfg.runtime.seed, session_id="run",
                      verbose=not args.quiet)
    seq = eng.rollout(T=args.T, seed=args.seed or cfg.runtime.seed)
    payload = {"goal": list(seq.goal), "start": list(seq.start),
               "actions": seq.actions.tolist(),
               "steps": [o.as_dict() for o in seq.outcomes],
               "stats": eng.stats()}
    human = (f"[run] T={args.T} 目标格 {seq.goal}（模型自己看不见，只能靠状态/记忆）\n"
             + "\n".join(f"  {o.text}" for o in seq.outcomes[:10])
             + f"\n  统计：{_kv(payload['stats']['governor'], ('mean_k', 'degraded_steps', 'timeout_steps'))}\n"
             f"  云端：{payload['stats']['cloud']['mode']}，调用 {payload['stats']['cloud']['calls']} 次\n"
             f"  说明：这是无监督标签的自由运行，只验证闭环能跑，不据此声称能力。")
    eng.close()
    return _emit(payload, args, default_name="run.json", human=human)


# ---------------------------------------------------------------------- evaluate

def cmd_evaluate(args: argparse.Namespace) -> int:
    """在 checkpoint 上跑完整任务套件：状态预测、决策指标、读出探针、记忆跨度、预算-难度。"""
    cfg = _cfg(args)
    ckpt = args.ckpt or str(_cfg_root(args) / "results" / "demo.ckpt.pt")
    model, meta = load_checkpoint(ckpt, cfg=cfg, device=cfg.runtime.device,
                                  allow_config_mismatch=True)
    dev = model.block_load_count.device
    from .evaluation.tasks import TaskSuite
    from .learning.checkpoint import load_probe
    from .learning.trainer import _make_probe, evaluate

    probe = _make_probe(cfg.model.block_hidden, 36).to(dev)
    # 探针权重存在 checkpoint 的 "probe" 段里；没有就保持随机初始化并明确说明。
    import torch as _t

    blob = _t.load(Path(ckpt), map_location=dev, weights_only=True)
    has_probe = load_probe(probe, blob.get("probe"), device=dev)
    task_seed = int(args.seed or cfg.runtime.seed)
    data = make_data(cfg, D.GridTask(goal_visible_steps=args.visible_steps, seed=task_seed),
                     seed=task_seed)
    payload = TaskSuite().run(model, probe, cfg=cfg, device=dev, task_seed=task_seed, T=args.T)
    payload["_meta"] = {"checkpoint": ckpt, "config_hash": meta.get("config_hash"),
                        "git_revision": meta.get("git_revision"),
                        "probe_restored_from_checkpoint": has_probe,
                        "dataset_hash": data["dataset_hash"],
                        "task": {"goal_visible_steps": args.visible_steps, "seed": task_seed}}
    human = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    return _emit(payload, args, default_name="evaluate.json", human=human)


# ---------------------------------------------------------------------- memory

def cmd_memory(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    from .memory.sqlite_store import MemoryStore

    db = cfg.resolve(cfg.memory.db_path)
    store = MemoryStore(db)
    try:
        if args.action == "stats":
            payload = {"db": str(db), **store.stats(), "capabilities": store.capabilities}
            human = f"[memory] {db}\n  " + json.dumps(payload, ensure_ascii=False, default=str)
        elif args.action == "search":
            from .memory.retriever import MemoryRetriever

            retr = MemoryRetriever(store, top_k=cfg.memory.retrieval_top_k)
            hits = retr.search(args.query or "", session_id=args.session)
            payload = {"query": args.query, "mode": retr.mode,
                       "vector": retr.vector_context(args.query or "", dim=16).tolist(),
                       "hits": [h.__dict__ for h in hits]}
            human = (f"[memory search '{args.query}'] 检索模式={retr.mode} 命中={len(hits)}\n"
                     + "\n".join(f"  {i+1:2d}. score={h.score:.3f} conf={h.confidence:.2f} "
                                 f"[{h.kind}] {h.text[:60]}" for i, h in enumerate(hits)))
        elif args.action == "recent":
            eps = store.recent_episodes(limit=args.limit, session_id=args.session)
            payload = {"episodes": [e.__dict__ for e in eps]}
            human = "\n".join(f"  #{e.id} [{e.task}] {e.observation[:48]} → {e.action} ({e.outcome})"
                             for e in eps) or "  （空库）"
        elif args.action == "facts":
            fs = store.active_facts(limit=args.limit)
            payload = {"facts": [f.__dict__ for f in fs]}
            human = "\n".join(f"  #{f.id} {f.subject} | {f.predicate} | {f.object} "
                             f"[{f.status}] conf={f.confidence:.2f}" for f in fs) or "  （无事实）"
        elif args.action == "consolidate":
            from .memory.consolidation import Consolidator

            payload = Consolidator(store).run(min_evidence=args.min_evidence,
                                              min_feedback=args.min_feedback)
            human = f"[consolidate] {json.dumps(payload, ensure_ascii=False, default=str)}\n" \
                    f"  说明：巩固只写事实表，不改动任何神经参数。"
        elif args.action == "delete":
            ok = store.delete_episode(int(args.id), reason=args.reason or "cli")
            payload = {"deleted": ok, "id": args.id}
            human = f"[memory delete] episode #{args.id} → {'已删除并留墓碑' if ok else '不存在'}"
        elif args.action == "export":
            path = args.out or str(_cfg_root(args) / "results" / "memory_export.json")
            Path(path).write_text(store.export_json(), encoding="utf-8")
            payload = {"export": path}
            human = f"[memory export] → {path}"
        else:
            raise SystemExit(f"未知 memory 动作 {args.action}")
    finally:
        store.close()
    return _emit(payload, args, default_name=f"memory_{args.action}.json", human=human)


# ---------------------------------------------------------------------- benchmark / baseline / ablate

def cmd_benchmark(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    from .evaluation.benchmarks import cpu_benchmark

    payload = cpu_benchmark(cfg, seed=args.seed, seq_len=args.seq_len, repeats=args.repeats,
                            device=cfg.runtime.device, quiet=args.quiet,
                            n_blocks_list=tuple(int(x) for x in args.n_blocks.split(",")))
    pe = payload.get("per_event", {})
    comp = payload.get("comparison", {})
    human = (f"[benchmark] {payload['hardware']['platform']} torch {payload['hardware']['torch']} "
             f"线程 {payload['threads']}  指令集 avx2={payload['hardware']['cpu_features']['avx2']} "
             f"avx512={payload['hardware']['cpu_features']['avx512']}\n"
             + "\n".join(f"  {k:22s} 序列 {v['mean_ms']:8.2f}ms  P95 {v['p95_ms']:8.2f}ms  "
                         f"单事件 {v['per_event_ms']:6.3f}ms  RSS {v.get('rss_peak_mb')}MB"
                         for k, v in pe.items())
             + f"\n  稀疏 vs 稠密平均：{comp.get('vs_dense_mean', 'n/a')}  "
               f"块计算量比 {comp.get('blocks_touched_ratio', 'n/a')}\n"
             f"  能耗：{payload.get('energy_note', '未测量')}\n"
             f"  提醒：激活比例低 ≠ 延迟等比下降，以上数字只在本机这代 CPU 上成立。")
    return _emit(payload, args, default_name="benchmark_cpu.json", human=human)


def cmd_baseline(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    from .evaluation.baselines import run_baselines

    payload = run_baselines(cfg, seed=args.seed, epochs=args.epochs, device=cfg.runtime.device,
                            quiet=args.quiet)
    human = ("[baseline] 相同数据/预算下的主干对比（遮挡段准确率）\n"
             + "\n".join(f"  {r['name']:14s} 参数 {r['params']:>9,}  acc={r['accuracy']:.3f} "
                         f"occ={r['accuracy_occluded']:.3f}  seq={r['seq_ms']:.2f}ms"
                         for r in payload["rows"])
             + "\n  共享主干（Kai-0.6B 那种每任务独占干 vs 共享前缀）：\n"
             + "\n".join(f"    {s['variant']:13s} 共享 {s['shared_blocks']} 私有 {s['private_blocks']} "
                         f"参数 {s['params']:,} acc={s['accuracy']:.3f}" for s in payload["shared_trunk"])
             + f"\n  {payload['note']}")
    return _emit(payload, args, default_name="baselines.json", human=human)


def cmd_ablate(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    from .evaluation.ablations import run_ablations, run_cloud_ablation, run_memory_ablation

    seeds = tuple(int(x) for x in args.seeds.split(","))
    if args.only == "memory":
        payload = run_memory_ablation(cfg, seed=seeds[0], device=cfg.runtime.device, quiet=args.quiet)
    elif args.only == "cloud":
        payload = run_cloud_ablation(cfg, seed=seeds[0], device=cfg.runtime.device, quiet=args.quiet)
    else:
        payload = run_ablations(cfg, seeds=seeds, device=cfg.runtime.device, quiet=args.quiet,
                                include_memory=(args.only != "structural"),
                                out_dir=args.out_dir)
        return _emit(payload if args.json else {"see_files": payload.get("files", {})},
                     args, default_name="ablation.json",
                     human=f"[ablate] 报告已写入 {json.dumps(payload.get('files', {}), ensure_ascii=False)}"
                           f"\n  总用时 {payload['wall_s']}s，seeds={seeds}")
    return _emit(payload, args, default_name=f"ablation_{args.only}.json",
                 human=json.dumps(payload, ensure_ascii=False, indent=2, default=str))


# ---------------------------------------------------------------------- cloud

def cmd_cloud(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    from .bridge.policy import CloudPolicy
    from .bridge.privacy import build_payload

    pol = CloudPolicy(cfg)
    payload = build_payload(category=args.category, question=args.text or "示例问题",
                            context=[f"记忆条目{i}" for i in range(args.n_context)],
                            numeric={"x": 1}, privacy_levels={"normal": args.n_context},
                            max_chars=cfg.cloud.max_payload_chars)
    decision = pol.evaluate(payload)
    out = {"mode": cfg.effective_cloud_mode(), "offline_forced": cfg.offline_forced,
           "payload_blocked": list(payload.blocked), "payload_chars": payload.payload_chars,
           "original_chars": payload.original_chars,
           "decision": decision.__dict__, "policy": pol.status(),
           "key_env": cfg.cloud.api_key_env, "key_present": cfg.cloud_key_present(),
           "note": "本命令不会发起网络请求；要看真实连通性请显式设置密钥并自行执行 check_cloud 脚本。"}
    human = (f"[cloud] mode={out['mode']}  offline={cfg.offline_forced}  "
             f"密钥变量={cfg.cloud.api_key_env}（存在={out['key_present']}，值不读取不打印）\n"
             f"  脱敏后 {payload.payload_chars}/{payload.original_chars} 字符，"
             f"拦截字段={out['payload_blocked'] or '无'}\n"
             f"  策略判定：allowed={decision.allowed} 理由={decision.reason}\n"
             f"  门顺序：模式→敏感→白名单→预算→批准；任一不过即拒绝，不发出请求。")
    return _emit(out, args, default_name="cloud_policy.json", human=human)


# ---------------------------------------------------------------------- 解析器

def _add_common(target: argparse.ArgumentParser, *, suppress: bool) -> None:
    """全局开关：既允许写在子命令前，也允许写在子命令后（用户怎么顺手怎么来）。

    子命令那份用 SUPPRESS 做默认值，否则"子命令里没写"会被解释成
    "用默认值覆盖掉外层已经写过的值"。
    """
    d = argparse.SUPPRESS if suppress else None
    false_d = argparse.SUPPRESS if suppress else False
    target.add_argument("--config", default=d, help="configs/ 下的文件名或任意 YAML 路径")
    target.add_argument("--offline", action="store_true", default=false_d,
                        help="硬断网：云端一律关闭")
    target.add_argument("--device", default=d, choices=["auto", "cpu", "cuda"])
    target.add_argument("--seed", type=int, default=d)
    target.add_argument("--json", action="store_true", default=false_d,
                        help="输出机器可读 JSON 文件路径")
    target.add_argument("--out", default=d, help="--json 时的输出文件")
    target.add_argument("--quiet", action="store_true", default=false_d)
    target.add_argument("--traceback", action="store_true", default=false_d,
                        help="出错时打印完整栈，便于报 bug")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="nmind",
                                description="N-MIND 2.0：本机 CPU 可跑的本地优先研究模型")
    _add_common(p, suppress=False)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("doctor", help="硬件/依赖/配置体检")
    sp.set_defaults(fn=cmd_doctor)

    sp = sub.add_parser("demo", help="M1：CPU 训练→保存→加载→预测")
    sp.add_argument("--epochs", type=int, default=None)
    sp.set_defaults(fn=cmd_demo)

    sp = sub.add_parser("train", help="训练并保存 checkpoint")
    sp.add_argument("--epochs", type=int, default=None)
    sp.add_argument("--continual", action="store_true", help="A→B→重测 A")
    sp.add_argument("--no-replay", action="store_true", help="持续学习不回放（消融）")
    sp.add_argument("--ablation", default=None, help="no_router / no_error_gating / no_workspace")
    sp.add_argument("--ckpt", default=None)
    sp.add_argument("--traj-len", type=int, default=None)
    sp.set_defaults(fn=cmd_train)

    sp = sub.add_parser("predict", help="加载权重做预测（观测来自 --observe 或随机轨迹）")
    sp.add_argument("--ckpt", default=None)
    sp.add_argument("--observe", default=None, help="每行 26 个浮点数的 CSV 文件")
    sp.add_argument("--n", type=int, default=2)
    sp.add_argument("--max-steps", type=int, default=24)
    sp.add_argument("--visible-steps", type=int, default=4)
    sp.add_argument("--no-memory", action="store_true")
    sp.add_argument("--allow-config-mismatch", action="store_true")
    sp.set_defaults(fn=cmd_predict)

    sp = sub.add_parser("run", help="自由运行 rollout（看闭环）")
    sp.add_argument("--T", type=int, default=12)
    sp.add_argument("--ckpt", default=None)
    sp.add_argument("--no-memory", action="store_true")
    sp.set_defaults(fn=cmd_run)

    sp = sub.add_parser("evaluate", help="任务套件：状态预测/决策/读出探针/记忆跨度/预算-难度")
    sp.add_argument("--ckpt", default=None)
    sp.add_argument("--visible-steps", type=int, default=4)
    sp.add_argument("--T", type=int, default=16, help="预算-难度实验的序列长度")
    sp.set_defaults(fn=cmd_evaluate)

    sp = sub.add_parser("memory", help="记忆库：stats/search/recent/facts/consolidate/delete/export")
    sp.add_argument("action", choices=["stats", "search", "recent", "facts", "consolidate",
                                       "delete", "export"])
    sp.add_argument("--query", default="")
    sp.add_argument("--session", default=None)
    sp.add_argument("--limit", type=int, default=20)
    sp.add_argument("--id", type=int, default=-1)
    sp.add_argument("--reason", default="cli")
    sp.add_argument("--min-evidence", type=int, default=2)
    sp.add_argument("--min-feedback", type=float, default=0.0)
    sp.set_defaults(fn=cmd_memory)

    sp = sub.add_parser("benchmark", help="CPU 基准：延迟/吞吐/RSS/子模块耗时")
    sp.add_argument("--seq-len", type=int, default=48)
    sp.add_argument("--repeats", type=int, default=60)
    sp.add_argument("--n-blocks", default="8")
    sp.set_defaults(fn=cmd_benchmark)

    sp = sub.add_parser("baseline", help="基线对比：稠密 N-Cell / GRU / 共享主干")
    sp.add_argument("--epochs", type=int, default=8)
    sp.set_defaults(fn=cmd_baseline)

    sp = sub.add_parser("ablate", help="消融：结构 / 记忆 / 云端，写 JSON+CSV")
    sp.add_argument("--seeds", default="11,12,13")
    sp.add_argument("--only", default="all", choices=["all", "structural", "memory", "cloud"])
    sp.add_argument("--out-dir", default="results")
    sp.set_defaults(fn=cmd_ablate)

    sp = sub.add_parser("cloud", help="云端策略自检（不发请求）")
    sp.add_argument("--category", default="public_web_query")
    sp.add_argument("--text", default="这个网格任务该怎么走？")
    sp.add_argument("--n-context", type=int, default=3)
    sp.set_defaults(fn=cmd_cloud)

    for child in sub.choices.values():
        _add_common(child, suppress=True)      # 允许 nmind demo --epochs 8 --seed 11 这种写法
    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    try:
        return int(args.fn(args))
    except KeyboardInterrupt:
        print("已中断（未写入不完整产物）。", file=sys.stderr)
        return 130
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 - 默认把异常说成人话；--traceback 时保留完整栈
        if args.traceback:
            raise
        print(f"失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        print("（加 --traceback 看完整栈；错误不会被静默当成成功。）", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
