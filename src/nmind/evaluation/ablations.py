"""消融实验：一次只关掉一个部件，其余代码路径完全相同。

消融开关全部内建在 NModel / NMindEngine 的构造参数与 learning.trainer 的函数签名里，
不靠外部脚本 monkey-patch —— 否则"基线"和"消融"跑的就不是同一份代码，结论作废。

指令第 8 节要求的五项，对应关系：
  no_error_gating      路由看不到真实预测误差（恒 0.5）
  no_router            不做稀疏选择，全部块激活（execution=dense）
  no_memory            记忆通道置零：训练与评测都不给检索向量（run_memory_ablation）
  no_replay            持续学习时不重放旧任务（run_continual_ablation）
  no_cloud             云端模式 off 与 ask/auto_allowed 对照（run_cloud_ablation）

每项都跑多个 seed，报均值与极差；所有数字都是本机实测，不外推、不编造。
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from ..config import Config, default_config
from ..core.architecture import NModel
from ..language.interface import describe_event
from ..learning import datasets as D
from ..learning.trainer import evaluate, make_data, train, train_continual
from ..memory.retriever import MemoryRetriever
from ..memory.sqlite_store import MemoryStore
from ..runtime.engine import _claims_from

REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class Variant:
    name: str
    execution: str = "sparse"
    use_router: bool = True
    use_error_gating: bool = True
    use_workspace: bool = True
    top_k: int | None = None
    trainable_adapter: bool = False
    continual: bool = False
    use_replay: bool = True
    note: str = ""

    def key(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("note")
        return d


DEFAULT_VARIANTS = (
    Variant("baseline", note="完整配置：稀疏路由 + 误差门控 + 共享工作空间"),
    Variant("no_error_gating", use_error_gating=False, note="门控看不到真实误差，恒 0.5"),
    Variant("no_router", execution="dense", use_router=False, note="全块激活（无稀疏选择）"),
    Variant("no_workspace", use_workspace=False, note="共享工作空间读与写都关掉"),
    Variant("sparse_k1", top_k=1, note="每步只激活 1 块"),
    Variant("sparse_k4", top_k=4, note="每步激活 4 块"),
    Variant("adapter_only", trainable_adapter=True,
            note="局部更新：只放开决策头 / 动作嵌入 / 块→工作空间投影"),
    Variant("continual_with_replay", continual=True, use_replay=True,
            note="A→B→重测 A，旧任务经验回放"),
    Variant("continual_no_replay", continual=True, use_replay=False,
            note="A→B→重测 A，不回放（= no_replay 消融）"),
)


# ---------------------------------------------------------------------- 计时

def _timing(model: NModel, rows: list[dict[str, Any]], device: torch.device, *,
            k: int | None, reps: int = 30) -> dict[str, float]:
    """序列级墙钟：延迟按整条序列计（含路由、工作空间与预测器开销）。"""
    model.eval()
    peak_mb = 0.0
    try:
        import psutil

        proc = psutil.Process()
        peak_mb = proc.memory_info().rss / 2**20
    except Exception:               # noqa: BLE001 - psutil 是软依赖，缺了只是少一个 RAM 数字
        proc = None
    samples: list[float] = []
    touched = 0.0
    n_blocks = model.cfg.n_blocks
    with torch.no_grad():
        for i in range(reps + 5):
            r = rows[i % len(rows)]
            obs = torch.tensor(r["obs"][None], device=device)
            acts = torch.tensor(r["actions"][None], device=device)
            t0 = time.perf_counter()
            out = model.forward_sequence(obs, acts, k=k)
            ms = (time.perf_counter() - t0) * 1000.0
            touched = float(out["blocks_touched"])
            if i >= 5:
                samples.append(ms)
                if proc is not None:
                    peak_mb = max(peak_mb, proc.memory_info().rss / 2**20)
    ss = sorted(samples)
    T = float(len(rows[0]["obs"]))
    return {"mean_ms": round(statistics.fmean(ss), 4),
            "p50_ms": round(ss[len(ss) // 2], 4),
            "p95_ms": round(ss[min(len(ss) - 1, int(0.95 * (len(ss) - 1)))], 4),
            "max_ms": round(ss[-1], 4),
            "per_event_ms": round(statistics.fmean(ss) / T, 4),
            "peak_rss_mb": round(peak_mb, 1),
            "blocks_touched": touched,
            "activation_ratio": round(touched / (T * n_blocks), 4),
            "T": T}


# ---------------------------------------------------------------------- 结构类消融

def run_structural_ablations(cfg: Config | None = None, *,
                             seeds: tuple[int, ...] = (11, 12, 13),
                             variants: tuple[Variant, ...] = DEFAULT_VARIANTS,
                             epochs: int | None = None, device: str = "auto",
                             quiet: bool = False) -> dict[str, Any]:
    cfg = cfg or default_config()
    from ..runtime.hardware import detect

    dev_obj, report = detect(device, torch_threads=cfg.runtime.torch_threads)
    dev = dev_obj if isinstance(dev_obj, torch.device) else torch.device(str(dev_obj))
    task = D.GridTask(goal_visible_steps=4, seed=seeds[0])
    data = make_data(cfg, task, seed=seeds[0])
    out: dict[str, Any] = {
        "kind": "structural",
        "hardware": report.as_dict(),
        "seeds": list(seeds),
        "epochs": int(epochs or cfg.learning.epochs),
        "task": D.task_to_dict(task),
        "dataset_hash": data["dataset_hash"],
        "config_hash": _cfg_hash(cfg),
        # 随机猜的基准线：动作 4 类、目标格 36 类。低于这条线不能说"学到了"。
        "chance_level": {"action": 0.25, "goal_probe": round(1.0 / 36, 4)},
        "variants": [],
    }
    for v in variants:
        t0 = time.perf_counter()
        per_seed = []
        for seed in seeds:
            per_seed.append(_run_one_variant(v, cfg, seed=seed, epochs=epochs,
                                            task=task, data=data, dev=dev))
        row = _aggregate(v, per_seed, time.perf_counter() - t0)
        out["variants"].append(row)
        if not quiet:
            print(f"  {v.name:22s} occ={row['accuracy_occluded']['mean']:.3f}"
                  f"±{row['accuracy_occluded']['spread']:.3f} "
                  f"all={row['accuracy_all']['mean']:.3f} "
                  f"probe={row['probe_accuracy']['mean']:.3f} "
                  f"p50={row['p50_ms']['mean']:.2f}ms "
                  f"act={row['activation_ratio']['mean']:.3f} "
                  f"dead={row['dead_blocks']['mean']:.1f}")
    return out


def _run_one_variant(v: Variant, cfg: Config, *, seed: int, task: D.GridTask,
                     data: dict[str, Any], dev: torch.device) -> dict[str, Any]:
    k = v.top_k or (cfg.model.n_blocks if v.execution == "dense" else cfg.model.top_k)
    if v.continual:
        res = train_continual(cfg, seed=seed, use_replay=v.use_replay,
                              trainable_adapter=v.trainable_adapter,
                              device=str(dev), quiet=True)
        model, probe = res["model"], res["probe"]
        cont = {key: res[key] for key in
                ("A_before", "A_after", "B_after", "forgetting_occluded_accuracy",
                 "forgetting_probe_accuracy", "use_replay", "buffer_size")}
    else:
        res = train(cfg, seed=seed, epochs=None, task=task, device=str(dev), quiet=True,
                    trainable_adapter=v.trainable_adapter, execution=v.execution,
                    use_router=v.use_router, use_error_gating=v.use_error_gating,
                    use_workspace=v.use_workspace)
        model, probe = res.model, res.probe
        cont = {}
    occ = evaluate(model, probe, data["test"], device=dev, occluded_only=True)
    full = evaluate(model, probe, data["test"], device=dev)
    timing = _timing(model, data["test"], dev, k=k)
    return {"seed": seed, "occ": occ, "full": full, "timing": timing,
            "route": model.active_block_stats(), "train_wall_s": round(res.wall_s, 2),
            "continual": cont}


def _aggregate(v: Variant, per_seed: list[dict[str, Any]], wall: float) -> dict[str, Any]:
    metrics = {
        "accuracy_occluded": lambda r: r["occ"].get("accuracy_occluded", float("nan")),
        "accuracy_all": lambda r: r["full"]["accuracy"],
        "probe_accuracy": lambda r: r["occ"]["probe_accuracy"],
        "mean_ms": lambda r: r["timing"]["mean_ms"],
        "p50_ms": lambda r: r["timing"]["p50_ms"],
        "p95_ms": lambda r: r["timing"]["p95_ms"],
        "per_event_ms": lambda r: r["timing"]["per_event_ms"],
        "peak_rss_mb": lambda r: r["timing"]["peak_rss_mb"],
        "activation_ratio": lambda r: r["timing"]["activation_ratio"],
        "top1_block_share": lambda r: r["route"]["top1_block_share"],
        "dead_blocks": lambda r: r["route"]["dead_blocks"],
        "train_wall_s": lambda r: r["train_wall_s"],
    }
    agg: dict[str, Any] = {}
    for name, getter in metrics.items():
        vals = [float(getter(r)) for r in per_seed]
        agg[name] = {"mean": round(statistics.fmean(vals), 4),
                     "spread": round(max(vals) - min(vals), 4),
                     "values": [round(x, 4) for x in vals]}
    row: dict[str, Any] = {"variant": v.name, "note": v.note, "config": v.key(),
                           "n_seeds": len(per_seed), **agg,
                           "load_per_block": per_seed[0]["route"]["load_per_block"]}
    if per_seed[0].get("continual"):
        row["continual"] = {k: [r["continual"].get(k) for r in per_seed]
                            for k in ("A_before", "A_after", "B_after",
                                      "forgetting_occluded_accuracy")}
        row["continual"] = {k: _mean_of_evals(vv) for k, vv in row["continual"].items()}
    return row


def _mean_of_evals(vals: list[Any]) -> Any:
    nums = [float(v.get("accuracy_occluded")) for v in vals
            if isinstance(v, dict) and "accuracy_occluded" in v]
    if nums:
        return round(statistics.fmean(nums), 4)
    plain = [float(v) for v in vals if isinstance(v, (int, float))]
    return round(statistics.fmean(plain), 4) if plain else None


# ---------------------------------------------------------------------- 记忆消融

MemoryProvider = Callable[[dict[str, Any]], np.ndarray]


def make_episode_memory_provider(*, db_path: str | Path, visible_steps: int,
                                 dim: int = 16, top_k: int = 4,
                                 session: str = "ablation") -> MemoryProvider:
    """真实 SQLite 走一遍，产出 NModel.step(mem=...) 需要的 [len, dim] 向量。

    顺序与引擎一致：某一步先用"截至上一步已写入的经历"检索，再写入本步。
    遮挡期的输入里没有目标，向量只能来自之前可见的那几步 —— 这就是记忆通道，
    不是把 oracle 目标偷偷塞进输入（oracle 标签只在打分时用）。
    """
    path = Path(db_path)
    if path.exists():
        path.unlink()
    for suffix in ("-wal", "-shm"):
        p = Path(str(path) + suffix)
        if p.exists():
            p.unlink()
    cache: dict[str, np.ndarray] = {}
    stats = {"rows": 0, "hits": 0, "steps": 0}

    def provider(row: dict[str, Any]) -> np.ndarray:
        key = row["obs"].tobytes()[:64] + str(row["goal_code"]).encode()
        if key in cache:
            return cache[key]
        store = MemoryStore(path)
        try:
            retr = MemoryRetriever(store, top_k=top_k)
            vecs = np.zeros((row["len"], dim), dtype=np.float32)
            for t in range(row["len"]):
                v = row["obs"][t]
                query = describe_event(v, size=6)
                vecs[t] = retr.vector_context(query, dim=dim, k=top_k, session_id=session)
                stats["steps"] += 1
                stats["hits"] += int(bool(np.abs(vecs[t]).sum() > 0))
                claims = _claims_from(v, int(row["actions"][t]), bool(v[24] > 0.5))
                store.add_episode(session_id=session, task="grid", observation=query,
                                  action=str(int(row["actions"][t])), outcome="recorded",
                                  feedback=0.0, source="local", confidence=0.5,
                                  observation_json={"claims": claims})
        finally:
            store.close()
        cache[key] = vecs
        stats["rows"] += 1
        return vecs

    provider.stats = stats          # type: ignore[attr-defined]
    return provider


def run_memory_ablation(cfg: Config | None = None, *, seed: int = 11,
                        device: str = "auto", epochs: int | None = None,
                        quiet: bool = False) -> dict[str, Any]:
    """no_memory 消融：同一份"带记忆训练"的权重，评测时把记忆向量置零再跑一遍。

    两个条件只差 mem 张量（检索自 SQLite vs 全零），权重、轨迹、代码路径全同。
    另附一个"从未见过记忆输入"的对照模型，用来区分"通道没训练出来"与
    "训练了但用不上"。
    """
    cfg = cfg or default_config()
    from ..runtime.hardware import detect

    dev_obj, _ = detect(device, torch_threads=cfg.runtime.torch_threads)
    dev = dev_obj if isinstance(dev_obj, torch.device) else torch.device(str(dev_obj))
    task = D.GridTask(goal_visible_steps=4, seed=seed)
    data = make_data(cfg, task, seed=seed)
    provider = make_episode_memory_provider(db_path=cfg.resolve(f"data/abl_mem_{seed}.db"),
                                            visible_steps=task.goal_visible_steps)

    with_mem = train(cfg, seed=seed, epochs=epochs, task=task, device=str(dev), quiet=True,
                     use_memory_input=True, mem_provider=provider)
    rows_mem = make_data(cfg, task, seed=seed)          # 同 seed ⇒ 同数据；这次给每行填上 mem
    for r in rows_mem["test"]:
        r["mem"] = provider(r)
    acc_on = evaluate(with_mem.model, with_mem.probe, rows_mem["test"],
                      device=dev, occluded_only=True)
    acc_on_all = evaluate(with_mem.model, with_mem.probe, rows_mem["test"], device=dev)
    zeroed = [{**r, "mem": np.zeros((r["len"], 16), dtype=np.float32)} for r in rows_mem["test"]]
    acc_off = evaluate(with_mem.model, with_mem.probe, zeroed, device=dev, occluded_only=True)
    acc_off_all = evaluate(with_mem.model, with_mem.probe, zeroed, device=dev)

    no_mem = train(cfg, seed=seed, epochs=epochs, task=task, device=str(dev), quiet=True,
                   use_memory_input=False)
    acc_never = evaluate(no_mem.model, no_mem.probe, data["test"], device=dev, occluded_only=True)

    out = {
        "kind": "memory",
        "seed": seed,
        "task": D.task_to_dict(task),
        "dataset_hash": data["dataset_hash"],
        "retrieval": provider.stats,
        "trained_with_memory": {
            "occluded_accuracy_on": round(acc_on["accuracy_occluded"], 4),
            "occluded_accuracy_off": round(acc_off["accuracy_occluded"], 4),
            "delta_occluded": round(acc_on["accuracy_occluded"] - acc_off["accuracy_occluded"], 4),
            "accuracy_on": round(acc_on_all["accuracy"], 4),
            "accuracy_off": round(acc_off_all["accuracy"], 4),
            "probe_accuracy_on": round(acc_on["probe_accuracy"], 4),
            "probe_accuracy_off": round(acc_off["probe_accuracy"], 4),
            "train_wall_s": round(with_mem.wall_s, 2),
        },
        "never_trained_with_memory": {
            "occluded_accuracy": round(acc_never.get("accuracy_occluded", float("nan")), 4),
            "train_wall_s": round(no_mem.wall_s, 2),
        },
        "note": ("记忆通道的贡献 = 同一权重下 mem 检索向量 vs 全零的差。"
                 "SQLite 里存下的内容本身不等于网络参数学会了；"
                 "探针准确率衡量的是循环状态里的信息，不是数据库。"),
    }
    if not quiet:
        print(f"  memory  on={out['trained_with_memory']['occluded_accuracy_on']:.3f} "
              f"off={out['trained_with_memory']['occluded_accuracy_off']:.3f} "
              f"Δ={out['trained_with_memory']['delta_occluded']:+.3f} "
              f"(未训练记忆通道={out['never_trained_with_memory']['occluded_accuracy']:.3f}, "
              f"检索命中步={provider.stats['hits']}/{provider.stats['steps']})")
    db = cfg.resolve(f"data/abl_mem_{seed}.db")
    if db.exists():
        db.unlink()
    return out


# ---------------------------------------------------------------------- 云端消融

def run_cloud_ablation(cfg: Config | None = None, *, seed: int = 11,
                       n_rows: int = 4, device: str = "auto",
                       quiet: bool = False) -> dict[str, Any]:
    """off / ask / auto_allowed 三档：只统计请求、放行、拒绝原因与预算占用。

    全程 MockProvider —— 不产生任何真实网络请求，因此这里量的是**策略行为**，
    不是云端延迟。真实 provider 的可用性检查在 scripts/check_cloud.py（需自备密钥）。
    """
    from ..bridge.providers import MockProvider
    from ..runtime.engine import NMindEngine

    cfg = cfg or default_config()
    task = D.GridTask(goal_visible_steps=3, seed=seed)
    data = make_data(cfg, task, seed=seed)
    rows = data["test"][:n_rows]
    out: dict[str, Any] = {"kind": "cloud", "seed": seed, "n_rows": len(rows),
                           "provider": "mock", "modes": {}}
    for mode in ("off", "ask", "auto_allowed"):
        c = cfg.model_copy(update={"cloud": cfg.cloud.model_copy(update={
            "mode": mode,
            "require_explicit_approval": mode != "auto_allowed",
            "cost_budget_units_total": 0.0 if mode == "off" else 10.0,
            "cost_estimate_per_call": 0.0 if mode == "off" else 0.01,
        })})
        eng = NMindEngine(c, use_memory=False, seed=seed, session_id=f"cloud_{mode}")
        eng.provider = MockProvider(answer="MOCK 答复：目标在较远处，建议绕行")
        used = denied = steps = 0
        reasons: dict[str, int] = {}
        for r in rows:
            for t in range(r["obs"].shape[0]):
                oc = eng.step(r["obs"][t], action=int(r["actions"][t - 1]) if t else None,
                              record=False)
                steps += 1
                used += int(oc.cloud_used)
                if not oc.cloud_used and oc.cloud_summary:
                    reason = str(oc.cloud_summary.get("reason", ""))
                    if reason:
                        reasons[reason] = reasons.get(reason, 0) + 1
                        denied += 1
        out["modes"][mode] = {"steps": steps, "cloud_used_steps": used,
                              "denied_steps": denied, "deny_reasons": reasons,
                              "policy": eng.policy.status(),
                              "provider_calls": eng.cloud_calls}
        eng.close()
    out["note"] = ("off 与 ask 的请求都在策略层被拒，未发起任何调用；"
                   "auto_allowed 才真正走到 provider（本机为 Mock，无真实网络）。")
    if not quiet:
        for m, d in out["modes"].items():
            print(f"  cloud {m:13s} used={d['cloud_used_steps']:3d} denied={d['denied_steps']:3d} "
                  f"calls={d['provider_calls']:3d} reasons={d['deny_reasons']}")
    return out


# ---------------------------------------------------------------------- 汇总

def run_ablations(cfg: Config | None = None, *, seeds: tuple[int, ...] = (11, 12, 13),
                  out_dir: str | Path | None = "results", include_memory: bool = True,
                  device: str = "auto", quiet: bool = False) -> dict[str, Any]:
    t0 = time.perf_counter()
    data: dict[str, Any] = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                            "seeds": list(seeds)}
    data["structural"] = run_structural_ablations(cfg, seeds=seeds, device=device, quiet=quiet)
    if include_memory:
        data["memory"] = run_memory_ablation(cfg, seed=seeds[0], device=device, quiet=quiet)
    data["cloud"] = run_cloud_ablation(cfg, seed=seeds[0], device=device, quiet=quiet)
    data["wall_s"] = round(time.perf_counter() - t0, 1)
    if out_dir is not None:
        data["files"] = write_reports(data, out_dir)
    return data


def write_reports(data: dict[str, Any], out_dir: str | Path, *,
                  stem: str = "ablation") -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    js = out / f"{stem}.json"
    js.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    rows = data.get("structural", {}).get("variants") or []
    cols = ["variant", "n_seeds", "accuracy_occluded", "accuracy_all", "probe_accuracy",
            "mean_ms", "p50_ms", "p95_ms", "per_event_ms", "peak_rss_mb",
            "activation_ratio", "top1_block_share", "dead_blocks", "train_wall_s", "note"]
    lines = [",".join(cols)]
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c, "")
            if isinstance(v, dict):
                v = v.get("mean", "")
            v = str(v).replace('"', "'")
            cells.append(f'"{v}"' if ("," in v or '"' in v) else v)
        lines.append(",".join(cells))
    csv = out / f"{stem}.csv"
    csv.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    return {"json": str(js), "csv": str(csv)}


def _cfg_hash(cfg: Config) -> str:
    from ..learning.checkpoint import config_hash

    return config_hash(cfg)


__all__ = ["Variant", "DEFAULT_VARIANTS", "run_structural_ablations", "run_memory_ablation",
           "run_cloud_ablation", "run_ablations", "write_reports",
           "make_episode_memory_provider"]
