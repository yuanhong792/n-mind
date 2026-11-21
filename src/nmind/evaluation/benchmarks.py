"""CPU 基准：端到端延迟分布、峰值内存、路由与预测器的附加成本。

测量纪律（这些不是形式主义，是为了让数字能被别人复算）：
  * 先 warmup 再采样，冷启动的 mmap/线程池开销不算稳态延迟；
  * P50/P95 用同一批样本、同一随机种子；
  * 用**序列级墙钟**而不是算子求和：稀疏调度本身的开销必须包含在结论里，
    否则就是在拿对自己有利的口径比较；
  * 逐样本与批量分开测：批内共享计算会掩盖真实差异；
  * RSS 取 max，不取首尾平均；
  * 没有电力传感器 ⇒ 能耗字段恒为 None，报告里写"未测量"，绝不换算成假数字。
"""

from __future__ import annotations

import gc
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import torch

from ..config import Config, default_config
from ..core.architecture import NModel
from ..core.predictor import WorldPredictor
from ..learning import datasets as D
from ..learning.trainer import _assemble, make_data


def _rss_mb() -> float | None:
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2**20
    except Exception:                 # noqa: BLE001 - 缺 psutil 就只报时间，不假装测了内存
        return None


@dataclass
class Timing:
    n: int
    p50_ms: float
    p95_ms: float
    mean_ms: float
    max_ms: float
    rss_before_mb: float | None
    rss_peak_mb: float | None
    rss_after_mb: float | None
    total_s: float
    events_per_s: float
    blocks_touched: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        out = {**self.__dict__}
        for k in ("p50_ms", "p95_ms", "mean_ms", "max_ms", "total_s", "events_per_s"):
            out[k] = round(float(out[k]), 4)
        ev_s = float(out["events_per_s"])
        out["per_event_ms"] = round(1000.0 / ev_s, 4) if ev_s > 0 else None
        for k in ("rss_before_mb", "rss_peak_mb", "rss_after_mb"):
            out[k] = None if out[k] is None else round(float(out[k]), 1)
        return out


def measure(fn: Callable[[int], Any], *, n: int, warmup: int, per_event: int = 1,
            label: str = "") -> Timing:
    """n 次调用 fn(i)，返回延迟分布 + 内存轨迹。"""
    t0_wall = time.perf_counter()
    rss_before = _rss_mb()
    for i in range(warmup):
        fn(i % n)
    gc.collect()
    rss_start = _rss_mb()
    samples: list[float] = []
    peak = rss_start if rss_start is not None else None
    blocks = 0
    for i in range(n):
        t0 = time.perf_counter()
        res = fn(i)
        samples.append((time.perf_counter() - t0) * 1000.0)
        if isinstance(res, dict) and "blocks_touched" in res:
            blocks += int(res["blocks_touched"])
        cur = _rss_mb()
        if cur is not None:
            peak = cur if peak is None else max(peak, cur)
    total = time.perf_counter() - t0_wall
    ss = sorted(samples)
    events = n * per_event
    return Timing(n=n,
                  p50_ms=statistics.median(ss),
                  p95_ms=ss[min(len(ss) - 1, int(0.95 * (len(ss) - 1)))],
                  mean_ms=statistics.fmean(ss),
                  max_ms=ss[-1],
                  rss_before_mb=rss_before, rss_peak_mb=peak, rss_after_mb=_rss_mb(),
                  total_s=total, events_per_s=(events / total) if total > 0 else 0.0,
                  blocks_touched=blocks, meta={"label": label, "warmup": warmup})


def _submodule_ms(model: NModel, batch: dict[str, torch.Tensor], *, steps: int = 20,
                  warmup: int = 5) -> dict[str, float]:
    """分别空跑各子模块，量"路由器与预测器本身要花多少时间"。

    这是指令第 4 节点名的负收益风险：Router+Predictor 的开销可能吃掉稀疏省下的算力。
    只用整链墙钟看不出来，所以单独测。
    """
    obs, acts = batch["obs"], batch["actions"]
    b, T, _ = obs.shape
    z = model.encoder(obs.reshape(-1, obs.shape[-1]).view(b * T, -1)[:b], None)
    state = model.initial_state(b, obs.device)
    dummy_mask = torch.zeros(b, model.cfg.n_blocks)
    dummy_idx = torch.arange(min(model.top_k, model.cfg.n_blocks), device=obs.device)
    dummy_idx = dummy_idx.unsqueeze(0).expand(b, -1).contiguous()
    acts_idx = torch.arange(model.n_actions, device=obs.device)

    parts = {
        "encoder": lambda: model.encoder(obs.reshape(-1, obs.shape[-1])[:b], None),
        "router": lambda: model.router.route(z, state.h, torch.zeros(b, model.cfg.workspace_dim),
                                             torch.full((b, 1), 0.5), torch.full((b, 1), 0.5),
                                             k=model.top_k),
        "workspace_read": lambda: model.workspace.read(state.ws, z),
        "blocks_dense_all": lambda: model.blocks(z, state.h,
                                                 torch.zeros(b, model.cfg.workspace_dim),
                                                 torch.ones(b, model.cfg.n_blocks)),
        "blocks_sparse_topk": lambda: model.blocks.forward_sparse(z, state.h,
                                                                  torch.zeros(b, model.cfg.workspace_dim),
                                                                  dummy_idx),
        "predictor": lambda: model.predictor(z, model.action_emb(acts_idx[:b % model.n_actions]
                                                                 if b <= model.n_actions
                                                                 else acts_idx.repeat(int(np.ceil(b / model.n_actions)))[:b]),
                                            torch.zeros(b, model.cfg.workspace_dim)),
        "head": lambda: model.head.decide(state.h.mean(1), torch.zeros(b, model.cfg.workspace_dim),
                                          torch.zeros(b, model.mem_dim)),
    }
    out: dict[str, float] = {}
    with torch.no_grad():
        for name, fn in parts.items():
            for _ in range(warmup):
                fn()
            t0 = time.perf_counter()
            for _ in range(steps):
                fn()
            out[name] = round((time.perf_counter() - t0) * 1000.0 / steps, 4)
    return out


def cpu_benchmark(cfg: Config | None = None, *, seed: int | None = None,
                  n_blocks_list: tuple[int, ...] = (8,), top_ks: tuple[int, ...] = (1, 2, 4),
                  seq_len: int = 48, device: str = "auto", repeats: int = 100,
                  warmup: int = 20, quiet: bool = False) -> dict[str, Any]:
    """同一份权重（同 seed）在稠密与不同 k 下的延迟对比 + 子模块耗时表。"""
    from ..runtime.hardware import detect

    cfg = cfg or default_config()
    seed = int(cfg.runtime.seed if seed is None else seed)
    dev_obj, report = detect(device, torch_threads=cfg.runtime.torch_threads)
    dev = dev_obj if isinstance(dev_obj, torch.device) else torch.device(str(dev_obj))
    torch.manual_seed(seed)

    task = D.GridTask(goal_visible_steps=4, seed=seed)
    data = make_data(cfg, task, seed=seed)
    rows = data["test"]
    # 定长长序列：T=seq_len，让"事件吞吐"这个口径在各组之间可比
    # repeats 可能超过测试轨迹数：按 i % len(rows) 循环取用，保证 n=repeats 次都有样本
    long_rows = []
    for i in range(repeats):
        r = rows[i % len(rows)]
        long_rows.append(dict(r, obs=r["obs"][:seq_len], actions=r["actions"][:seq_len],
                              mem=r.get("mem", np.zeros((seq_len, 16), np.float32))[:seq_len]))
    batch = _assemble(long_rows, dev)
    idx = list(range(len(long_rows)))

    results: dict[str, Any] = {"hardware": report.as_dict(), "seed": seed,
                               "seq_len": seq_len, "repeats": repeats,
                               "batched": {}, "per_event": {}, "submodule_ms": {},
                               "threads": int(torch.get_num_threads())}

    for nb in n_blocks_list:
        m_cfg = cfg.model.model_copy(update={"n_blocks": int(nb), "top_k": min(cfg.model.top_k, nb)})
        dense = NModel(m_cfg, n_actions=D.N_ACTIONS, mem_dim=16, execution="dense",
                       use_router=False).to(dev).eval()
        with torch.no_grad():
            t = measure(lambda i: dense.forward_sequence(batch["obs"][[i]], batch["actions"][[i]]),
                        n=repeats, warmup=warmup, per_event=seq_len,
                        label=f"dense_n{nb}")
        results["per_event"][f"dense_n{nb}"] = t.as_dict()
        with torch.no_grad():
            t = measure(lambda i: dense.forward_sequence(batch["obs"], batch["actions"]),
                        n=repeats // 10 or 1, warmup=warmup // 10 or 1,
                        per_event=seq_len * len(idx), label=f"dense_batch_n{nb}")
        results["batched"][f"dense_n{nb}"] = t.as_dict()

        sparse = NModel(m_cfg, n_actions=D.N_ACTIONS, mem_dim=16, execution="sparse",
                        use_router=True).to(dev).eval()
        for k in top_ks:
            if k > nb:
                continue
            with torch.no_grad():
                t = measure(lambda i, k=k: sparse.forward_sequence(batch["obs"][[i]],
                                                                   batch["actions"][[i]], k=k),
                            n=repeats, warmup=warmup, per_event=seq_len, label=f"sparse_k{k}_n{nb}")
            results["per_event"][f"sparse_k{k}_n{nb}"] = t.as_dict()
            with torch.no_grad():
                t = measure(lambda i, k=k: sparse.forward_sequence(batch["obs"], batch["actions"], k=k),
                            n=max(1, repeats // 10), warmup=max(1, warmup // 10),
                            per_event=seq_len * len(idx), label=f"sparse_batch_k{k}_n{nb}")
            results["batched"][f"sparse_k{k}_n{nb}"] = t.as_dict()
        results["submodule_ms"][f"n{nb}"] = _submodule_ms(sparse, batch)

    d = results["per_event"]
    comp = {}
    for key in [k for k in d if k.startswith("sparse_")]:
        nb = key.rsplit("n", 1)[-1]
        base = f"dense_n{nb}"
        if base in d:
            speedup = d[base]["mean_ms"] / max(1e-9, d[key]["mean_ms"])
            comp[key] = {"vs_dense_mean": round(speedup, 4),
                         "p50_delta_ms": round(d[base]["p50_ms"] - d[key]["p50_ms"], 4),
                         "blocks_touched_ratio": round(
                             d[key]["blocks_touched"] / max(1, d[base]["blocks_touched"]), 4)}
    results["comparison"] = comp
    results["energy"] = None
    results["energy_note"] = "未测量：本机无电力传感器，不做任何换算估计"
    if not quiet:
        for key, v in d.items():
            print(f"  {key:24s} mean={v['mean_ms']:8.3f}ms p50={v['p50_ms']:8.3f} "
                  f"p95={v['p95_ms']:8.3f} ev/s={v['events_per_s']:7.1f}")
    return results
