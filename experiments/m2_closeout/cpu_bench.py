"""M2.4 独立 CPU 基准：重载 checkpoint 的端到端 p50/p95、吞吐、峰值 RSS。

与历史数字的关系：此前 benchmark_cpu.json 的"稀疏 1.68×"是在旧代码/旧环境上测的，
且延迟列被确认受机器噪声污染（提示词 B 节）。本脚本在当前机器独占运行时重测，
若 1.68× 不能复现，以本次结果为准并在报告中撤回旧口径。

口径（全部写进输出 JSON）：
  - 只用重载评分通过的 checkpoint；model.eval() + torch.no_grad()
  - 端到端 = encoder → 路由 → 块 → 工作空间 → 头 → 预测器（forward_sequence 整链）
  - batch=1、T=24（训练长度）、float32、输入在进程内预生成
  - 线程 {1, 4, 8}，每组独立 set_num_threads；预热 10 次，正式 60 次
  - 各线程组内随机交叉测试顺序（防系统性漂移）；p50/p95/mean
  - 峰值 RSS 用 psutil 采样进程级常驻内存
  - 冷启动单独测：新进程里首条序列的耗时（由 --cold 模式单独跑，不在热循环里混）

用法（建议独占机器时运行）：
  python experiments/m2_closeout/cpu_bench.py --arms A_dense_ce,C_sparse_ce,D_sparse_joint,F1_detach,F7_detach_soft_balent --seed 20261121
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path

import numpy as np
import psutil
import torch

ROOT = Path(__file__).resolve().parents[2]
sys_path = str(ROOT / "src")
if sys_path not in __import__("sys").path:
    __import__("sys").path.insert(0, sys_path)

from nmind.config import default_config  # noqa: E402
from nmind.learning import datasets as D  # noqa: E402
from nmind.learning.checkpoint import load_checkpoint  # noqa: E402
from nmind.learning.trainer import make_data  # noqa: E402

SRC = ROOT / "experiments" / "m2_ablation"
OUT = ROOT / "experiments" / "m2_closeout"


def rss_mb() -> float:
    proc = psutil.Process()
    return proc.memory_info().rss / (1 << 20)


def bench_one(model, rows, *, repeats: int, warmup: int, device: torch.device) -> dict:
    """batch=1、T=24 的单序列端到端延迟分布。"""
    model.eval()
    b = rows[0]
    obs = torch.tensor(b["obs"][None], device=device)
    acts = torch.tensor(b["actions"][None], device=device)
    with torch.no_grad():
        for _ in range(warmup):
            model.forward_sequence(obs, acts)
    samples: list[float] = []
    rss_peak = rss_mb()
    with torch.no_grad():
        for _ in range(repeats):
            r0 = rss_mb()
            t0 = time.perf_counter()
            model.forward_sequence(obs, acts)
            samples.append((time.perf_counter() - t0) * 1000.0)
            rss_peak = max(rss_peak, rss_mb(), r0)
    ss = sorted(samples)
    p95_idx = min(len(ss) - 1, int(0.95 * (len(ss) - 1)))
    return {"p50_ms": round(statistics.median(ss), 3),
            "p95_ms": round(ss[p95_idx], 3),
            "mean_ms": round(statistics.fmean(ss), 3),
            "min_ms": round(ss[0], 3),
            "seq_per_s": round(1000.0 / statistics.median(ss), 1),
            "rss_peak_mb": round(rss_peak, 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="A_dense_ce,C_sparse_ce,D_sparse_joint,F1_detach,F7_detach_soft_balent")
    ap.add_argument("--seed", type=int, default=20261121)
    ap.add_argument("--threads", default="1,4,8")
    ap.add_argument("--repeats", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    threads_list = [int(t) for t in args.threads.split(",")]
    cfg = default_config()
    device = torch.device("cpu")
    data = make_data(cfg, D.GridTask(goal_visible_steps=4, seed=args.seed), seed=args.seed)
    rows = data["test"]

    results: dict[str, dict] = {}
    base_rss = rss_mb()
    for n_threads in threads_list:
        torch.set_num_threads(n_threads)
        order = arms[:]
        random.Random(1234 + n_threads).shuffle(order)   # 组内随机交叉顺序
        for arm in order:
            ckpt = SRC / f"{arm}_s{args.seed}.pt"
            if not ckpt.exists():
                results.setdefault(arm, {})[f"threads_{n_threads}"] = {"missing": str(ckpt)}
                print(f"[skip] {ckpt} 不存在", flush=True)
                continue
            model, _meta = load_checkpoint(ckpt, cfg=cfg, device="cpu",
                                           allow_config_mismatch=True)
            t0 = time.perf_counter()
            res = bench_one(model, rows, repeats=args.repeats, warmup=args.warmup,
                            device=device)
            res["bench_wall_s"] = round(time.perf_counter() - t0, 1)
            results.setdefault(arm, {})[f"threads_{n_threads}"] = res
            print(f"[{arm} thr={n_threads}] p50={res['p50_ms']}ms p95={res['p95_ms']}ms "
                  f"throughput={res['seq_per_s']}/s rss={res['rss_peak_mb']}MB", flush=True)
            del model

    # 稀疏 vs 稠密的真实加速比（同线程数），端到端口径
    speedups = {}
    for n_threads in threads_list:
        a = results.get("A_dense_ce", {}).get(f"threads_{n_threads}", {})
        c = results.get("C_sparse_ce", {}).get(f"threads_{n_threads}", {})
        if a.get("p50_ms") and c.get("p50_ms"):
            speedups[f"threads_{n_threads}"] = round(a["p50_ms"] / c["p50_ms"], 3)

    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "env": {"torch": torch.__version__, "cpu": "i7-14700KF 28 逻辑核",
                "threads_tested": threads_list, "batch": 1, "T": 24, "dtype": "float32",
                "warmup": args.warmup, "repeats": args.repeats, "seed": args.seed,
                "base_rss_mb": round(base_rss, 1),
                "note": "独占运行时执行；与其他训练/评测并行时的数字不可比"},
        "results": results,
        "speedup_dense_vs_sparse_p50": speedups,
        "honesty": "这是端到端 wall-clock；'稀疏理论计算量减少'与'真实加速比'分开报告，不得混淆",
    }
    dest = OUT / "cpu_benchmark.json"
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"→ {dest}\nspeedup(dense/sparse p50): {speedups}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
