"""GPU 冒烟实验：CUDA 可用性 + GPU/CPU 短训对比 + GPU训练→CPU推理一致性。

回答三个问题（seed 全部固定 20261121，数字来自本次实际运行）：
  G. torch CUDA 轮子在本机（RTX 5070, sm_120）是否真的可用、显存多少。
  T. 同一份 6 epoch demo 训练，GPU 与 CPU 的墙钟时间对比（小模型+小批次，
     GPU 未必更快——传输与 kernel 启动开销可能吃掉收益，如实记录）。
  X. GPU 上训练保存的 checkpoint，加载到 CPU 推理是否逐位一致
     （"GPU 训练、CPU 推理"是项目硬要求）。

用法（仓库根目录）:
    python experiments/2026-11-21_gpu_smoke.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nmind.config import default_config  # noqa: E402
from nmind.learning.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from nmind.learning.trainer import train  # noqa: E402

SEED = 20261121


def run_train(device: str) -> dict:
    cfg = default_config()
    cfg = cfg.model_copy(update={"runtime": cfg.runtime.model_copy(update={"device": device})})
    t0 = time.perf_counter()
    res = train(cfg, epochs=6, seed=SEED, device=device, quiet=True)
    wall = time.perf_counter() - t0
    return {
        "device_used": str(res.model.block_load_count.device),
        "wall_s": round(wall, 2),
        "grad_steps": res.steps,
        "loss_first": round(res.curve["total"][0], 4),
        "loss_last": round(res.curve["total"][-1], 4),
        "test": res.final["eval"],
        "params": res.final["params"]["total"],
        "seed": res.final["seed"],
    }


def predict_actions(model, row) -> np.ndarray:
    model.eval()
    dev = next(model.parameters()).device          # 输入必须跟权重同一设备
    with torch.no_grad():
        out = model.forward_sequence(torch.tensor(row["obs"][None], device=dev),
                                     torch.tensor(row["actions"][None], device=dev))
        return out["action_probs"][0].cpu().numpy()


def main() -> int:
    out: dict = {"experiment": "GPU 冒烟：可用性 + 训练对比 + GPU→CPU 一致性",
                 "seed": SEED, "date": "2026-11-21", "torch": torch.__version__}

    # --- G. CUDA 可用性 ---
    ok = torch.cuda.is_available()
    out["cuda_available"] = ok
    if ok:
        out["gpu"] = {"name": torch.cuda.get_device_name(0),
                      "capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
                      "vram_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2),
                      "vram_reserved_mb": round(torch.cuda.memory_reserved(0) / 2**20, 1)}
        print(f"[G] CUDA OK: {out['gpu']['name']} sm_{out['gpu']['capability'].replace('.', '')} "
              f"{out['gpu']['vram_total_gb']}GB", flush=True)
    else:
        print("[G] CUDA 不可用（驱动或轮子问题），只跑 CPU 对照", flush=True)

    # --- T. GPU/CPU 短训对比 ---
    if ok:
        out["train_gpu"] = run_train("cuda")
        t = out["train_gpu"]
        print(f"[T-GPU] {t['grad_steps']} 步 {t['wall_s']}s  loss {t['loss_first']}→{t['loss_last']}  "
              f"occ={t['test'].get('accuracy_occluded', float('nan')):.3f}  "
              f"显存峰值 {torch.cuda.max_memory_allocated(0) / 2**20:.0f}MB", flush=True)
        out["train_gpu"]["gpu_peak_mb"] = round(torch.cuda.max_memory_allocated(0) / 2**20, 1)
        torch.cuda.reset_peak_memory_stats(0)

    out["train_cpu"] = run_train("cpu")
    t = out["train_cpu"]
    print(f"[T-CPU] {t['grad_steps']} 步 {t['wall_s']}s  loss {t['loss_first']}→{t['loss_last']}  "
          f"occ={t['test'].get('accuracy_occluded', float('nan')):.3f}", flush=True)

    # --- X. GPU 训练 → CPU 推理逐位一致 ---
    if ok:
        cfg = default_config()
        cfg = cfg.model_copy(update={"runtime": cfg.runtime.model_copy(update={"device": "cuda"})})
        g = train(cfg, epochs=6, seed=SEED, device="cuda", quiet=True)
        ckpt = ROOT / "results" / "gpu_smoke.ckpt.pt"
        save_checkpoint(g.model, cfg=cfg, path=ckpt, probe=g.probe,
                        meta={"seed": SEED, "trained_on": "cuda"})
        cpu_model, _meta = load_checkpoint(ckpt, cfg=cfg, device="cpu")

        from nmind.learning import datasets as D
        from nmind.learning.trainer import make_data
        data = make_data(cfg, D.GridTask(goal_visible_steps=4, seed=SEED), seed=SEED)
        max_diff = 0.0
        for row in data["test"][:8]:
            p_gpu = predict_actions(g.model, row)
            p_cpu = predict_actions(cpu_model, row)
            max_diff = max(max_diff, float(np.abs(p_gpu - p_cpu).max()))
        out["gpu_train_cpu_infer"] = {
            "checkpoint": str(ckpt),
            "max_prob_diff_gpu_vs_cpu": round(max_diff, 8),
            "bitwise_identical": max_diff == 0.0,
            "n_rows_checked": 8,
            "note": "GPU(NCHW/不同kernel)与CPU浮点非结合顺序可能不同；"
                    "一致=完全可复现，不一致=记录实际差异量级，不许说谎。",
        }
        print(f"[X] GPU训练→CPU推理 最大概率差 {max_diff:.2e} "
              f"({out['gpu_train_cpu_infer']['bitwise_identical'] and '逐位一致' or '有浮点差异,见JSON'})",
              flush=True)

    dest = ROOT / "experiments" / "2026-11-21_gpu_smoke.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"→ {dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
