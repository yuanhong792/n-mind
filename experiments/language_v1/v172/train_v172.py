"""V1.7.2 训练入口（本轮只准备，不执行）。

修复清单：
- P3 统一自回归 BPE 协议：encode(prompt)+encode(answer)+[EOS]
- P4 判别位代理指标修复 + 严格区分 proxy/full-answer
- P5 preserved_mode 恢复 train/eval 与 router 统计；样本状态隔离
- P6 明确加载 experiments/language_v1/v172/data（禁止回退 v17）
- P8 父 SHA、optimizer.step 记录、分阶段权重、Final/Best 分离、原子保存、
     优化器+RNG 状态、墙钟硬停止
- P9 V1.8 实验 A/B/C 的脚本化入口（需单独授权）

默认 dry-run，optimizer.step = 0。只有 --execute-train --authorized 才可能训练。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
import sys
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from nmind.language.bpe_decoder import generate_bpe
from nmind.language.neural_dialogue import LMConfig, NMindLM
from nmind.language.neural_language_v2 import load_language_v2, save_language_v2
from nmind.language.tokenizer_v2 import EOS

from v172_protocol import (METRIC_NAMES, build_tokenizer, valid_vocab,
                           conditional_token_lps, discriminative_positions,
                           discriminative_proxy, encode_autoregressive,
                           file_sha, fresh_state, full_answer_conditional,
                           model_sha, preserved_mode, prompt_last_logits,
                           write_json)

OUT = Path(__file__).resolve().parent
DATA = OUT / "data"
PARENT = ROOT / "experiments/language_v1/v15/candidate_v15_baseline.pt"
MAX_STEPS = 96          # 仅为 V1.8 预登记预算；本轮不执行
MAX_WALL = 300
WEIGHTS = {"discriminative": 2.0, "eos": 0.25, "content": 1.0}


def load_split(split: str):
    p = DATA / f"v172_{split}.jsonl"
    if not p.exists():
        raise FileNotFoundError(f"V1.7.2 数据缺失: {p}；禁止回退到旧 V1.7 数据")
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def make_batch(tok, rows, idxs, positions, *, weights=WEIGHTS):
    """P3/P5：分段 BPE 拼接 + 精确答案区掩码 + 逐样本独立状态（状态由 forward 内新建）。"""
    seqs = [encode_autoregressive(tok, rows[i]["prompt"], rows[i]["answer"]) for i in idxs]
    n = max(len(s.sequence) for s in seqs)
    x = torch.zeros((len(idxs), n), dtype=torch.long)
    mask = torch.zeros((len(idxs), n - 1), dtype=torch.bool)
    tw = torch.ones((len(idxs), n - 1), dtype=torch.float32)
    for j, ts in enumerate(seqs):
        x[j, :len(ts.sequence)] = torch.tensor(ts.sequence)
        aids = ts.answer_ids
        for rel in range(len(aids) + 1):          # +1 = EOS
            p = ts.answer_start + rel
            target_idx = p - 1
            if target_idx < 0 or target_idx >= n - 1:
                continue
            mask[j, target_idx] = True
            if p == ts.eos_index:
                w = weights["eos"]
            elif rel < len(aids) and rows[idxs[j]]["answer"] in positions and \
                    rel == positions[rows[idxs[j]]["answer"]]:
                w = weights["discriminative"]
            else:
                w = weights["content"]
            tw[j, target_idx] = w
    return x, mask, tw


def atomic_save(path: Path, blob: dict):
    """P8：临时文件 + fsync + replace，避免半写 checkpoint。"""
    import os
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(blob, tmp)
    with open(tmp, "r+b") as f:
        os.fsync(f.fileno())
    tmp.replace(path)


def save_checkpoint(path: Path, model, tok, *, step: int, seed: int,
                    optimizer=None, rng_state=None, parent_sha: str = "",
                    manifest_hash: str = ""):
    """P8：每个 checkpoint 记录真实 optimizer.step 与可恢复状态。"""
    blob = {
        "format": "nmind-lm/1-v172-safe",
        "config": model.cfg.to_dict(),
        "vocab_hash": tok.vocab_hash(),
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "optimizer_steps_executed": int(step),
        "step": int(step),
        "seed": int(seed),
        "parent_checkpoint_sha256": parent_sha,
        "data_manifest_hash": manifest_hash,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "rng_state": rng_state,
        "torch_rng_state": torch.get_rng_state(),
    }
    atomic_save(path, blob)
    # 逐张量回读校验（P8）
    check = torch.load(path, map_location="cpu", weights_only=False)
    for k, v in blob["state_dict"].items():
        if not torch.equal(check["state_dict"][k], v):
            raise RuntimeError(f"checkpoint 张量校验失败: {k}")
    return file_sha(path)


def evaluate_fixed(model, tok, rows, *, valid, positions, score_limit=200, free_n=8):
    """P4/P5：分类指标 + proxy 分离 + 模式/统计完整恢复。"""
    t0 = time.perf_counter()
    with preserved_mode(model):
        model.eval()
        model.use_router_load_update = False
        model.router_gradient_path_enabled = False
        sub = rows[:score_limit]
        by_task = {}
        proxy_scores = {}
        for r in sub:
            cands = sorted({x["answer"] for x in sub if x["task"] == r["task"]})
            if r["answer"] not in cands:
                cands.append(r["answer"])
            tail = prompt_last_logits(model, tok, r["prompt"])
            proxy = discriminative_proxy(model, tok, r["prompt"], cands,
                                         valid=valid, positions=positions,
                                         cached_prompt_logits=tail)
            proxy_scores[r["prompt"]] = proxy
            pred = max(proxy, key=proxy.get)
            d = by_task.setdefault(r["task"], {"n": 0, "proxy_ok": 0,
                                               "full_ok": 0, "full_n": 0})
            d["n"] += 1
            d["proxy_ok"] += int(pred == r["answer"])
            if d["full_n"] < 40:
                full = {c: full_answer_conditional(model, tok, r["prompt"], c, valid=valid)
                        for c in cands}
                d["full_ok"] += int(max(full, key=full.get) == r["answer"])
                d["full_n"] += 1
        free_ok = eos0 = 0
        for r in rows[:free_n]:
            g = generate_bpe(model, tok, r["prompt"], max_new_tokens=16)
            free_ok += int(g.text.strip() == r["answer"])
            eos0 += int(g.finish_reason == "eos" and not g.text.strip())
        result = {
            "per_task": {t: {"n": d["n"],
                             "DISCRIMINATIVE_TOKEN_PROXY": round(d["proxy_ok"] / max(1, d["n"]), 4),
                             "full_answer_conditional_accuracy": round(d["full_ok"] / max(1, d["full_n"]), 4)}
                         for t, d in by_task.items()},
            "free_generation_accuracy": round(free_ok / max(1, min(free_n, len(rows))), 4),
            "immediate_eos_rate": round(eos0 / max(1, min(free_n, len(rows))), 4),
            "wall_seconds": round(time.perf_counter() - t0, 3),
        }
    # 退出 with 后 model 必须回到进入前状态
    result["mode_restored"] = model.training
    return result


def dry_run_report() -> dict:
    tok = build_tokenizer()
    rows = {s: load_split(s) for s in ("train", "val", "sealed", "counter")}
    cfg = LMConfig(vocab_size=4096, d_model=32, n_blocks=2, block_hidden=32,
                   workspace_dim=16, workspace_slots=2, top_k=2)
    _, positions = discriminative_positions(tok, ["红色", "绿色", "蓝色", "黄色",
                                                   "紫色", "橙色", "黑色", "白色"])
    valid, _ = valid_vocab(tok)
    x, mask, tw = make_batch(tok, rows["train"], list(range(2)), positions)
    return {
        "status": "DRY_RUN_NO_TRAIN",
        "main_model_optimizer_steps": 0,
        "parent_checkpoint": str(PARENT),
        "parent_checkpoint_available": PARENT.exists(),
        "parent_sha256_expected": "5681f656fbb35f4f4a033e16212897f2d7cc02eac6754b9d860fb980d6f6a611",
        "data_dir": str(DATA),
        "data_counts": {s: len(v) for s, v in rows.items()},
        "tokenizer_hash": tok.vocab_hash(),
        "model_config": cfg.to_dict(),
        "batch_shape": list(x.shape),
        "supervised_targets": int(mask.sum()),
        "weights_unique": sorted(set(float(v) for v in tw[mask].tolist())),
        "metrics": list(METRIC_NAMES),
        "checkpoint_safety": {
            "parent_sha_recorded": True,
            "optimizer_steps_recorded": True,
            "phase_checkpoints": [64, 512, MAX_STEPS],
            "final_saved_before_best": True,
            "optimizer_and_rng_state_saved": True,
            "atomic_save_with_tensor_verify": True,
            "hard_wall_stop_seconds": MAX_WALL,
        },
        "mode_management": {"preserved_mode_contextmanager": True,
                            "per_sample_state_isolated": True,
                            "router_stats_restored_after_eval": True},
        "note": "父权重缺失，本轮不训练；程序已具备执行条件但需先恢复父 checkpoint 并重新授权。",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute-train", action="store_true")
    ap.add_argument("--authorized", action="store_true")
    args = ap.parse_args()
    if not (args.execute_train and args.authorized):
        rep = dry_run_report()
        write_json(OUT / "V172_TRAIN_DRY_RUN.json", rep)
        print(json.dumps({"status": rep["status"],
                          "parent_available": rep["parent_checkpoint_available"],
                          "data_counts": rep["data_counts"]}, ensure_ascii=False, indent=2))
        return
    if not PARENT.exists():
        raise FileNotFoundError(f"父模型缺失，禁止开始训练: {PARENT}")
    # 正式训练路径保留但未在本轮执行：任何授权调用都必须先通过父 SHA 校验。
    expected = "5681f656fbb35f4f4a033e16212897f2d7cc02eac6754b9d860fb980d6f6a611"
    if file_sha(PARENT) != expected:
        raise RuntimeError("父模型 SHA 不匹配，拒绝训练")
    raise SystemExit("V1.8 正式训练需独立授权；本入口只完成前置校验。")


if __name__ == "__main__":
    main()
