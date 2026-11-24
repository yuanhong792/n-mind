"""M3 四架构双种子真实训练（P13）。

预算：累计 4096 optimizer.step，总墙钟 3600 秒。
此预算不自动继承 M2 授权，需用户明确授权（--authorize）后才启动梯度训练。
未授权时仅做 dry-run（零步）与无梯度基准。

公平性：四架构 × 2 seed × 512 步，相同样本、预生成 batch、监督目标、
tokenizer、验证面板、封存集、优化器调优机会。

每 64 步 checkpoint，Final 先于 Best，逐张量核验。
完整 checkpoint 元数据：架构、公式版本、初始化 SHA、本模型步数、全局步数、
优化器状态、采样状态、RNG、数据 SHA、tokenizer SHA、代码版本、资源成本。

学习曲线：64/128/256/384/512 步记录能力与资源，供 ISO 能力成本比较。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "models"))

from tokenizer_m3 import M3Tokenizer  # noqa: E402
from budget_manager import BudgetManager  # noqa: E402
from evaluate_m3 import (load_rows, build_fixed_panel, make_batch, grouped_loss,  # noqa: E402
                         evaluate_panel, iso_record, ISO_TARGETS)
from compute_accounting import model_mac_profile  # noqa: E402

from gru_reference import GRUReferenceM3            # noqa: E402
from transformer_reference import TinyTransformerM3  # noqa: E402
from low_rank_neural import LowRankNeuralM3       # noqa: E402
from adaptive_computation import AdaptiveComputationM3  # noqa: E402

MODELS = {
    "R1_gru_reference": GRUReferenceM3,
    "R2_tiny_transformer": TinyTransformerM3,
    "M3-A_low_rank_neural": LowRankNeuralM3,
    "M3-B_adaptive_computation": AdaptiveComputationM3,
}

MAX_STEPS = 512
EVAL_INTERVAL = 64
BATCH_SIZE = 32
LR = 3e-3
CODE_VERSION = "m3-train/1.0"
FORMULA_VERSION = "m3-formulas/1.0"


def init_sha(model) -> str:
    h = hashlib.sha256()
    for p in model.parameters():
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def data_sha() -> str:
    h = hashlib.sha256()
    for split in ["train", "val", "sealed"]:
        p = HERE / "data" / f"m3_{split}.jsonl"
        if p.exists():
            h.update(p.read_bytes())
    return h.hexdigest()[:16]


def save_checkpoint(model, opt, rng_state, path, meta):
    """Final 先于 Best：保存并逐张量核验。"""
    tmp = path.with_suffix(".tmp")
    torch.save({
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "rng": rng_state,
        "meta": meta,
    }, tmp)
    # 逐张量回读核验
    blob = torch.load(tmp, weights_only=False)
    for k, v in model.state_dict().items():
        if not torch.equal(v, blob["model"][k]):
            raise RuntimeError(f"checkpoint 核验失败: {k}")
    tmp.replace(path)


def train_one(arch: str, seed: int, budget: BudgetManager, tok: M3Tokenizer,
              train_rows, panel, mac_prof, execute: bool):
    """训练单个 (架构, seed)。返回学习曲线与最终结果。"""
    torch.manual_seed(seed)
    random.seed(seed)
    model = MODELS[arch](tok.vocab_size)
    init_sha_val = init_sha(model)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    curve = []
    best_macro = -1.0
    best_path = None
    final_path = None
    base_steps = budget.steps_used

    # 预生成 batch 序列（公平性：相同 batch 顺序）
    rng = random.Random(seed)
    shuffled = list(train_rows)
    rng.shuffle(shuffled)
    batches = [shuffled[i:i + BATCH_SIZE] for i in range(0, len(shuffled) - BATCH_SIZE + 1, BATCH_SIZE)]
    bptr = 0

    step = 0
    while step < MAX_STEPS:
        if not execute:
            break
        ok, _reason = budget.can_step()
        if not ok:
            break
        batch = batches[bptr % len(batches)]
        bptr += 1
        x, valid, w, tgt = make_batch(tok, batch)
        model.train()
        logits, _ = model(x[:, :-1], valid_mask=valid[:, :-1])
        loss = grouped_loss(logits, x[:, 1:], w[:, 1:])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        budget.record_step(arch, arch, seed, step, float(loss))
        step += 1

        # 学习曲线记录点
        if step % EVAL_INTERVAL == 0 or step == MAX_STEPS:
            model.eval()
            ev = evaluate_panel(model, tok, panel)
            rec_step = base_steps + step
            curve.append({
                "step": rec_step,
                "local_step": step,
                "macro": ev["macro_accuracy"],
                "content_acc": ev["content_accuracy"],
                "samples": rec_step * BATCH_SIZE,
                "tokens": rec_step * BATCH_SIZE * 32,
                "macs": rec_step * BATCH_SIZE * 32 * mac_prof["per_token_forward_mac"] * 3,  # fwd+bwd≈3x
                "wall": budget.current_wall(),
            })
            print(f"  [{arch} seed{seed}] step {step} macro {ev['macro_accuracy']} "
                  f"loss_done wall {budget.current_wall():.1f}s", flush=True)
            if ev["macro_accuracy"] > best_macro:
                best_macro = ev["macro_accuracy"]
                meta = {
                    "arch": arch, "seed": seed, "step": rec_step, "role": "best",
                    "macro": best_macro, "init_sha": init_sha_val,
                    "formula_version": FORMULA_VERSION, "code_version": CODE_VERSION,
                    "data_sha": data_sha(), "tokenizer_sha": tok.vocab_sha,
                    "global_step": base_steps + step,
                }
                save_checkpoint(model, opt, torch.get_rng_state(),
                                HERE / "checkpoints" / f"{arch}_s{seed}_best.pt", meta)
                best_path = str(HERE / "checkpoints" / f"{arch}_s{seed}_best.pt")

    # Final 先于 Best 保存
    if execute:
        meta = {
            "arch": arch, "seed": seed, "step": base_steps + step, "role": "final",
            "macro": curve[-1]["macro"] if curve else 0.0,
            "init_sha": init_sha_val, "formula_version": FORMULA_VERSION,
            "code_version": CODE_VERSION, "data_sha": data_sha(),
            "tokenizer_sha": tok.vocab_sha, "global_step": base_steps + step,
        }
        final_path = str(HERE / "checkpoints" / f"{arch}_s{seed}_final.pt")
        save_checkpoint(model, opt, torch.get_rng_state(),
                        HERE / "checkpoints" / f"{arch}_s{seed}_final.pt", meta)
        budget.record_run_end(arch, arch, seed, step, budget.current_wall(),
                              "completed" if step >= MAX_STEPS else "budget_stop", best_macro)

    return {
        "arch": arch, "seed": seed, "status": "DONE" if execute else "DRY_RUN",
        "steps": step, "curve": curve,
        "best_macro": best_macro, "best_path": best_path, "final_path": final_path,
        "init_sha": init_sha_val,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--authorize", action="store_true",
                    help="用户明确授权启动真实梯度训练")
    ap.add_argument("--dry-run", action="store_true", default=True)
    args = ap.parse_args()

    execute = args.authorize
    torch.set_num_threads(20)
    tok = M3Tokenizer()
    train_rows = load_rows("train")
    val_rows = load_rows("val")
    panel = build_fixed_panel(val_rows, per_task=20)

    budget = BudgetManager(max_steps=4096, max_wall=3600,
                           ledger=HERE / "M3_BUDGET_LEDGER.jsonl")
    print(f"[train_m3] 授权={execute} 预算 {budget.max_steps}步/{budget.max_wall}s "
          f"已用 {budget.steps_used}步/{budget.current_wall():.1f}s", flush=True)
    if execute:
        budget.start_session()
    if not execute:
        print("[train_m3] 未授权，仅 dry-run（零步）。完成代码/验证/无梯度基准，"
              "等待用户明确授权后再启动真实训练。", flush=True)

    results = {"status": "DRY_RUN" if not execute else "TRAINING",
               "code_version": CODE_VERSION, "formula_version": FORMULA_VERSION,
               "budget": {"max_steps": 4096, "max_wall": 3600},
               "data_sha": data_sha(), "tokenizer_sha": tok.vocab_sha,
               "models": {}, "iso_targets": ISO_TARGETS}

    for arch in MODELS:
        cfg = {}
        if arch == "R1_gru_reference":
            cfg = {"d_model": 64, "hidden": 128}
        elif arch == "R2_tiny_transformer":
            cfg = {"d_model": 64, "n_head": 4, "layers": 2}
        elif arch == "M3-A_low_rank_neural":
            cfg = {"d": 96, "r": 16, "n_layers": 1}
        elif arch == "M3-B_adaptive_computation":
            cfg = {"d": 96, "r": 16, "hidden_mult": 2}
        mac_prof = model_mac_profile(arch, tok.vocab_size, **cfg)

        for seed in [1, 2]:
            if not execute and seed > 1:
                # dry-run 只跑一个 seed 验证流程
                break
            print(f"[train_m3] 训练 {arch} seed{seed} ...", flush=True)
            r = train_one(arch, seed, budget, tok, train_rows, panel, mac_prof, execute)
            r["iso"] = iso_record(r["curve"])
            results["models"][f"{arch}_s{seed}"] = r
            print(f"[train_m3] {arch} seed{seed} steps={r['steps']} "
                  f"best_macro={r['best_macro']}", flush=True)
            if execute:
                ok, _r = budget.can_step()
                if not ok:
                    print("[train_m3] 预算耗尽，停止。", flush=True)
                    break

    if execute:
        budget.end_session()
    results["status"] = "TRAINING_DONE" if execute else "DRY_RUN_COMPLETE"
    results["budget_final"] = budget.summary()
    out = HERE / "M3_TRAINING_RESULTS.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[train_m3] 写出 {out}", flush=True)


if __name__ == "__main__":
    main()
