"""M4 真实训练器（正控制 + 四架构双种子）。

预算：正控制 256 步（R1/R2 各 128）+ 正式 3840 步（4架构×2seed×480）= 4096。
**必须 --authorize 才执行梯度更新**；未授权仅 dry-run（零步）。
每 64 步评测并保存真实 checkpoint；Final 先于 Best；台账逐 step 登记。
所有模型随机初始化，不使用预训练权重/LLM/解释器/硬编码答案。

运行：
    python train_m4.py                 # dry-run（零步）
    python train_m4.py --authorize     # 需用户明确授权
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

HERE = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "models"))

from tokenizer_m4 import M4Tokenizer  # noqa: E402
from budget_manager import BudgetManager, STAGE_POSITIVE  # noqa: E402
from evaluate_m4 import (load_rows, build_fixed_panel, make_batch, grouped_loss,  # noqa: E402
                         eval_rows, iso_record)
from compute_accounting_m4 import batch_cost  # noqa: E402

from gru_reference import GRUReferenceM3  # noqa: E402
from transformer_reference import TinyTransformerM3  # noqa: E402
from low_rank_interaction import LowRankInteractionM4  # noqa: E402
from selective_interaction import SelectiveInteractionM4  # noqa: E402
from positive_control import control_rows  # noqa: E402

MODELS = {
    "R1_gru_reference": GRUReferenceM3,
    "R2_tiny_transformer": TinyTransformerM3,
    "M4-A_low_rank_interaction": LowRankInteractionM4,
    "M4-B_selective_interaction": SelectiveInteractionM4,
}
CFG = {
    "R1_gru_reference": {},
    "R2_tiny_transformer": {"d_model": 64, "nhead": 4, "layers": 2},
    "M4-A_low_rank_interaction": {"d_model": 64, "nhead": 4, "layers": 2, "r": 16},
    "M4-B_selective_interaction": {"d_model": 64, "nhead": 4, "layers": 2,
                                    "group_size": 4, "top_k": 2},
}
LR, BATCH_SIZE, CLIP = 0.001, 16, 1.0
EVAL_INTERVAL = 64
STEPS_MAIN = 480
STEPS_POS = 128


def sha_state(model) -> str:
    h = hashlib.sha256()
    for k, p in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def save_ckpt(model, opt, path, meta):
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "meta": meta}, tmp)
    blob = torch.load(tmp, weights_only=False)
    for k, v in model.state_dict().items():
        if not torch.equal(v, blob["model"][k]):
            raise RuntimeError(f"checkpoint 核验失败 {k}")
    tmp.replace(path)


def train_one(arch, seed, stage, budget, tok, train_rows, panel, execute):
    steps = STEPS_POS if stage == "positive" else STEPS_MAIN
    torch.manual_seed(seed)
    random.seed(seed)
    model = MODELS[arch](tok.vocab_size, **CFG[arch])
    init_sha = sha_state(model)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    rng = random.Random(seed + 1000)
    shuffled = list(train_rows)
    rng.shuffle(shuffled)
    batches = [shuffled[i:i + BATCH_SIZE] for i in range(0, len(shuffled), BATCH_SIZE)]
    bptr = 0
    curve, best, best_path = [], -1.0, None
    step = 0
    base = budget.steps_used
    wall0 = budget.current_wall()
    while step < steps and execute:
        ok, reason = budget.can_step(stage)
        if not ok:
            break
        batch = batches[bptr % len(batches)]
        bptr += 1
        x, valid, w, tgt = make_batch(tok, batch, "A")
        model.train()
        logits, _ = model(x[:, :-1], valid_mask=valid[:, :-1])
        loss = grouped_loss(logits, tgt[:, 1:], w[:, 1:])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        opt.step()
        bc = batch_cost(model, x, valid, w, backward=True, optimizer=True)
        budget.record_step(f"{arch}_s{seed}_{stage}", arch, seed, step, float(loss.item()),
                           stage, {"B": bc.batch_size, "T": bc.tensor_length,
                                   "valid": bc.valid_tokens, "supervised": bc.supervised_tokens},
                           bc.analytic_fwd_macs + bc.analytic_bwd_macs)
        step += 1
        if step % EVAL_INTERVAL == 0 or step == steps:
            model.eval()
            ev = eval_rows(model, tok, panel)
            curve.append({"step": base + step, "local_step": step, "macro": ev["macro_accuracy"],
                          "content": ev["content_accuracy"],
                          "full_eos": ev["full_answer_natural_eos"]})
            print(f"  [{arch} s{seed} {stage}] {step}/{steps} macro={ev['macro_accuracy']} "
                  f"wall={budget.current_wall():.1f}s", flush=True)
            if ev["macro_accuracy"] > best:
                best = ev["macro_accuracy"]
                best_path = HERE / "checkpoints" / f"{arch}_s{seed}_{stage}_best.pt"
                save_ckpt(model, opt, best_path,
                          {"arch": arch, "seed": seed, "stage": stage, "step": base + step,
                           "role": "best", "macro": best, "init_sha": init_sha,
                           "tokenizer_sha": tok.vocab_sha, "lr": LR})

    if execute and step > 0:
        final_path = HERE / "checkpoints" / f"{arch}_s{seed}_{stage}_final.pt"
        save_ckpt(model, opt, final_path,
                  {"arch": arch, "seed": seed, "stage": stage, "step": base + step,
                   "role": "final", "macro": curve[-1]["macro"] if curve else 0.0,
                   "init_sha": init_sha, "tokenizer_sha": tok.vocab_sha, "lr": LR})
        budget.record_run_end(f"{arch}_s{seed}_{stage}", arch, seed, step,
                              budget.current_wall() - wall0,
                              "completed" if step == steps else "budget_stop", stage, best,
                              hashlib.sha256(final_path.read_bytes()).hexdigest()[:16])
    return {"arch": arch, "seed": seed, "stage": stage, "status": "DONE" if execute else "DRY_RUN",
            "steps": step, "init_sha": init_sha, "best_macro": best, "curve": curve,
            "best_path": str(best_path) if best_path else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--authorize", action="store_true")
    ap.add_argument("--stage", choices=["all", "positive", "main"], default="all")
    args = ap.parse_args()
    torch.set_num_threads(20)
    tok = M4Tokenizer()
    train = load_rows("train")
    val = load_rows("val")
    panel = build_fixed_panel(val, per_task=20)
    budget = BudgetManager()
    print(f"[train_m4] authorized={args.authorize} stage={args.stage} "
          f"budget used={budget.steps_used}/{budget.max_steps}", flush=True)
    if args.authorize:
        budget.start_session("M4 authorized training")
    results = {"status": "TRAINING" if args.authorize else "DRY_RUN",
               "authorization": "explicit --authorize" if args.authorize else "NOT_AUTHORIZED",
               "tokenizer_sha": tok.vocab_sha, "panel_n": len(panel), "models": {}}
    if args.stage in ("all", "positive"):
        pos_rows = control_rows(tok, n=64)
        for arch in ("R1_gru_reference", "R2_tiny_transformer"):
            for seed in (1,):
                r = train_one(arch, seed, "positive", budget, tok, pos_rows, panel, args.authorize)
                results["models"][f"positive_{arch}_s{seed}"] = r
    if args.stage in ("all", "main"):
        for arch in MODELS:
            for seed in (1, 2):
                ok, _ = budget.can_step("main")
                if not ok:
                    break
                r = train_one(arch, seed, "main", budget, tok, train, panel, args.authorize)
                results["models"][f"main_{arch}_s{seed}"] = r
                results[f"main_{arch}_s{seed}_iso"] = iso_record(r["curve"])
    if args.authorize:
        budget.end_session()
    results["status"] = "TRAINING_DONE" if args.authorize else "DRY_RUN_COMPLETE"
    results["budget_final"] = budget.summary()
    p = HERE / "M4_TRAINING_RESULTS.json"
    p.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[train_m4] 写出 {p}")


if __name__ == "__main__":
    main()
