"""M3 v1 最小闭环：从随机初始化训练 N-MIND 核心做微型语言的下一字符预测与问答。

复用方式（一字节不改核心）：NModel 的决策头本身就是"输入状态→logits"的读出，
把 n_actions 设为词表大小 V，它就是 LM 头；字符以 one-hot 进 encoder（学习式嵌入
等价于 one-hot→MLP，全部从零训练）；action_emb 以上一字符 id 作为条件输入。
对齐：forward_sequence 在第 t 步看到 chars[0..t]，logits[t] 即 chars[t+1] 的分布。

训练口径（由 M2 证据决定，是假设不是结论）：
  - CE（只在答案字符位置）+ pred 损失（detach 目标，F1 语义）+ 软化均衡/熵（F7 语义）
    + router_ste=True（F10 候选机制：稀疏路由接任务梯度）
  - 这四项组合在 M2 收尾时点尚未多 seed 定案；M3 v1 用它是因为每一项单独都有
    实测依据（detach 使 D 0.57→0.90；任务盲路由已被梯度诊断证实）。记录在案。

评测：teacher-forced 答案字符准确率 + 每 span 精确匹配（自由生成留 M3.2）。
OOD：OOD-name（新实体）/ OOD-compose（新组合+全代词）/ OOD-length（更长串联），
外加递归深度 depth∈{1,2,4} 的对比（提示词 M3.2 要求）。

用法：
  python experiments/m3_language/run_m3_v1.py --steps 600
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from nmind.config import default_config  # noqa: E402
from nmind.language import micro_lang as L  # noqa: E402
from nmind.learning.losses import balance_loss, entropy_loss, prediction_loss  # noqa: E402
from nmind.learning.trainer import _make_probe  # noqa: E402
from nmind.core.architecture import NModel  # noqa: E402

OUT = ROOT / "experiments" / "m3_language"
CKPT = OUT / "m3_v1.ckpt.pt"


def make_batches(split: dict, *, batch_size: int, seed: int, pad_id: int):
    """按长度分桶后组批；返回 [(obs_onehot [B,T,V], ids [B,T], mask [B,T]), ...]。"""
    rng = np.random.default_rng(seed)
    eps = sorted(split["episodes"], key=lambda e: len(e["ids"]))
    batches = []
    for s in range(0, len(eps), batch_size):
        chunk = eps[s:s + batch_size]
        T = max(len(e["ids"]) for e in chunk)
        B = len(chunk)
        obs = np.zeros((B, T, L.V), dtype=np.float32)
        ids = np.full((B, T), pad_id, dtype=np.int64)
        mask = np.zeros((B, T), dtype=np.float32)
        for b, e in enumerate(chunk):
            n = len(e["ids"])
            obs[b, :n, :] = np.eye(L.V, dtype=np.float32)[e["ids"]]
            ids[b, :n] = e["ids"]
            mask[b, :n] = e["loss_mask"]
        perm = rng.permutation(B)
        batches.append((torch.tensor(obs[perm]), torch.tensor(ids[perm]),
                        torch.tensor(mask[perm])))
    rng.shuffle(batches)
    return batches


def eval_split(model, split: dict, *, device, batch_size: int, pad_id: int,
               depth: int = 1) -> dict:
    """teacher-forced 答案字符准确率 + span 精确匹配（逐样本，数量小可接受）。"""
    model.eval()
    inv = {v: k for k, v in L.VOCAB.items()}
    n_tok = hit = n_span = span_hit = 0
    wrong_examples: list[dict] = []
    with torch.no_grad():
        for obs, ids, mask in make_batches(split, batch_size=batch_size, seed=0,
                                           pad_id=pad_id):
            obs, ids, mask = obs.to(device), ids.to(device), mask.to(device)
            out = model.forward_sequence(obs, ids, depth=depth)
            logits = out["action_logits"][:, :-1]          # 预测 ids[:,1:]
            targets = ids[:, 1:]
            m = mask[:, :-1]
            pred = logits.argmax(-1)
            hit += int(((pred == targets).float() * m).sum())
            n_tok += int(m.sum())
    # span 级评测需要每个样本自己的 answer_spans，按样本逐个跑
    for e in split["episodes"]:
        n = len(e["ids"])
        obs = torch.zeros(1, n, L.V)
        obs[0, torch.arange(n), torch.tensor(e["ids"])] = 1.0
        ids_t = torch.tensor(e["ids"])[None]
        with torch.no_grad():
            out = model.forward_sequence(obs.to(device), ids_t.to(device), depth=depth)
        pred = out["action_logits"][0, :-1].argmax(-1).tolist()
        for (a, b), ans in zip(e["answer_spans"], e["answers"]):
            # span [a,b) 的字符由位置 a-1..b-2 预测（位置 i 预测 i+1）
            pred_chars = [inv.get(pred[i], "?") for i in range(a - 1, b - 1)]
            got = "".join(pred_chars)
            n_span += 1
            if got == ans:
                span_hit += 1
            elif len(wrong_examples) < 10:
                wrong_examples.append({"expected": ans, "got": got,
                                       "context": e["prompt"][max(0, a - 40):a]})
    return {"answer_char_acc": round(hit / max(1, n_tok), 4),
            "n_answer_chars": int(n_tok),
            "span_exact": round(span_hit / max(1, n_span), 4),
            "n_spans": n_span,
            "wrong_examples": wrong_examples}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.003)
    ap.add_argument("--seed", type=int, default=3001)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    pad_id = L.VOCAB[L.PAD]
    torch.manual_seed(args.seed)
    np.random.seed(args.seed % 2**32)

    # 数据（train/dev/holdout 同分布不同种子；三组 OOD 各自不同策略）
    train = L.make_split("train", n=400, seed=args.seed)
    dev = L.make_split("dev", n=64, seed=args.seed + 1)
    holdout = L.make_split("holdout", n=64, seed=990001)          # 封存：报告前只评一次
    ood_name = L.make_split("OOD-name", n=48, seed=990002, names=L.OOD_NAMES)
    ood_comp = L.make_split("OOD-compose", n=48, seed=990003, force_it=True,
                            ood_combo_pool=L.OOD_COMBOS)
    ood_len = L.make_split("OOD-length", n=48, seed=990004,
                           chain_steps=L.MAX_TRAIN_CHAIN + 2)
    max_T = max(len(e["ids"]) for e in train["episodes"])
    print(f"V={L.V} max_T={max_T} train={len(train['episodes'])}", flush=True)

    # 模型：n_actions=V → 决策头即字符 LM 头；数值输入维度 = V（one-hot）。
    # 直接构造 NModel（build_model 硬编码 n_actions=4，M3 需要 V；核心代码零改动）。
    cfg = default_config()
    cfg = cfg.model_copy(update={
        "model": cfg.model.model_copy(update={
            "event_numeric_dim": L.V, "event_text_dim": 8}),
        "learning": cfg.learning.model_copy(update={
            "batch_size": args.batch_size, "lr": args.lr, "traj_len": max_T}),
    })
    torch.manual_seed(args.seed)
    np.random.seed(args.seed % 2**32)
    model = NModel(cfg.model, n_actions=L.V, mem_dim=16, execution="sparse",
                   use_router=True, router_ste=True).to("cpu")
    n_params = model.n_params()
    print(f"params total={n_params['total']:,}", flush=True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=1e-4)
    rng = np.random.default_rng(args.seed + 500)
    batches = make_batches(train, batch_size=args.batch_size, seed=args.seed + 7,
                           pad_id=pad_id)
    curve: list[float] = []
    t0 = time.perf_counter()
    step = 0
    done = False
    for epoch in range(200):
        if done:
            break
        model.train()
        for obs, ids, mask in batches:
            if step >= args.steps:
                done = True
                break
            out = model.forward_sequence(obs, ids, depth=1)
            logits = out["action_logits"][:, :-1]
            targets = ids[:, 1:]
            m = mask[:, :-1]
            ce = (F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                  targets.reshape(-1), reduction="none")
                  .reshape(m.shape) * m).sum() / m.sum().clamp_min(1.0)
            pred = prediction_loss(out["z_hat"], out["z"], detach_target=True)
            bal = balance_loss(out["route_probs"], n_blocks=out["route_probs"].shape[-1])
            ent = entropy_loss(out["route_probs"])
            total = ce + 1.0 * pred + 0.02 * bal + 0.002 * ent
            if not torch.isfinite(total):
                opt.zero_grad(set_to_none=True)
                step += 1
                continue
            opt.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            curve.append(float(ce.detach()))
            step += 1
            if step % 100 == 0:
                print(f"[step {step}] answer_ce={curve[-1]:.4f} "
                      f"({time.perf_counter() - t0:.0f}s)", flush=True)

    dev_m = eval_split(model, dev, device="cpu", batch_size=8, pad_id=pad_id)
    hold_m = eval_split(model, holdout, device="cpu", batch_size=8, pad_id=pad_id)
    oodn_m = eval_split(model, ood_name, device="cpu", batch_size=8, pad_id=pad_id)
    oodc_m = eval_split(model, ood_comp, device="cpu", batch_size=8, pad_id=pad_id)
    oodl_m = eval_split(model, ood_len, device="cpu", batch_size=8, pad_id=pad_id)
    depth_cmp = {}
    for d in (1, 2, 4):
        depth_cmp[f"depth{d}"] = eval_split(model, holdout, device="cpu",
                                            batch_size=8, pad_id=pad_id, depth=d)
        depth_cmp[f"depth{d}"] = {k: v for k, v in depth_cmp[f"depth{d}"].items()
                                  if k != "wrong_examples"}

    from nmind.learning.checkpoint import save_checkpoint
    save_checkpoint(model, cfg=cfg, path=CKPT, probe=None,
                    meta={"kind": "m3_v1", "seed": args.seed,
                          "steps": args.steps, "vocab": L.V})

    payload = {
        "m3_version": "v1", "seed": args.seed, "steps": args.steps,
        "wall_s": round(time.perf_counter() - t0, 1),
        "params": n_params, "vocab_size": L.V, "max_T": max_T,
        "loss_config": "CE(answer chars) + pred(detach_target) + 0.02*balance + 0.002*entropy + router_ste",
        "honesty": "teacher-forced 评测；自由生成属 M3.2。M3 v1 不得宣称语言理解能力。",
        "curve_answer_ce_head": [round(x, 4) for x in curve[:5]],
        "curve_answer_ce_tail": [round(x, 4) for x in curve[-5:]],
        "dev": dev_m, "holdout": hold_m,
        "ood_name": oodn_m, "ood_compose": oodc_m, "ood_length": oodl_m,
        "depth_compare_holdout": depth_cmp,
        "checkpoint": str(CKPT),
        "manifest_hashes": {"train": train["manifest_hash"],
                            "holdout": holdout["manifest_hash"],
                            "ood_name": ood_name["manifest_hash"],
                            "ood_compose": ood_comp["manifest_hash"],
                            "ood_length": ood_len["manifest_hash"]},
    }
    dest = OUT / "m3_v1_results.json"
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"dev char_acc={dev_m['answer_char_acc']} span={dev_m['span_exact']} | "
          f"holdout char_acc={hold_m['answer_char_acc']} span={hold_m['span_exact']}", flush=True)
    print(f"OOD name/comp/len span: {oodn_m['span_exact']}/{oodc_m['span_exact']}/{oodl_m['span_exact']}",
          flush=True)
    print(f"depth1/2/4 span: {depth_cmp['depth1']['span_exact']}/"
          f"{depth_cmp['depth2']['span_exact']}/{depth_cmp['depth4']['span_exact']}", flush=True)
    print(f"→ {dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
