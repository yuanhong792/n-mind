"""M5-Language 训练入口（任务书 P2）：单 seed、≤200 optimizer 更新、≤300s 墙钟、预算硬守卫。

* 数据：experiments/m5_language/data（dialogue_data.build_all 生成的不可变 manifest）；
* 词表只从 train split 文本构建并冻结（sha256 记录）；
* 模型 = NMindLM（随机初始化；torch.manual_seed 显式设置）；
* 预算：AdaptationBudget(max_updates) 硬守卫 + 墙钟护栏；到点即停并如实报告学习曲线；
* 产出：nmind-lm/1 checkpoint + tokenizer + 训练 JSON（曲线/步数/耗时/哈希）。
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from ..learning.online_adaptation import AdaptationBudget
from .dialogue_data import build_all
from .neural_dialogue import LMConfig, NMindLM, save_lm
from .tokenizer import EOS, PAD, CharTokenizer

ROOT = Path(__file__).resolve().parents[3]   # src/nmind/language/ → 仓库根


@dataclass
class LanguageTrainConfig:
    seed: int = 20261241
    max_updates: int = 200
    batch_size: int = 16
    max_len: int = 160
    lr: float = 3e-3
    d_model: int = 96
    n_blocks: int = 8
    block_hidden: int = 96
    workspace_dim: int = 64
    workspace_slots: int = 4
    top_k: int = 2
    wall_limit_s: float = 300.0
    eval_interval: int = 20
    data_dir: str = "experiments/m5_language/data"
    out_dir: str = "experiments/m5_language"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def dialogue_ids(dialogue: dict, tok: CharTokenizer, max_len: int) -> list[int]:
    ids: list[int] = []
    for t in dialogue["turns"]:
        if t["role"] == "user":
            ids += tok.encode(f"用户：{t['text']}\n")
        else:
            ids += tok.encode(f"助手：{t['text']}", add_eos=True)
    return ids[:max_len]


def run_train(cfg: LanguageTrainConfig, *, root: Path = ROOT, device: str = "cpu",
              quiet: bool = False) -> dict[str, Any]:
    t0 = time.perf_counter()
    dev = torch.device(device)
    torch.manual_seed(cfg.seed)
    data_dir = root / cfg.data_dir
    out_dir = root / cfg.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if not (data_dir / "manifest.json").exists():
        build_all(data_dir)
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))

    train_payload = json.loads((data_dir / "splits" / "train.json").read_text(encoding="utf-8"))
    dev_payload = json.loads((data_dir / "splits" / "dev.json").read_text(encoding="utf-8"))
    train_texts = ["\n".join(f"用户：{t['text']}" if t["role"] == "user" else f"助手：{t['text']}"
                              for t in d["turns"]) for d in train_payload]
    tok = CharTokenizer.train_from_texts(train_texts)
    tok.save(out_dir / "tokenizer.json")

    lm_cfg = LMConfig(vocab_size=tok.vocab_size, d_model=cfg.d_model, n_blocks=cfg.n_blocks,
                       block_hidden=cfg.block_hidden, workspace_dim=cfg.workspace_dim,
                       workspace_slots=cfg.workspace_slots, top_k=cfg.top_k)
    model = NMindLM(lm_cfg).to(dev)
    ids_list = [dialogue_ids(d, tok, cfg.max_len) for d in train_payload]
    dev_ids = [dialogue_ids(d, tok, cfg.max_len) for d in dev_payload]

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr)
    budget = AdaptationBudget(cfg.max_updates, label="m5-language-train")
    rng = random.Random(cfg.seed)
    ce = torch.nn.CrossEntropyLoss(ignore_index=PAD)

    def batch_loss(batch_ids: list[list[int]]) -> tuple[torch.Tensor, int]:
        m = max(len(x) for x in batch_ids)
        x = torch.full((len(batch_ids), m), PAD, dtype=torch.long)
        for i, seq in enumerate(batch_ids):
            x[i, :len(seq)] = torch.tensor(seq)
        inputs, targets = x[:, :-1], x[:, 1:]
        logits, _st = model(inputs)
        loss = ce(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        return loss, int((targets != PAD).sum())

    curve: list[float] = []
    dev_curve: list[dict[str, float]] = []
    non_finite = 0
    wall_stop = False
    model.train()
    while budget.used < cfg.max_updates:
        if time.perf_counter() - t0 > cfg.wall_limit_s:
            wall_stop = True
            break
        pick = rng.sample(range(len(ids_list)), min(cfg.batch_size, len(ids_list)))
        batch = [ids_list[i] for i in pick]
        if max(len(x) for x in batch) < 2:
            continue
        try:
            budget.spend(1, "train")            # 每步记账（超限即抛）
        except Exception:
            break
        loss, _n = batch_loss(batch)
        if not torch.isfinite(loss):
            non_finite += 1
            opt.zero_grad(set_to_none=True)
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
        opt.step()
        curve.append(round(float(loss.detach()), 4))
        if budget.used % cfg.eval_interval == 0 or budget.used == cfg.max_updates:
            model.eval()
            with torch.no_grad():
                dloss, dn = batch_loss(dev_ids[:32])
            model.train()
            dev_curve.append({"step": budget.used,
                               "dev_loss": round(float(dloss), 4),
                               "dev_tokens": dn})
            if not quiet:
                print(f"[step {budget.used}] train_loss={curve[-1]:.4f} dev_loss={dev_curve[-1]['dev_loss']:.4f}",
                      flush=True)
    wall = time.perf_counter() - t0
    model.eval()

    ckpt_path = save_lm(out_dir / "language_checkpoint.pt", model=model, tokenizer=tok,
                         cfg=lm_cfg, steps=budget.used, seed=cfg.seed,
                         parent_sha="random-init(from-scratch)",
                         manifest_hash=manifest.get("manifest_hash", ""))
    payload = {
        "experiment": "m5-language-train", "captured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": cfg.seed, "device": device, "torch_threads": int(torch.get_num_threads()),
        "config": cfg.to_dict(),
        "vocab_size": tok.vocab_size, "vocab_hash": tok.vocab_hash(),
        "manifest_hash": manifest.get("manifest_hash", ""),
        "params_total": model.n_params(),
        "optimizer_steps_executed": budget.used, "budget_cap": cfg.max_updates,
        "non_finite_skipped": non_finite, "wall_stop": wall_stop,
        "wall_s": round(wall, 2), "wall_limit_s": cfg.wall_limit_s,
        "train_loss_curve": curve, "dev_loss_curve": dev_curve,
        "loss_first": curve[0] if curve else None,
        "loss_last": curve[-1] if curve else None,
        "checkpoint": ckpt_path, "tokenizer": str(out_dir / "tokenizer.json"),
        "honesty": "teacher-forced 训练；自由生成能力以 evaluate-language 为准，不以此处 loss 冒充对话能力",
    }
    (out_dir / "train_language.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"→ {out_dir / 'train_language.json'}（steps={budget.used} wall={wall:.1f}s）", flush=True)
    return payload
