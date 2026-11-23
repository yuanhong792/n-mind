"""V3.7 neural-only architecture ablation.

No cloud, memory, tool-answer, or pre-trained teacher pathway. The synthetic
oracle exists only in dataset construction and *after* model inference.
All options are candidate experiments; stable checkpoint is never loaded/written.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from nmind.language.decoder import free_generation_v2
from nmind.language.neural_dialogue import LMConfig, NMindLM, save_lm
from nmind.language.tokenizer import EOS, PAD, UNK, CharTokenizer


@dataclass(frozen=True)
class Item:
    group: str
    prompt: str
    answer: str
    query_entity: str
    split: str

    def as_dict(self) -> dict[str, str]:
        return vars(self).copy()


# Each (two distinct positions, statement order, acknowledgement) is a semantic
# group. Both query variants from a group go into the SAME split.
POSITIONS = ("1号位置", "2号位置", "3号位置", "4号位置")
ACKS = ("", "好的。", "明白了。", "收到。")
NAMES = ("小明", "小红")
SPLITS = {"train": (0, 60), "dev": (60, 80), "sealed": (80, 96)}


def build_dataset(seed: int = 20261123) -> dict[str, list[Item]]:
    groups = [(a, b, order, ack)
              for a in POSITIONS for b in POSITIONS if a != b
              for order in (0, 1) for ack in ACKS]
    assert len(groups) == 96
    random.Random(seed).shuffle(groups)
    result: dict[str, list[Item]] = {k: [] for k in SPLITS}
    for split, (lo, hi) in SPLITS.items():
        for idx in range(lo, hi):
            a, b, order, ack = groups[idx]
            first, second = NAMES if order == 0 else tuple(reversed(NAMES))
            loc = {first: a, second: b}
            context = f"用户：{first}在{a}。{second}在{b}。\n{ack}"
            group = f"{first}:{a}|{second}:{b}|{ack}"
            for who in NAMES:
                q = f"{who}在哪里？"
                result[split].append(Item(
                    group, context + f"用户：{q}\n助手：",
                    f"{who}在{loc[who]}。", who, split))
    validate_splits(result)
    return result


def validate_splits(data: dict[str, list[Item]]) -> dict[str, Any]:
    group_sets = {name: {x.group for x in rows} for name, rows in data.items()}
    prompt_sets = {name: {x.prompt for x in rows} for name, rows in data.items()}
    for a in SPLITS:
        for b in SPLITS:
            if a < b:
                if group_sets[a] & group_sets[b] or prompt_sets[a] & prompt_sets[b]:
                    raise ValueError(f"semantic leakage between {a} and {b}")
    for name, rows in data.items():
        counts: dict[str, int] = {}
        for row in rows:
            counts[row.group] = counts.get(row.group, 0) + 1
        if not rows or any(n != 2 for n in counts.values()):
            raise ValueError(f"{name}: expected exactly 2 queries per context")
    return {k: {"samples": len(v), "groups": len(group_sets[k])} for k, v in data.items()}


def build_tokenizer(train: list[Item], others: list[Item]) -> CharTokenizer:
    tok = CharTokenizer.train_from_texts([x.prompt + x.answer for x in train], max_size=256)
    # Frozen training-only tokenizer. Report unsupported OOV instead of cheating
    # by rebuilding the vocabulary from the dev/sealed sets.
    for x in train + others:
        if UNK in tok.encode(x.prompt + x.answer):
            raise ValueError("frozen training vocabulary has UNK; cannot fairly score")
    return tok


def manifest(data: dict[str, list[Item]], tok: CharTokenizer, seed: int) -> dict[str, Any]:
    def fp(rows: list[Item]) -> str:
        payload = json.dumps([x.as_dict() for x in rows], ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return {
        "seed": seed, "generator": "v37-two-entity-context-group-v1",
        "semantic_split": validate_splits(data),
        "split_hashes": {k: fp(v) for k, v in data.items()},
        "vocab_sha": tok.vocab_hash(), "vocab_size": tok.vocab_size,
        "sealed_policy": "withheld unless --final-sealed; never used for training",
    }


ARMS: dict[str, dict[str, Any]] = {
    "legacy": {},
    "router_ste": {"use_router_ste": True},
    "attention": {"readout_mode": "query_attention"},
    "router_attention": {"use_router_ste": True, "readout_mode": "query_attention"},
    "reset": {"use_reset_gate": True},
    "all": {"use_reset_gate": True, "use_router_ste": True,
            "readout_mode": "query_attention"},
}


def make_model(vocab_size: int, arm: str, seed: int, *,
               d_model: int = 16, block_hidden: int = 16) -> tuple[NMindLM, int]:
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    base = LMConfig(vocab_size=vocab_size, d_model=d_model, n_blocks=4,
                    block_hidden=block_hidden, workspace_dim=8,
                    workspace_slots=2, top_k=2)
    torch.manual_seed(seed)
    legacy = NMindLM(base)
    torch.manual_seed(seed)
    cfg = LMConfig(**{**base.to_dict(), **ARMS[arm]})
    candidate = NMindLM(cfg)
    # Identical initialization for every parameter shared with Legacy;
    # new parameters in reset/attention arms stay independently initialized.
    shared = legacy.state_dict()
    candidate.load_state_dict(
        {k: shared.get(k, v) for k, v in candidate.state_dict().items()},
        strict=True,
    )
    candidate.use_router_load_update = False  # no buffer adaptation confound
    return candidate, legacy.n_params()


def state_hash(model: NMindLM) -> str:
    h = hashlib.sha256()
    for name, v in sorted(model.state_dict().items()):
        h.update(name.encode("utf-8"))
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def encode_batch(rows: list[Item], tok: CharTokenizer) -> tuple[torch.Tensor, torch.Tensor, int]:
    seqs: list[list[int]] = []
    masks: list[list[int]] = []
    n_supervised = 0
    for item in rows:
        prompt_ids = tok.encode(item.prompt)
        all_ids = prompt_ids + tok.encode(item.answer) + [EOS]
        if UNK in all_ids:
            raise ValueError("unexpected OOV after tokenizer freeze")
        if len(prompt_ids) == 0:
            raise ValueError("empty prompt")
        x = all_ids[:-1]
        # Supervise answer+EOS ONLY; never train against prompt/hidden Oracle answer
        # before the answer span.
        y = [-100] * (len(prompt_ids) - 1) + all_ids[len(prompt_ids):]
        if len(x) != len(y):
            raise AssertionError("answer-span loss misalignment")
        n_supervised += sum(t != -100 for t in y)
        seqs.append(x)
        masks.append(y)
    longest = max(map(len, seqs))
    xpad = torch.full((len(rows), longest), PAD, dtype=torch.long)
    ypad = torch.full((len(rows), longest), -100, dtype=torch.long)
    for i, (x, y) in enumerate(zip(seqs, masks)):
        xpad[i, :len(x)] = torch.tensor(x)
        ypad[i, :len(y)] = torch.tensor(y)
    return xpad, ypad, n_supervised


def gradient_norms(model: NMindLM) -> dict[str, float]:
    out: dict[str, float] = {}
    for name, prefix in (("encoder", "tok_emb."), ("ncell", "blocks."),
                         ("router", "router.score_mlp."),
                         ("workspace", "workspace."),
                         ("readout", "readout_"), ("head", "head.")):
        sq = sum(float(p.grad.detach().square().sum())
                 for n, p in model.named_parameters()
                 if n.startswith(prefix) and p.grad is not None)
        out[name] = round(math.sqrt(sq), 8)
    return out


def train_candidate(model: NMindLM, rows: list[Item], tok: CharTokenizer,
                    *, steps: int, batch_size: int, seed: int,
                    max_seconds: float = 300.0, lr: float = 0.003,
                    max_steps: int = 128) -> dict[str, Any]:
    if steps < 0 or steps > max_steps or not 1 <= batch_size <= len(rows):
        raise ValueError(f"invalid candidate training budget (0..{max_steps} steps)")
    if max_seconds <= 0:
        raise ValueError("invalid wall-time budget")
    before = state_hash(model)
    rng = random.Random(seed + 1001)
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    curve: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    non_finite = 0
    executed = 0
    model.train()
    for _ in range(steps):
        if time.perf_counter() - t0 > max_seconds:
            break
        batch = rng.sample(rows, batch_size)
        inputs, targets, supervised = encode_batch(batch, tok)
        device = next(model.parameters()).device
        inputs, targets = inputs.to(device), targets.to(device)
        model.zero_grad(set_to_none=True)
        logits, _ = model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, tok.vocab_size),
                               targets.reshape(-1), ignore_index=-100)
        if not torch.isfinite(loss):
            non_finite += 1
            continue
        loss.backward()
        grads = gradient_norms(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optim.step()
        executed += 1
        curve.append({"step": executed, "loss_answer_eos": float(loss.detach()),
                      "answer_eos_tokens": supervised, "grad": grads})
    model.eval()
    after = state_hash(model)
    return {"requested_steps": steps, "optimizer_steps_executed": executed,
            "non_finite_steps": non_finite, "stop_reason": (
                "WALL_BUDGET" if executed < steps and time.perf_counter() - t0 > max_seconds
                else "COMPLETED" if executed == steps else "INCOMPLETE"),
            "wall_seconds": round(time.perf_counter() - t0, 3),
            "state_sha_before": before, "state_sha_after": after,
            "state_changed": before != after, "curve": curve}


def diagnostics(model: NMindLM, tok: CharTokenizer, item: Item) -> dict[str, Any]:
    model.eval()
    device = next(model.parameters()).device
    x = torch.tensor([tok.encode(item.prompt)], dtype=torch.long, device=device)
    with torch.no_grad():
        logits, state = model(x)
        z = model.tok_emb(x[:, 0])
        ws = model.workspace.initial_state(1, x.device)
        wr = model.workspace.read(ws, z)
        err = torch.full((1, 1), 0.5, device=device)
        route = model.router.route(z, model.initial_state(1, x.device).h,
                                   wr.value, err, err, k=model.top_k)
    return {
        "first_token_route_indices": route.indices[0].tolist(),
        "first_token_route_entropy": route.stats["entropy"],
        "first_token_route_probs": [round(float(v), 6) for v in route.probs[0]],
        "final_block_l2": [round(float(v), 6) for v in
                           state.h[0].square().sum(-1).sqrt()],
        "workspace_occupied": [round(float(v), 6) for v in state.ws["occupied"][0]],
        "workspace_utility": [round(float(v), 6) for v in state.ws["utility"][0]],
        "final_logit_range": [round(float(logits[0, -1].min()), 6),
                              round(float(logits[0, -1].max()), 6)],
        "scope": "diagnostic snapshot, not a claim of learned semantics",
    }


def evaluate(model: NMindLM, rows: list[Item], tok: CharTokenizer,
             *, limit: int = 8, max_new_tokens: int = 16) -> dict[str, Any]:
    model.eval()
    selected = rows[:max(0, limit)]
    records: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for item in selected:
            rec = free_generation_v2(model, tok, item.prompt,
                                     max_new_tokens=max_new_tokens)
            # CharTokenizer always returns Unicode. U+FFFD represents a generated
            # UNK; byte-level MINLEARN2 invalid UTF8 is NOT measured by this stack.
            valid = "\ufffd" not in rec.text
            records.append({
                "group": item.group, "query_entity": item.query_entity,
                "expected": item.answer, "actual": rec.text,
                "full_answer_exact": rec.text == item.answer,
                "contains_unk_replacement": not valid,
                "unicode_encodable": True,
                "eos_after_exact_answer": (
                    rec.finish_reason == "eos" and rec.text == item.answer),
                "finish_reason": rec.finish_reason,
                "token_ids": rec.token_ids,
                "sample_source": "neural_only_no_teacher_memory_or_tool",
            })
    correct = sum(x["full_answer_exact"] for x in records)
    groups: dict[str, list[dict[str, Any]]] = {}
    for x in records:
        groups.setdefault(x["group"], []).append(x)
    pairs = [pair for pair in groups.values() if len(pair) == 2]
    both_correct = sum(all(x["full_answer_exact"] for x in pair) for pair in pairs)
    changed = sum(pair[0]["actual"] != pair[1]["actual"] for pair in pairs)
    return {
        "evaluated": len(records), "full_answer_exact_count": correct,
        "full_answer_exact_rate": correct / len(records) if records else None,
        "paired_contexts": len(pairs),
        "pair_both_correct": both_correct,
        "pair_both_correct_rate": both_correct / len(pairs) if pairs else None,
        "pair_prediction_changed": changed,
        "unk_replacement_count": sum(x["contains_unk_replacement"] for x in records),
        "eos_after_exact_count": sum(x["eos_after_exact_answer"] for x in records),
        "char_unicode_status": "unicode_native_not_byte_utf8_validation",
        "time_seconds": round(time.perf_counter() - t0, 3),
        "records": records,
    }


def run_experiment(*, seed: int = 20261123, steps: int = 8,
                   batch_size: int = 2, eval_samples: int = 8,
                   arms: tuple[str, ...] = tuple(ARMS),
                   final_sealed: bool = False,
                   d_model: int = 16, block_hidden: int = 16,
                   wall_seconds_per_arm: float = 300.0,
                   device: str = "cpu", max_steps_per_arm: int = 128,
                   checkpoint_dir: str | Path | None = None) -> dict[str, Any]:
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; do not silently fall back to CPU")
    if not arms or any(a not in ARMS for a in arms):
        raise ValueError("arms must be a nonempty subset of registered ablations")
    if steps > max_steps_per_arm or steps < 0:
        raise ValueError(f"steps must stay within controlled 0..{max_steps_per_arm} budget")
    if eval_samples < 1:
        raise ValueError("eval_samples must be >= 1")
    data = build_dataset(seed)
    tok = build_tokenizer(data["train"], data["dev"] + data["sealed"])
    m = manifest(data, tok, seed)
    candidate_root = Path(checkpoint_dir) if checkpoint_dir is not None else None
    if candidate_root is not None:
        candidate_root.mkdir(parents=True, exist_ok=True)
        vocab_path = candidate_root / "tokenizer.json"
        if vocab_path.exists():
            existing = CharTokenizer.load(vocab_path)
            if existing.vocab_hash() != tok.vocab_hash():
                raise ValueError("existing tokenizer in candidate directory differs")
        else:
            tok.save(vocab_path)
    out: dict[str, Any] = {
        "experiment": "ncell-v37-structural-ablation",
        "interpretation": "exploratory_not_verified_generalization",
        "training_data_source": "deterministic_synthetic_oracle",
        "teacher_calls": 0, "memory_reads": 0, "tool_calls": 0,
        "stable_checkpoint_written": False, "sealed_used_for_training": False,
        "sealed_eval_performed": final_sealed, "manifest": m,
        "config": {"seed": seed, "steps_per_arm": steps,
                   "batch_size": batch_size, "eval_samples": eval_samples,
                   "arms": list(arms), "d_model": d_model,
                   "block_hidden": block_hidden, "device": device,
                   "wall_seconds_per_arm": wall_seconds_per_arm},
        "arms": {},
    }
    for arm in arms:
        model, legacy_params = make_model(tok.vocab_size, arm, seed,
                                          d_model=d_model, block_hidden=block_hidden)
        model = model.to(device)
        before = evaluate(model, data["dev"], tok, limit=eval_samples)
        diag_before = diagnostics(model, tok, data["dev"][0])
        train = train_candidate(model, data["train"], tok, steps=steps,
                                batch_size=batch_size, seed=seed,
                                max_seconds=wall_seconds_per_arm,
                                max_steps=max_steps_per_arm)
        dev = evaluate(model, data["dev"], tok, limit=eval_samples)
        diag_after = diagnostics(model, tok, data["dev"][0])
        sealed = (evaluate(model, data["sealed"], tok, limit=eval_samples)
                  if final_sealed else {"status": "NOT_RUN", "reason": "sealed withheld"})
        ckpt = None
        if candidate_root is not None and train["optimizer_steps_executed"] > 0:
            # No Stable paths are accepted or opened; only local opt-in candidates.
            unique = train["state_sha_after"][:12]
            ckpt_path = candidate_root / (
                f"candidate_{arm}_seed{seed}_steps{train['optimizer_steps_executed']}_{unique}.pt")
            if ckpt_path.exists():
                raise FileExistsError(f"candidate already exists: {ckpt_path}")
            save_lm(ckpt_path, model=model, tokenizer=tok, cfg=model.cfg,
                    steps=train["optimizer_steps_executed"], seed=seed,
                    parent_sha="random_init_v37", manifest_hash=m["split_hashes"]["train"])
            ckpt = str(ckpt_path)
        out["arms"][arm] = {
            "candidate_checkpoint": ckpt,
            "architecture": model.cfg.to_dict(),
            "parameters": model.n_params(), "legacy_parameters": legacy_params,
            "parameter_delta": model.n_params() - legacy_params,
            "before_dev": before, "training": train,
            "after_dev": dev, "sealed": sealed,
            "diagnostics_before": diag_before, "diagnostics_after": diag_after,
        }
    return out


def markdown_report(report: dict[str, Any]) -> str:
    cfg = report["config"]
    lines = [
        "# N-Cell V3.7 — Structural Ablation Results",
        "",
        f"Seed: `{cfg['seed']}` | Budget per arm: `{cfg['steps_per_arm']}` optimizer steps "
        f"| Sealed opened: `{report['sealed_eval_performed']}`",
        "",
        "These are neural-only char-tokenizer experiments. Loss reduction, nonzero "
        "gradients, and short dev gains do **not** establish language understanding.",
        "No teacher/Memory/Tool answers are used. Previous MINLEARN-2 byte-level "
        "UTF-8 failures are separate and remain unresolved by this experiment.",
        "",
        "| Arm | Params | Δ params | Steps | Dev exact before | Dev exact after | Paired exact | Last loss |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in report["arms"].items():
        b, a, train = item["before_dev"], item["after_dev"], item["training"]
        c = train["curve"]
        lines.append(
            f"| {name} | {item['parameters']} | {item['parameter_delta']:+d} "
            f"| {train['optimizer_steps_executed']} "
            f"| {b['full_answer_exact_count']}/{b['evaluated']} "
            f"| {a['full_answer_exact_count']}/{a['evaluated']} "
            f"| {a['pair_both_correct']}/{a['paired_contexts']} "
            f"| {c[-1]['loss_answer_eos']:.5f} |" if c else
            f"| {name} | {item['parameters']} | {item['parameter_delta']:+d} "
            f"| 0 | {b['full_answer_exact_count']}/{b['evaluated']} "
            f"| {a['full_answer_exact_count']}/{a['evaluated']} "
            f"| {a['pair_both_correct']}/{a['paired_contexts']} | NOT_RUN |"
        )
    lines += [
        "", "## Interpretation limits", "",
        "- Same seed and shared initial parameter tensors across arms; opt-in layers "
        "add parameters, so reset/attention effects are not parameter-matched.",
        "- Prompt/answer pairs are split by full semantic context **before** training; "
        "two queries for each context never cross splits.",
        "- All answers are generated freely without reading the target. "
        "Teacher forcing trains answer-span+EOS only.",
        "- The sealed split is never used in optimizer updates and is withheld "
        "unless explicitly opened for a one-time final evaluation.",
        "- Router gradient depends on STE for selected blocks only; task success "
        "requires separate independent evidence.",
        "- No Stable promotion is permitted by this script.",
        "", "## Provenance",
        "", "```json", json.dumps(report["manifest"], ensure_ascii=False, indent=2),
        "```", "",
    ]
    return "\n".join(lines)


def write_report(report: dict[str, Any], out_dir: str | Path) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    j = out / "V37_ABLATION_RESULTS.json"
    md = out / "V37_ABLATION_REPORT.md"
    j.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md.write_text(markdown_report(report), encoding="utf-8")
    return j, md
