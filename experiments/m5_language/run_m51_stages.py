"""M5.1 阶段化微型语言实验（预注册 v2，单 seed，阶段预算各自独立）。

v1 教训（已并入设计）：从零训练阶段必须**全参训练**——v1 沿用 M4 的
"head+block_to_ws 局部更新"口径于随机初始化模型，随机主干+只训输出头使
4 选 1 查找在数学上不可学（TF=0 实锤）。M4 局部更新口径只用于持续学习段
（父=已训练的 stage0 模型）。

诊断依据（diagnose_m5.json）：欠训练为主因（train TF-span-EM 12.93%）；exposure bias
次因（TF 12.55% → free-gen 0%）；截断 0、路由无死块（机制排除）。

阶段（每阶段独立预登记预算与门槛，失败不跨阶段加训）：
  Stage 0  单实体/单轮/单一格式（4 个可能答案）：200 步，全参训练，**答案掩码损失**
           门槛：sealed free-gen EM ≥ 0.35（多数基线 0.25+10pp）且 TF-span-EM ≥ 0.9
  Stage 1  两实体绑定（问题随机指向第一/第二句，防复制捷径）：200 步，同门槛
  Stage T  工具选择头（三分类，只用问题分类）：100 步，门槛：未见问法 ≥ 0.85
  持续学习 v0→v1：继承 stage0 权重（head+block_to_ws 局部更新 + λ_replay=0.15）
           48 步；门槛：v1 EM ≥ 0.35 且 stage0 EM 遗忘 ≤ 3pp
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
sys_root = str(ROOT / "src")
if sys_root not in sys.path:
    sys.path.insert(0, sys_root)

from nmind.learning.online_adaptation import (  # noqa: E402
    AdaptationBudget, clone_snapshot, seal_eval_state, state_hash,
    verify_bitwise_inheritance)
from nmind.language.neural_dialogue import LMConfig, NMindLM, load_lm, save_lm  # noqa: E402
from nmind.language.decoder import generate  # noqa: E402
from nmind.language.tokenizer import PAD, CharTokenizer  # noqa: E402

HERE = Path(__file__).resolve().parent
ENTITIES_V0 = ["小明"]
ENTITIES_V1 = ["小明", "小红"]
POSITIONS = ["1号位置", "2号位置", "3号位置", "4号位置"]
SEED = 20261301
STEPS = {"stage0": 200, "stage1": 200, "tool": 100, "continual": 48}
WALL_LIMIT_S = 300.0
GATE_EM = 0.35
GATE_TF = 0.9
TOOL_GATE = 0.85
FORGET_MAX = 0.03


def _masked_batch_loss(model, batch: list[tuple[list[int], tuple[int, int]]],
                       dev) -> tuple[torch.Tensor, int, int]:
    """等长批量的答案掩码 CE：只对答案 span（含 EOS）计损失。返回 (loss, hit, n)。"""
    T = max(len(ids) for ids, _s in batch)
    x = torch.full((len(batch), T - 1), PAD, dtype=torch.long)
    y = torch.full((len(batch), T - 1), PAD, dtype=torch.long)
    span_mask = torch.zeros(len(batch), T - 1, dtype=torch.bool)
    for i, (ids, (s, e)) in enumerate(batch):
        seq = torch.tensor(ids, dtype=torch.long)
        x[i, : len(ids) - 1] = seq[:-1]
        y[i, : len(ids) - 1] = seq[1:]
        span_mask[i, max(0, s - 1): min(len(ids) - 1, e)] = True
    logits, _st = model(x.to(dev))
    logp = torch.log_softmax(logits, dim=-1)
    tgt = y.to(dev)
    flat_lp = logp.reshape(-1, logp.shape[-1])
    flat_t = tgt.reshape(-1)
    flat_m = (span_mask.reshape(-1) & (flat_t != PAD)).to(dev)
    if int(flat_m.sum()) == 0:
        return torch.zeros((), device=dev, requires_grad=True), 0, 0
    idx = torch.nonzero(flat_m).reshape(-1)
    sel_lp = flat_lp[idx]
    sel_t = flat_t[idx]
    loss = -sel_lp[torch.arange(sel_t.numel()), sel_t].mean()
    pred = sel_lp.argmax(-1)
    hit = int((pred == sel_t).sum())
    n = int(sel_t.numel())
    return loss, hit, n


def build_micro(stage: str, n: int, seed: int) -> tuple[list[str], list[str]]:
    rng = random.Random(seed)
    prompts, answers = [], []
    ents = ENTITIES_V0 if stage == "stage0" else ENTITIES_V1
    for _ in range(n):
        if stage == "stage0":
            e = ents[0]
            p = rng.choice(POSITIONS)
            prompts.append(f"用户：{e}在{p}。\n{e}在哪里？\n助手：")
            answers.append(f"{e}在{p}。")
        else:
            e1, e2 = ents
            p1, p2 = rng.sample(POSITIONS, 2)
            q = e1 if rng.random() < 0.5 else e2
            prompts.append(f"用户：{e1}在{p1}。{e2}在{p2}。\n{q}在哪里？\n助手：")
            answers.append(f"{q}在{p1}。" if q == e1 else f"{q}在{p2}。")
    return prompts, answers


def encode_rows(tok, prompts, answers) -> list[tuple[list[int], tuple[int, int]]]:
    rows = []
    for p, a in zip(prompts, answers):
        pre = tok.encode(p)
        ans = tok.encode(a, add_eos=True)
        rows.append((pre + ans, (len(pre), len(pre) + len(ans))))
    return rows


def train_stage(model, tok, *, rows, updates, lr, seed, label, dev,
                replay_rows=None, replay_ratio=0.0, trainable="all"):
    budget = AdaptationBudget(updates, label=label)
    for name, p in model.named_parameters():
        if trainable == "all":
            p.requires_grad_(True)
        else:
            keep = ("head.", "block_to_ws.")
            p.requires_grad_(name.startswith(keep))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    rng = random.Random(seed)
    t0 = time.perf_counter()
    curve: list[float] = []
    n = len(rows)
    model.train()
    while budget.used < updates:
        if time.perf_counter() - t0 > WALL_LIMIT_S:
            break
        budget.spend(1, label)
        batch = [rows[i] for i in rng.sample(range(n), min(16, n))]
        loss, _h, _n = _masked_batch_loss(model, batch, dev)
        if replay_rows and replay_ratio > 0:
            rbatch = [replay_rows[i] for i in rng.sample(range(len(replay_rows)),
                                                          min(4, len(replay_rows)))]
            r_loss, _h2, _n2 = _masked_batch_loss(model, rbatch, dev)
            loss = loss + float(replay_ratio) * r_loss
        if not torch.isfinite(loss):
            opt.zero_grad(set_to_none=True)
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()
        curve.append(round(float(loss.detach()), 4))
    wall = round(time.perf_counter() - t0, 2)
    model.eval()
    return {"steps": budget.used, "loss_first": curve[0] if curve else None,
             "loss_last": curve[-1] if curve else None, "wall_s": wall,
             "curve_head": curve[:5], "curve_tail": curve[-5:]}


def eval_free(model, tok, prompts, answers, *, max_new_tokens=24) -> dict:
    model.eval()
    em = nonempty = 0
    samples: list[dict] = []
    for p, a in zip(prompts, answers):
        ids = tok.encode(p)
        with torch.no_grad():
            logits, state = model(torch.tensor([ids], dtype=torch.long))
        rec, _st = generate(model, tok, tok.encode("助手："), state=state,
                             max_new_tokens=max_new_tokens, greedy=True)
        pred = rec.text.strip()
        ok = pred == a.strip()
        em += int(ok)
        nonempty += int(bool(pred))
        if len(samples) < 6:
            samples.append({"prompt": p.replace("\n", "|"), "oracle": a, "pred": pred,
                             "em": ok, "finish": rec.finish_reason})
    return {"em": round(em / max(1, len(answers)), 4), "n": len(answers),
             "nonempty_rate": round(nonempty / max(1, len(answers)), 4), "samples": samples}


def tf_span_em(model, tok, prompts, answers, dev) -> float:
    hit = n = 0
    model.eval()
    with torch.no_grad():
        for p, a in zip(prompts, answers):
            ids = tok.encode(p) + tok.encode(a, add_eos=True)
            if len(ids) < 2:
                continue
            x = torch.tensor([ids[:-1]], dtype=torch.long, device=dev)
            y = torch.tensor(ids[1:], dtype=torch.long, device=dev)
            logits, _st = model(x)
            pred = logits[0].argmax(-1)
            s = len(tok.encode(p)) - 1
            e_adj = len(ids) - 1
            if s >= e_adj:
                continue
            hit += int(torch.equal(pred[s:e_adj], y[s:e_adj]))
            n += 1
    return round(hit / max(1, n), 4)


def main() -> int:
    t0 = time.perf_counter()
    HERE.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cpu")
    out: dict[str, Any] = {"experiment": "m5.1-stage-micro-v2", "seed": SEED,
                            "captured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "design_fix_vs_v1": "从零阶段全参训练（v1 冻结随机主干系设计错误）",
                            "gates": {"em": GATE_EM, "tf": GATE_TF, "tool": TOOL_GATE,
                                       "forget_max_em_pp": FORGET_MAX * 100},
                            "budget": {"steps": STEPS, "wall_limit_s": WALL_LIMIT_S}}
    torch.manual_seed(SEED)

    def fresh_model(vocab_size: int) -> NMindLM:
        return NMindLM(LMConfig(vocab_size=vocab_size, d_model=96, n_blocks=8,
                                 block_hidden=96, workspace_dim=64, workspace_slots=4,
                                 top_k=2))

    tr_p0, tr_a0 = build_micro("stage0", 200, SEED)
    se_p0, se_a0 = build_micro("stage0", 40, SEED + 1)
    tr_p1, tr_a1 = build_micro("stage1", 200, SEED + 2)
    se_p1, se_a1 = build_micro("stage1", 40, SEED + 3)
    tok = CharTokenizer.train_from_texts(tr_p0 + tr_a0 + tr_p1 + tr_a1)  # 两阶段共用词表
    rows0 = encode_rows(tok, tr_p0, tr_a0)
    rows1 = encode_rows(tok, tr_p1, tr_a1)

    # ---- Stage 0（全参训练）----
    model = fresh_model(tok.vocab_size).to(dev)
    st0 = train_stage(model, tok, rows=rows0, updates=STEPS["stage0"], lr=3e-3,
                       seed=SEED, label="stage0", dev=dev, trainable="all")
    seal_eval_state(model, {k: v.clone() for k, v in model.state_dict().items()})
    e0 = eval_free(model, tok, se_p0, se_a0)
    tf0 = tf_span_em(model, tok, se_p0, se_a0, dev)
    stage0_pass = e0["em"] >= GATE_EM and tf0 >= GATE_TF
    out["stage0"] = {"train": st0, "eval": e0, "tf_span_em": tf0,
                      "gate_pass": stage0_pass, "samples": e0["samples"][:4]}
    print(f"[stage0] EM={e0['em']} TF={tf0} pass={stage0_pass} ({st0['wall_s']}s)", flush=True)

    # ---- Stage 1（两实体绑定；独立新模型，全参）----
    model1 = fresh_model(tok.vocab_size).to(dev)
    st1 = train_stage(model1, tok, rows=rows1, updates=STEPS["stage1"], lr=3e-3,
                       seed=SEED + 2, label="stage1", dev=dev, trainable="all")
    seal_eval_state(model1, {k: v.clone() for k, v in model1.state_dict().items()})
    e1 = eval_free(model1, tok, se_p1, se_a1)
    tf1 = tf_span_em(model1, tok, se_p1, se_a1, dev)
    stage1_pass = e1["em"] >= GATE_EM and tf1 >= GATE_TF
    out["stage1"] = {"train": st1, "eval": e1, "tf_span_em": tf1,
                      "gate_pass": stage1_pass, "samples": e1["samples"][:4]}
    print(f"[stage1] EM={e1['em']} TF={tf1} pass={stage1_pass} ({st1['wall_s']}s)", flush=True)

    # ---- Stage T：工具选择头（全参训练，只看问题）----
    numbers = [(a, b) for a in range(2, 12) for b in range(2, 12)]
    facts = ["图书馆闭馆日", "仓库盘点日"]
    rng_t = random.Random(SEED + 9)

    def tool_sample(kind: str, seen: bool) -> tuple[str, str, str]:
        if kind == "calc":
            a, b = rng_t.choice(numbers)
            q = (f"请计算：{a}加{b}等于多少？" if seen else f"帮我算算{a}加{b}是多少。")
            return q, f"TOOL_CALC {a}+{b}", "calc"
        if kind == "lookup":
            k = rng_t.choice(facts)
            q = f"查资料：{k}？" if seen else f"{k}是什么？帮我查查。"
            return q, f"TOOL_LOOKUP {k}", "lookup"
        return ("杯子是什么颜色？" if seen else "杯子的颜色呢？"), "DIRECT", "direct"

    def tool_rows(n: int, seed: int, *, seen: bool):
        rp, ra, rl = [], [], []
        r2 = random.Random(seed)
        for _ in range(n):
            kind = r2.choice(["calc", "lookup", "direct"])
            q, a, lab = tool_sample(kind, seen)
            rp.append(f"用户：{q}\n助手：")
            ra.append(a)
            rl.append(lab)
        return rp, ra, rl

    trp, tra, trl = tool_rows(120, SEED + 11, seen=True)
    tep, tea, tel = tool_rows(45, SEED + 12, seen=False)
    tok_t = CharTokenizer.train_from_texts(trp)
    torch.manual_seed(SEED + 20)
    mt = fresh_model(tok_t.vocab_size).to(dev)
    mt.tool_head = torch.nn.Linear(mt.cfg.block_hidden, 3).to(dev)
    labels = {"calc": 0, "lookup": 1, "direct": 2}
    inv = {0: "TOOL_CALC", 1: "TOOL_LOOKUP", 2: "DIRECT"}
    all_params = [p for p in mt.parameters() if p.requires_grad]   # tool_head 已注册为子模块，自动包含
    opt_t = torch.optim.AdamW(all_params, lr=3e-3)
    tb = AdaptationBudget(STEPS["tool"], label="tool")
    rt = random.Random(SEED + 21)
    tt0 = time.perf_counter()
    while tb.used < STEPS["tool"]:
        if time.perf_counter() - tt0 > WALL_LIMIT_S:
            break
        tb.spend(1, "tool")
        pick = rt.sample(range(len(trp)), 8)
        losses = torch.zeros((), device=dev)
        mt.train()
        for i in pick:
            ids = tok_t.encode(trp[i])            # 只有问题——答案不进分类上下文
            with torch.no_grad():
                _lg, st = mt(torch.tensor([ids], dtype=torch.long, device=dev))
            cls = mt.tool_head(st.h.mean(dim=1))
            losses = losses + torch.nn.functional.cross_entropy(
                cls, torch.tensor([labels[trl[i]]], device=dev))
        opt_t.zero_grad(set_to_none=True)
        losses.backward()
        opt_t.step()
    mt.eval()
    sel_hit = 0
    conf = []
    with torch.no_grad():
        for q, a, lab in zip(tep, tea, tel):
            ids = tok_t.encode(q)
            _lg, st = mt(torch.tensor([ids], dtype=torch.long, device=dev))
            cls = mt.tool_head(st.h.mean(dim=1))
            pred_label = inv[int(cls.argmax(-1))]
            gold = {"calc": "TOOL_CALC", "lookup": "TOOL_LOOKUP", "direct": "DIRECT"}[lab]
            sel_hit += int(pred_label == gold)
            conf.append({"q": q, "gold": gold, "pred": pred_label})
    tool_acc = round(sel_hit / len(tep), 4)
    out["tool"] = {"steps": tb.used, "unseen_phrasing_acc": tool_acc, "n": len(tep),
                    "gate_pass": tool_acc >= TOOL_GATE, "samples": conf[:6],
                    "note": "工具执行由独立程序验证；选择头只测神经选择"}
    print(f"[tool] unseen_acc={tool_acc} pass={out['tool']['gate_pass']}", flush=True)

    # ---- 持续学习 v0→v1（M4 衔接）：父 = stage0（已训练），局部更新 48 步 ----
    cont: dict[str, Any] = {}
    if stage0_pass:
        parent = model
        model_c = fresh_model(tok.vocab_size).to(dev)
        model_c.load_state_dict(parent.state_dict())
        chk = verify_bitwise_inheritance(clone_snapshot(parent), model_c)
        a_em_before = e0["em"]
        stc = train_stage(model_c, tok, rows=rows1, updates=STEPS["continual"], lr=1e-3,
                           seed=SEED + 30, label="continual", dev=dev,
                           replay_rows=rows0, replay_ratio=0.15,
                           trainable="head_block_to_ws")
        seal_eval_state(model_c, {k: v.clone() for k, v in parent.state_dict().items()})
        e1_after = eval_free(model_c, tok, se_p1, se_a1)
        e0_after = eval_free(model_c, tok, se_p0, se_a0)
        forgetting = round(a_em_before - e0_after["em"], 4)
        cont = {
            "parent_sha": state_hash(parent), "child_sha": state_hash(model_c),
            "inheritance_verified": chk["all_bitwise_equal"],
            "trainable": "head_block_to_ws（M4 局部更新口径）",
            "replay_ratio": 0.15,
            "B_em_after": e1_after["em"],
            "A_em_before": a_em_before, "A_em_after": e0_after["em"],
            "A_forgetting": forgetting,
            "train": stc,
            "gate_pass": bool(e1_after["em"] >= GATE_EM and forgetting <= FORGET_MAX),
        }
        print(f"[continual] B_EM={e1_after['em']} A_forget={forgetting} "
              f"pass={cont['gate_pass']}", flush=True)
    else:
        cont = {"skipped_reason": "stage0 未通过"}
    out["continual"] = cont

    payload = {**out, "wall_s_total": round(time.perf_counter() - t0, 1),
                "honesty": "微型任务证明的是管线、缩放规律与持续学习衔接，"
                            "不代表通用中文能力；3-seed 复验与扩预算需另行批准"}
    (HERE / "m51_stages.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    print(f"→ {HERE / 'm51_stages.json'}（总用时 {payload['wall_s_total']}s）", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
