"""阶段 2.4 条件性阶梯 v2（2026-11-25，AutoClaw）。

相对 20261124 版（ladder_parallel.py）的修订——本次指令步骤 4 的缺口补齐：
  1. LR 不再硬编码：从 lr_tasks_20261125/summary.json 的选择结果读取，任一模型
     缺选择即拒绝进入阶梯（INCONCLUSIVE_BUDGET，不训练）；
  2. 封存测试集（组合留出的 600 条）只在每臂停止点评测一次：EM / 候选排名准确率 /
     反事实跟随配对 / 捷径基线（最近实体、首位实体、固定位置、多数位置）；
     答案段逐样本 logits 存 .pt，per-sample 记录存 .json；
  3. 每 100 步写带时间戳的进度日志 → 真实斜率证据（20261124 版终止判断无留痕）；
  4. 臂级绝对 deadline + 训练前斜率外推：预计无法在全局 60 分钟内完成即停，
     停止前仍完成封存评测并保留已完成级别的 checkpoint，分类 INCONCLUSIVE_BUDGET；
  5. 每级保存 checkpoint（续训同一权重，不重置）。
"""
from __future__ import annotations

import json
import os
import platform
import random
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from models import (GRUBaseline, CausalTransformerBaseline, DeltaRuleLinearBaseline,  # noqa: E402
                     parameter_count, within_tolerance)
from nmind.language.large_binding_task import POSITIONS  # noqa: E402
from nmind.language.tokenizer import CharTokenizer  # noqa: E402
import lr_select_parallel_v2 as lrdata  # noqa: E402  同目录：数据缓存与路径定义

CLS = {"gru": GRUBaseline, "transformer": CausalTransformerBaseline,
        "delta": DeltaRuleLinearBaseline}

OUT = ROOT / "experiments" / "phase2_baselines" / "ladder_20261125"
LR_SUMMARY = ROOT / "experiments" / "phase2_baselines" / "lr_tasks_20261125" / "summary.json"
TOK = ROOT / "experiments" / "m5_language" / "tokenizer_m52.json"
DATA_CACHE = lrdata.DATA_CACHE
MODELS = ("gru", "transformer", "delta")
SEEDS = (20270101, 20270102, 20270103)
LEVELS = (1000, 3000, 10000, 30000)
BATCH = 16
GLOBAL_CAP_S = 3600.0              # 本次批准：60 分钟总墙钟
RESERVE_S = 150.0                  # 预留给封存评测与保存
REFERENCE_PARAMS = 434553

_STMT_RE = re.compile(r"(.+?)在([1-4]号位置)。")
_POS_RE = re.compile(r"在([1-4]号位置)。$")


def env_metadata() -> dict:
    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT),
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        rev = ""
    return {"python": platform.python_version(), "torch": torch.__version__,
             "platform": platform.platform(), "torch_threads": torch.get_num_threads(),
             "pid": os.getpid(), "git_rev": rev, "logical_cores": os.cpu_count(),
             "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z")}


def _decode(tok, ids) -> str:
    try:
        return tok.decode(list(ids))
    except Exception:
        inv = {v: k for k, v in tok.vocab.items()}
        return "".join(inv.get(int(i), "\uFFFD") for i in ids)


def _forward(model, z):
    q = model(z)
    return q[0] if isinstance(q, tuple) else q


def eval_split(model, tok, rows) -> tuple[float, float]:
    """EM（答案段逐 token argmax 精确匹配，与 LR 选择的 dev 口径一致）+ CE 损失。"""
    ce = torch.nn.CrossEntropyLoss()
    hits, loss_sum = 0, 0.0
    with torch.no_grad():
        for r in rows:
            z = torch.tensor(tok.encode(r["prompt"] + r["target"]))
            q = _forward(model, z[:-1].unsqueeze(0))
            p = len(tok.encode(r["prompt"]))
            ans = z[p:]
            hits += int(torch.equal(q[0, p - 1:p - 1 + len(ans)].argmax(-1), ans))
            loss_sum += float(ce(q.reshape(-1, tok.vocab_size),
                                  z[1:].unsqueeze(0).reshape(-1)))
    return hits / max(1, len(rows)), loss_sum / max(1, len(rows))


def sealed_eval(model, tok, sealed, majority_pos: str) -> dict:
    """封存评测（每臂一次）：EM/候选排名/捷径/跟随配对 + 答案段逐样本 logits。"""
    per, logits = [], []
    for r in sealed:
        z = torch.tensor(tok.encode(r["prompt"] + r["target"]))
        q = _forward(model, z[:-1].unsqueeze(0))
        lp = torch.log_softmax(q[0], -1)
        p = len(tok.encode(r["prompt"]))
        ans = z[p:]
        seg = lp[p - 1:p - 1 + len(ans)]
        pred = seg.argmax(-1)
        em = int(torch.equal(pred, ans))
        cands = [tok.encode(f"{r['query_entity']}在{pos}。") for pos in POSITIONS]
        scores = [sum(float(lp[p - 1 + j, t]) for j, t in enumerate(c)
                       if p - 1 + j < lp.shape[0]) for c in cands]
        target_str = r["target"]
        correct_i = next((i for i, c in enumerate(cands)
                            if _decode(tok, c) == target_str), None)
        rank_ok = None
        if correct_i is not None:
            rank_ok = int(max(range(len(cands)), key=lambda i: scores[i]) == correct_i)
        head = r["prompt"].split("\n")[0][3:]
        stmts = _STMT_RE.findall(head)
        sc = {}
        if stmts:
            le, lp_ = stmts[-1]
            fe, fp_ = stmts[0]
            sc["recent_entity"] = f"{le}在{lp_}。"
            sc["first_entity"] = f"{fe}在{fp_}。"
        sc["fixed_position"] = f"{r['query_entity']}在1号位置。"
        sc["majority_position"] = f"{r['query_entity']}在{majority_pos}。"
        parts = r["prompt"].split("\n")
        per.append({"fingerprint": r["fingerprint"], "target": target_str,
                     "pred": _decode(tok, pred.tolist()), "em": em,
                     "candidate_rank_ok": rank_ok, "query_entity": r["query_entity"],
                     "ctx": head, "ack": parts[1].split("用户：")[0] if len(parts) > 1 else "",
                     "shortcut_hit": {k: int(v == target_str) for k, v in sc.items()}})
        logits.append(seg.detach().cpu())
    n = max(1, len(per))
    ranks = [x["candidate_rank_ok"] for x in per if x["candidate_rank_ok"] is not None]
    groups: dict = {}
    for x in per:
        groups.setdefault((x["ctx"], x["ack"]), {}).setdefault(x["query_entity"], []).append(x)
    pairs = [g for g in groups.values() if len(g) >= 2]
    follow_hits = sum(1 for g in pairs
                       if all(it["em"] for items in g.values() for it in items))
    shortcut_ems = {k: sum(x["shortcut_hit"][k] for x in per) / n
                     for k in ("recent_entity", "first_entity", "fixed_position",
                                "majority_position")}
    return {"n": len(per), "em": round(sum(x["em"] for x in per) / n, 6),
             "candidate_rank_accuracy": round(sum(ranks) / max(1, len(ranks)), 6),
             "n_follow_pairs": len(pairs),
             "follow_rate": round(follow_hits / max(1, len(pairs)), 6),
             "follow_hits": follow_hits,
             "em_hits": sum(x["em"] for x in per),
             "candidate_rank_hits": sum(ranks),
             "shortcut_em": {k: round(v, 6) for k, v in shortcut_ems.items()},
             "per_sample": per, "logits": logits}


def worker(args):
    name, seed, lr, deadline_ts, out_dir_str = args
    out_dir = Path(out_dir_str)
    t0 = time.time()
    torch.set_num_threads(1)
    logp = out_dir / "logs" / f"{name}_seed{seed}.log"

    def log(msg: str) -> None:
        with logp.open("a", encoding="utf-8") as fh:
            fh.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")

    soft_deadline = deadline_ts - RESERVE_S
    blob = torch.load(str(DATA_CACHE), weights_only=False, map_location="cpu")
    enc, sealed, V = blob["enc"], blob["sealed"], blob["V"]
    majority_pos = blob["majority_pos"]
    tok = CharTokenizer.load(str(TOK))
    model = CLS[name](V)
    pc = parameter_count(model)
    if not within_tolerance(pc, REFERENCE_PARAMS, 0.05):
        raise SystemExit(f"参数硬闸失败：{name} {pc}")
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    ce = torch.nn.CrossEntropyLoss()
    log(f"init done train={len(enc)} sealed={len(sealed)} params={pc} "
         f"majority_pos={majority_pos} build_s={time.time()-t0:.1f}")
    step = 0
    events = []
    train_losses = []
    stopped = False
    stop_reason = "COMPLETE"
    order = list(range(len(enc)))
    j = 0
    epoch = 0
    random.Random(seed).shuffle(order)
    run_loss = 0.0
    for level in LEVELS:
        if step > 0:
            rate = (time.time() - t0) / step
            projected_end = time.time() + rate * (LEVELS[-1] - step) + RESERVE_S + 60
            if projected_end > deadline_ts:
                stopped = True
                stop_reason = (f"INCONCLUSIVE_BUDGET(step={step}: 观测斜率 "
                                f"{rate*1000:.1f}ms/步 外推剩余 "
                                f"{(projected_end - time.time()):.0f}s 超出全局上限)")
                log(f"early stop: {stop_reason}")
                break
        while step < level:
            if time.time() > soft_deadline:
                stopped = True
                stop_reason = f"INCONCLUSIVE_BUDGET(soft deadline at step {step})"
                log(f"soft deadline hit at step {step}")
                break
            if j + BATCH > len(order):
                epoch += 1
                order = list(range(len(enc)))
                random.Random(seed + epoch).shuffle(order)
                j = 0
            idx = order[j:j + BATCH]
            j += BATCH
            z = torch.nn.utils.rnn.pad_sequence([enc[i] for i in idx], batch_first=True)
            opt.zero_grad()
            q = _forward(model, z[:, :-1])
            loss = ce(q.reshape(-1, V), z[:, 1:].reshape(-1))
            loss.backward()
            opt.step()
            run_loss += float(loss)
            step += 1
            if step % 100 == 0:
                train_losses.append({"step": step, "loss": round(run_loss / 100, 4)})
                log(f"step {step} elapsed {time.time()-t0:.1f}s "
                     f"loss {run_loss/100:.4f}")
                run_loss = 0.0
        if stopped:
            break
        ck = out_dir / f"{name}_seed{seed}_step{level}.pt"
        torch.save({"model": model.state_dict(), "seed": seed, "step": level, "lr": lr,
                     "reference": "phase2.4-20261125",
                     "train_losses_tail": train_losses[-8:]}, ck)
        events.append({"level": level, "checkpoint": ck.name,
                        "elapsed_s": round(time.time() - t0, 1)})
        log(f"checkpoint level={level} saved elapsed={time.time()-t0:.1f}s")
    seal = sealed_eval(model, tok, sealed, majority_pos)
    seal_pub = {k: v for k, v in seal.items() if k not in ("logits", "per_sample")}
    (out_dir / f"seal_per_sample_{name}_seed{seed}.json").write_text(
        json.dumps({"model": name, "seed": seed, "lr": lr, "steps_done": step,
                     "stop_reason": stop_reason, **seal_pub,
                     "per_sample": seal["per_sample"]}, ensure_ascii=False),
        encoding="utf-8")
    torch.save({"logits": seal["logits"],
                 "fingerprints": [x["fingerprint"] for x in seal["per_sample"]],
                 "model": name, "seed": seed, "step": step},
                out_dir / f"seal_logits_{name}_seed{seed}.pt")
    row = {"model": name, "seed": seed, "lr": lr, "steps_done": step,
            "levels_done": [e["level"] for e in events], "stop_reason": stop_reason,
            "status": "COMPLETE" if step >= LEVELS[-1] else "INCONCLUSIVE_BUDGET",
            "wall_s": round(time.time() - t0, 1), "events": events,
            "train_losses_tail": train_losses[-12:], "sealed": seal_pub,
            "env": env_metadata()}
    (out_dir / f"arm_{name}_seed{seed}.json").write_text(
        json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"arm done status={row['status']} steps={step}")
    return {k: v for k, v in row.items() if k != "train_losses_tail"}


def _wilson(hits: int, n: int, z: float = 1.96) -> list:
    if n <= 0:
        return [None, None]
    p = hits / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    half = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / d
    return [round(max(0.0, c - half), 4), round(min(1.0, c + half), 4)]


def compute_gates(rows: list, *, v1_em_min=0.80, v1_follow_min=0.70,
                  v2_margin=0.10, follow_random=0.0625, min_seeds=2) -> dict:
    """v1/v2 分类（prereg amendment_v2 口径，门槛一字不改）。

    v1：EM ≥ 0.80 且 跟随率 ≥ 0.70；
    v2：EM Wilson 95%CI 下界 > 最强捷径 + 0.10，且跟随率 CI 下界 > 随机(0.25²=0.0625)
        + 0.10；每模型 ≥2/3 种子满足才算通过。
    """
    arms = [r for r in rows if r.get("model") and "sealed" in r]
    if not arms:
        return {"status": "NO_SEALED_RESULTS", "note": "没有任何臂完成封存评测"}
    shortcut_ems = arms[0]["sealed"]["shortcut_em"]
    strongest = max(shortcut_ems.values())
    per_model: dict = {}
    for m in MODELS:
        seeds = [r for r in arms if r["model"] == m]
        recs = []
        for r in seeds:
            s = r["sealed"]
            em_lo, em_hi = _wilson(s["em_hits"], s["n"])
            fo_lo, fo_hi = _wilson(s["follow_hits"], s["n_follow_pairs"])
            recs.append({"seed": r["seed"], "steps_done": r["steps_done"],
                          "status": r["status"], "stop_reason": r["stop_reason"],
                          "em": s["em"], "em_ci95": [em_lo, em_hi],
                          "candidate_rank_accuracy": s["candidate_rank_accuracy"],
                          "follow_rate": s["follow_rate"], "follow_ci95": [fo_lo, fo_hi],
                          "pass_v1": bool(s["em"] >= v1_em_min
                                           and s["follow_rate"] >= v1_follow_min),
                          "pass_v2": bool(em_lo is not None and em_lo > strongest + v2_margin
                                           and fo_lo is not None
                                           and fo_lo > follow_random + v2_margin)})
        n_v1 = sum(1 for x in recs if x["pass_v1"])
        n_v2 = sum(1 for x in recs if x["pass_v2"])
        cls = "v1_satisfied" if n_v1 >= min_seeds else (
            "v2_only" if n_v2 >= min_seeds else "neither")
        per_model[m] = {"n_seeds_completed_sealed": len(recs), "arms": recs,
                         "n_v1": n_v1, "n_v2": n_v2, "classification": cls}
    return {"strongest_shortcut": strongest, "shortcut_em": shortcut_ems,
             "thresholds": {"v1_em_min": v1_em_min, "v1_follow_min": v1_follow_min,
                             "v2_margin": v2_margin, "follow_random": follow_random,
                             "min_seeds": min_seeds}, "models": per_model}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(exist_ok=True)
    sel = json.loads(LR_SUMMARY.read_text(encoding="utf-8"))["selection"]
    lrs = {}
    for m in MODELS:
        s = sel.get(m, {})
        if s.get("status") != "SELECTED":
            raise SystemExit(f"模型 {m} 的 LR 未成功选择（{s.get('status')}）——"
                              "按规则整批不进入阶梯（INCONCLUSIVE_BUDGET）")
        lrs[m] = s["lr"]
    print(f"LR from selection: {lrs}")
    lrdata.ensure_data_cache()               # 确定性数据缓存（LR 阶段已建则直接命中）
    t0 = time.time()
    deadline = t0 + GLOBAL_CAP_S
    args = [(m, s, lrs[m], deadline, str(OUT)) for m in MODELS for s in SEEDS]
    rows = []
    timed_out = False
    with ProcessPoolExecutor(max_workers=min(9, max(1, (os.cpu_count() or 8) - 4))) as ex:
        fs = [ex.submit(worker, a) for a in args]
        try:
            for f in as_completed(fs, timeout=max(60.0, deadline + 120 - time.time())):
                try:
                    rows.append(f.result())
                except Exception as exc:                      # 单臂崩溃不拖垮全局
                    rows.append({"status": "ARM_CRASHED", "error": repr(exc)[:200]})
        except TimeoutError:
            timed_out = True
            for f in fs:
                if not f.done():
                    f.cancel()
            rows.append({"status": "GLOBAL_CAP_KILLED",
                          "note": "全局 60 分钟上限触发，未完成臂被终止（其已完成级别"
                                  " checkpoint 仍在磁盘）"})
    summary = {"wall_s": round(time.time() - t0, 1), "global_cap_s": GLOBAL_CAP_S,
                "timed_out": timed_out, "lrs": lrs, "rows": rows,
                "gates": compute_gates(rows)}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                        encoding="utf-8")
    print(json.dumps(summary["gates"], ensure_ascii=False, indent=2))
    print(f"wall_s={summary['wall_s']} rows={len(rows)} timed_out={timed_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
