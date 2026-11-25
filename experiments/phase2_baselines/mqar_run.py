"""阶段 2.5 MQAR：生成 / 审计 / 基准 / 训练编排 / 出图。

用法：
  python mqar_run.py gen         # 生成 8 个数据集（N×划分）并写 sha256
  python mqar_run.py audit      # 读取训练器将用的同一批文件做审计（哈希不一致即失败）
  python mqar_run.py bench      # GPU 单步计时（供预注册估算）
  python mqar_run.py train      # 正控制优先的两阶段训练，全局 1800s 硬闸
  python mqar_run.py worker --cfg <json>   # 内部：单臂训练
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

from models import GRUBaseline, CausalTransformerBaseline, DeltaRuleLinearBaseline, assert_reference_match  # noqa: E402
import mqar_task as T  # noqa: E402

OUT_DIR = HERE / "mqar_20261125"
FIG_DIR = ROOT / "reports" / "figs"
PREREG = HERE / "preregister_phase25.json"
TOK = T.load_tok()
V = TOK.vocab_size
assert V == 31, V

MODELS = {"gru": GRUBaseline, "transformer": CausalTransformerBaseline, "delta": DeltaRuleLinearBaseline}
LRS = {"gru": 1e-3, "transformer": 1e-3, "delta": 3e-3}  # 沿用 2.4 选择，预注册固定
SEEDS = (20270101, 20270102, 20270103)
LEVELS = (1000, 3000, 10000, 30000)
BATCH = 32
GLOBAL_CAP_S = 1800.0
RESERVE_S = 45.0
WORKERS = 6
GATE_MODELS, GATE_NS, GATE_SPLIT = ("transformer",), (2, 4), "heldout_seq"
GATE_MIN_CI = 0.90
GATE_MIN_SEEDS = 2
DEV = "cuda"


def log(msg: str) -> None:
    print(time.strftime("[%H:%M:%S] "), msg, flush=True)


def atomic_json(path: Path, obj) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def wilson(hits: int, n: int, z: float = 1.959964) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    p = hits / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round(max(0.0, (c - r) / d), 6), round(min(1.0, (c + r) / d), 6)]


def bootstrap_ci(x: np.ndarray, b: int = 1000, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (b, len(x)))
    means = x[idx].mean(1)
    return [round(float(np.percentile(means, 2.5)), 6), round(float(np.percentile(means, 97.5)), 6)]


def env_meta() -> dict:
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE",
        "device": DEV,
        "precision": "fp32_tf32_off",
        "batch": BATCH,
        "lrs": LRS,
        "seeds": list(SEEDS),
        "levels": list(LEVELS),
        "git": subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT),
                              capture_output=True, text=True).stdout.strip(),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


# ---------------------------------------------------------------- 生成 / 审计

def cmd_gen() -> int:
    pools = T.build_pools()
    T.DATA_DIR.mkdir(parents=True, exist_ok=True)
    metas = []
    for N in T.N_LIST:
        for split in T.SPLITS:
            ds = T.build_dataset(TOK, pools, N, split)
            meta = T.save_dataset(TOK, ds, T.DATA_DIR)
            metas.append(meta)
            log(f"saved N={N} {split} L={meta['L']} sha={meta['sha256'][:12]} "
                f"possible/train={meta['possible_over_train']:.3g}")
    (T.DATA_DIR / "all_meta.json").write_text(
        json.dumps({"pools_built": 6 * T.POOL_SIZE, "items": metas}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    return 0


def _seq_keys(tok, ids: np.ndarray) -> list[set[str]]:
    out = []
    for row in ids:
        s = T.parse_seq(tok, row)
        out.append({k for k, _ in s["pairs"]})
    return out


def cmd_audit() -> int:
    rows, fail = [], False
    for N in T.N_LIST:
        for split in T.SPLITS:
            ds, meta = T.load_dataset(N, split)  # 哈希不一致在此抛错
            tr, dv, te = ds["train"], ds["dev"], ds["test"]
            unk = int((np.isin(np.concatenate([tr.ravel(), dv.ravel(), te.ravel()]),
                               [TOK.vocab["<unk>"]])).sum())
            fp_tr = {tuple(r) for r in tr}
            fp_dv = {tuple(r) for r in dv}
            fp_te = {tuple(r) for r in te}
            pair_tr = set()
            key_tr = set()
            val_tr = set()
            for r in tr:
                s = T.parse_seq(TOK, r)
                for k, v in s["pairs"]:
                    pair_tr.add((k, v))
                    key_tr.add(k)
                    val_tr.add(v)
            pair_te, key_te, val_te = set(), set(), set()
            shortcuts = {"recent": 0, "first": 0, "global_majority": 0}
            gm_val = None
            # 全局多数值：训练集中出现最多的值
            from collections import Counter
            cnt = Counter()
            for r in tr:
                s = T.parse_seq(TOK, r)
                cnt.update(v for _, v in s["pairs"])
            gm_val = cnt.most_common(1)[0][0]
            nq = 0
            for r, cands, ti in zip(te, ds["test_cand"], ds["test_true"]):
                s = T.parse_seq(TOK, r)
                for k, v in s["pairs"]:
                    pair_te.add((k, v))
                    key_te.add(k)
                    val_te.add(v)
                tgt = TOK.decode(list(cands[ti]))
                shortcuts["recent"] += int(TOK.decode(list(cands[-1])) == tgt)
                shortcuts["first"] += int(TOK.decode(list(cands[0])) == tgt)
                shortcuts["global_majority"] += int(gm_val == tgt)
                nq += 1
            rand = 1.0 / N
            sc = {k: round(v / nq, 6) for k, v in shortcuts.items()}
            valid = all(v <= rand + T.SHORTCUT_TOL + 1e-9 for v in sc.values())
            name_overlap = len(key_te & key_tr) + len(val_te & val_tr)
            row = {
                "N": N, "split": split, "sha256": meta["sha256"],
                "unk": unk, "L": meta["L"], "prompt_len": meta["prompt_len"],
                "seq_overlap_train_dev": len(fp_tr & fp_dv),
                "seq_overlap_train_test": len(fp_tr & fp_te),
                "seq_overlap_dev_test": len(fp_dv & fp_te),
                "pair_overlap_train_test": len(pair_tr & pair_te),
                "name_overlap_train_test": name_overlap,
                "possible_sequences": meta["possible_sequences"],
                "possible_over_train": meta["possible_over_train"],
                "random_level": round(rand, 4),
                "shortcut_test": sc,
                "shortcut_le_rand_plus_tol": valid,
                "n_train_pairs_distinct": len(pair_tr),
            }
            bad = (unk > 0 or row["seq_overlap_train_test"] > 0
                   or row["seq_overlap_train_dev"] > 0 or row["seq_overlap_dev_test"] > 0
                   or not valid or (split == "heldout_name" and name_overlap != 0))
            row["ok"] = not bad
            fail = fail or bad
            rows.append(row)
            log(("OK  " if not bad else "FAIL") + f" N={N} {split} {row['shortcut_test']} "
                f"pair_ov={row['pair_overlap_train_test']} name_ov={name_overlap}")
    out = {"rows": rows, "all_ok": not fail, "tokenizer_vocab_hash": TOK.vocab_hash(),
           "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    (T.DATA_DIR / "audit.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                            encoding="utf-8")
    return 1 if fail else 0


# ---------------------------------------------------------------- 评测 / 训练

def _model(name: str) -> nn.Module:
    m = MODELS[name](V)
    assert_reference_match({name: m})
    return m.to(DEV)


@torch.no_grad()
def rank_eval(model, ids, cand, true, plen, batch=256) -> np.ndarray:
    hits = []
    for i in range(0, len(ids), batch):
        x = torch.from_numpy(ids[i:i + batch]).long().to(DEV)
        c = torch.from_numpy(cand[i:i + batch]).long().to(DEV)
        t = torch.from_numpy(true[i:i + batch]).to(DEV)
        lp = torch.log_softmax(model(x), -1)
        seg = lp[:, plen - 1:plen - 1 + T.NAME_LEN, :]
        sc = seg.gather(2, c).sum(-1)          # [B,N]
        hits.append((sc.argmax(-1) == t).cpu().numpy())
    return np.concatenate(hits).astype(np.float64)


@torch.no_grad()
def free_gen(model, ids, max_new=8, batch=256) -> np.ndarray:
    outs = []
    for i in range(0, len(ids), batch):
        cur = torch.from_numpy(ids[i:i + batch]).long().to(DEV)
        B = cur.shape[0]
        fin = torch.zeros(B, dtype=torch.bool, device=DEV)
        gen = []
        for _ in range(max_new):
            nxt = model(cur)[:, -1].argmax(-1)
            nxt = torch.where(fin, torch.zeros_like(nxt), nxt)
            gen.append(nxt)
            fin = fin | (nxt == TOK.vocab["<eos>"])
            cur = torch.cat([cur, nxt[:, None]], 1)
            if bool(fin.all()):
                break
        outs.append(torch.stack(gen, 1).cpu().numpy())
    return np.concatenate(outs)


@torch.no_grad()
def sealed_eval(model, ds) -> dict:
    """封存集只在停止点评测一次：排名（主）、自由生成 EM（次）、跟随（键值互换）。"""
    te, plen = ds["test"], int(ds["prompt_len"])
    rank = rank_eval(model, te, ds["test_cand"], ds["test_true"], plen)
    gen = free_gen(model, te)
    true_ids = ds["test_cand"][np.arange(len(te)), ds["test_true"]]
    true_str = [TOK.decode(list(r)) for r in true_ids]
    gen_str = [TOK.decode(list(r)) for r in gen]
    em = np.asarray([int(g == t) for g, t in zip(gen_str, true_str)])
    sw = free_gen(model, ds["test_swap"])
    sw_ans = [TOK.decode(list(r)) for r in ds["test_swap_ans"]]
    sw_gen = [TOK.decode(list(r)) for r in sw]
    follow = np.asarray([int(g == t) for g, t in zip(sw_gen, sw_ans)])
    # 答案段 teacher-forced logits（逐样本留存）
    lps = []
    for i in range(0, len(te), 256):
        x = torch.from_numpy(te[i:i + 256]).long().to(DEV)
        lp = torch.log_softmax(model(x), -1)
        lps.append(lp[:, plen - 1:plen + T.NAME_LEN, :].cpu())  # [B,5,V] 4字+eos
    logits = torch.cat(lps)
    qkeys = [T.parse_seq(TOK, r)["query_key"] for r in te]
    per = [{"i": int(i), "query_key": qkeys[i], "target": true_str[i], "pred": gen_str[i],
            "em": int(em[i]), "rank_ok": int(rank[i]), "follow_ok": int(follow[i]),
            "random_level": 1.0 / int(ds["N"])}
           for i in range(len(te))]
    return {
        "n": int(len(te)), "N": int(ds["N"]),
        "rank_acc": round(float(rank.mean()), 6),
        "rank_ci95_boot": bootstrap_ci(rank, seed=int(ds["N"])),
        "rank_ci95_wilson": wilson(int(rank.sum()), len(rank)),
        "em_free": round(float(em.mean()), 6),
        "em_free_ci95_boot": bootstrap_ci(em, seed=1),
        "follow": round(float(follow.mean()), 6),
        "follow_ci95_wilson": wilson(int(follow.sum()), len(follow)),
        "random_rank": 1.0 / int(ds["N"]),
        "random_follow": 1.0 / int(ds["N"]),
        "per_sample": per, "logits": logits,
    }


def train_arm(cfg: dict) -> dict:
    t0 = time.time()
    deadline = cfg["deadline_ts"] - RESERVE_S
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"] % (2 ** 31))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
    ds, meta = T.load_dataset(cfg["N"], cfg["split"])
    model = _model(cfg["model"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    train = torch.from_numpy(ds["train"]).long().to(DEV)
    plen = int(ds["prompt_len"])
    gated = (cfg["model"] in GATE_MODELS and cfg["N"] in GATE_NS and cfg["split"] == GATE_SPLIT)
    rec = dict(cfg)
    rec.update({"gated": gated, "data_sha256": meta["sha256"], "env": env_meta(),
                "status": "RUNNING", "levels": [], "t_start": t0})
    outp = OUT_DIR / "arms" / f"{cfg['arm_id']}.json"
    try:
        step = 0
        n = train.shape[0]
        while step < LEVELS[-1]:
            target = next(lv for lv in LEVELS if lv > step)
            perm = torch.randperm(n, device=DEV)
            i = 0
            while step < target:
                if time.time() > deadline:
                    rec["status"] = "INCONCLUSIVE_BUDGET"
                    rec["steps_done"] = step
                    rec["wall_s"] = round(time.time() - t0, 1)
                    _finalize(rec, model, ds, outp, step)
                    return rec
                idx = perm[i:i + BATCH]
                if idx.numel() < BATCH:
                    i = 0
                    perm = torch.randperm(n, device=DEV)
                    idx = perm[:BATCH]
                x = train[idx]
                loss = nn.functional.cross_entropy(
                    model(x[:, :-1]).reshape(-1, V), x[:, 1:].reshape(-1))
                opt.zero_grad()
                loss.backward()
                opt.step()
                step += 1
                i += BATCH
            # level 完成
            dv_rank = rank_eval(model, ds["dev"], ds["dev_cand"], ds["dev_true"], plen)
            lo = wilson(int(dv_rank.sum()), len(dv_rank))[0]
            rec["levels"].append({"step": step, "dev_rank": round(float(dv_rank.mean()), 6),
                                  "dev_wilson_lo": lo,
                                  "ce": round(float(loss), 4),
                                  "wall_s": round(time.time() - t0, 1)})
            torch.save({"model": model.state_dict(), "step": step, "arm_id": cfg["arm_id"]},
                       OUT_DIR / "arms" / f"{cfg['arm_id']}_step{step}.pt")
            atomic_json(outp, rec)
            log(f"{cfg['arm_id']} step {step} dev_rank={rec['levels'][-1]['dev_rank']} "
                f"ci_lo={lo} wall={rec['levels'][-1]['wall_s']}")
            if gated and lo >= GATE_MIN_CI:
                rec["status"] = "COMPLETE_GATE"
                rec["steps_done"] = step
                rec["wall_s"] = round(time.time() - t0, 1)
                _finalize(rec, model, ds, outp, step)
                return rec
        rec["status"] = "COMPLETE"
        rec["steps_done"] = step
        rec["wall_s"] = round(time.time() - t0, 1)
        _finalize(rec, model, ds, outp, step)
        return rec
    except Exception as e:  # noqa: BLE001
        rec["status"] = "ARM_CRASHED"
        rec["error"] = repr(e)
        rec["steps_done"] = rec.get("steps_done", 0)
        atomic_json(outp, rec)
        log(f"{cfg['arm_id']} CRASH {e!r}")
        return rec


def _finalize(rec, model, ds, outp, step) -> None:
    ev = sealed_eval(model, ds)
    per, logits = ev.pop("per_sample"), ev.pop("logits")
    aid = rec["arm_id"]
    (OUT_DIR / "arms" / f"{aid}_per_sample.json").write_text(
        json.dumps({"arm_id": aid, "per_sample": per}, ensure_ascii=False), encoding="utf-8")
    torch.save(logits, OUT_DIR / "arms" / f"{aid}_seal_logits.pt")
    rec["test"] = ev
    atomic_json(outp, rec)
    log(f"{aid} sealed rank={ev['rank_acc']} em={ev['em_free']} follow={ev['follow']} "
        f"steps={step} status={rec['status']}")


def cmd_worker(cfg_path: str) -> int:
    cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
    train_arm(cfg)
    return 0


# ---------------------------------------------------------------- 基准 / 编排

def cmd_bench() -> int:
    rows = []
    for name, cls in MODELS.items():
        for N in T.N_LIST:
            ds, _ = T.load_dataset(N, "heldout_seq")
            x = torch.from_numpy(ds["train"][:BATCH]).long().to(DEV)
            m = _model(name)
            opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
            for _ in range(5):
                opt.zero_grad()
                nn.functional.cross_entropy(m(x[:, :-1]).reshape(-1, V),
                                            x[:, 1:].reshape(-1)).backward()
                opt.step()
            torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(20):
                opt.zero_grad()
                nn.functional.cross_entropy(m(x[:, :-1]).reshape(-1, V),
                                            x[:, 1:].reshape(-1)).backward()
                opt.step()
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t) * 1000 / 20
            rows.append({"model": name, "N": N, "L": int(ds["L"]),
                         "step_ms": round(ms, 3), "est_30k_s": round(ms * 30000 / 1000, 1)})
            log(f"bench {name} N={N} L={ds['L']} {ms:.1f}ms/step 30k≈{ms*30:.0f}s")
            del m, opt, x
            torch.cuda.empty_cache()
    (T.DATA_DIR / "bench_gpu.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


def arm_id(model, N, split, seed) -> str:
    return f"{model}_N{N}_{split}_s{seed}"


def all_arms() -> list[dict]:
    arms = []
    for split in T.SPLITS:
        for N in T.N_LIST:
            for model in MODELS:
                for seed in SEEDS:
                    arms.append({"arm_id": arm_id(model, N, split, seed), "model": model,
                                 "N": N, "split": split, "seed": seed, "lr": LRS[model],
                                 "steps_max": LEVELS[-1]})
    def prio(a):
        gated = a["model"] in GATE_MODELS and a["N"] in GATE_NS and a["split"] == GATE_SPLIT
        return (0 if gated else 1, GATE_NS.index(a["N"]) if gated else T.N_LIST.index(a["N"]),
                0 if a["split"] == GATE_SPLIT else 1,
                ["transformer", "gru", "delta"].index(a["model"]), a["seed"])
    arms.sort(key=prio)
    return arms


def _spawn(arm: dict, deadline_ts: float):
    cfg = dict(arm)
    cfg["deadline_ts"] = deadline_ts
    p = OUT_DIR / "arms" / f"{arm['arm_id']}.cfg.json"
    atomic_json(p, cfg)
    logp = OUT_DIR / "arms" / f"{arm['arm_id']}.log"
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    proc = subprocess.Popen([sys.executable, str(HERE / "mqar_run.py"), "worker",
                             "--cfg", str(p)], stdout=open(logp, "w", encoding="utf-8"),
                            stderr=subprocess.STDOUT, env=env, cwd=str(ROOT))
    return proc


def _read_arm(arm) -> dict | None:
    p = OUT_DIR / "arms" / f"{arm['arm_id']}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def cmd_train() -> int:
    assert PREREG.exists(), "缺少预注册文件，禁止训练"
    pre = json.loads(PREREG.read_text(encoding="utf-8"))
    assert pre["status"].startswith("PREREGISTERED"), pre["status"]
    (OUT_DIR / "arms").mkdir(parents=True, exist_ok=True)
    t_end = time.time() + GLOBAL_CAP_S
    arms = all_arms()
    gated = [a for a in arms if a["model"] in GATE_MODELS and a["N"] in GATE_NS
             and a["split"] == GATE_SPLIT]
    rest = [a for a in arms if a not in gated]
    log(f"phase1 正控制 {len(gated)} 臂先行；全局截止 {time.ctime(t_end)}")

    running = {a["arm_id"]: _spawn(a, t_end) for a in gated}
    garm = {a["arm_id"]: a for a in gated}
    while running:
        for aid, p in list(running.items()):
            if p.poll() is not None:
                running.pop(aid)
        time.sleep(3)
    phase1 = [_read_arm(a) for a in gated]
    for r in phase1:
        if r is None or r["status"] not in ("COMPLETE", "COMPLETE_GATE"):
            log(f"phase1 未完成/失败: {r}")
            summary_fail(r)
            return 1
    # 正控制判定：N=2 与 N=4 各自 ≥2 种子 dev Wilson 下界 ≥0.90
    gates = {}
    for N in GATE_NS:
        rs = [r for r in phase1 if r["N"] == N]
        oks = [r for r in rs if max(l["dev_wilson_lo"] for l in r["levels"]) >= GATE_MIN_CI]
        gates[f"N{N}"] = {"n_pass_seeds": len(oks), "seeds": [
            {"seed": r["seed"], "dev_wilson_lo": max(l["dev_wilson_lo"] for l in r["levels"]),
             "status": r["status"], "steps": r["steps_done"]} for r in rs]}
    gate_pass = all(g["n_pass_seeds"] >= GATE_MIN_SEEDS for g in gates.values())
    log(f"正控制判定 gate_pass={gate_pass} {json.dumps(gates)}")
    if not gate_pass:
        summary_fail({"reason": "POSITIVE_CONTROL_FAILED", "gates": gates})
        return 1

    log(f"phase2 其余 {len(rest)} 臂，剩余预算 {t_end - time.time():.0f}s")
    running.clear()
    queue = list(rest)
    while queue or running:
        if time.time() > t_end:
            for p in running.values():
                p.terminate()
            break
        while queue and len(running) < WORKERS and time.time() < t_end:
            a = queue.pop(0)
            running[a["arm_id"]] = _spawn(a, t_end)
        for aid, p in list(running.items()):
            if p.poll() is not None:
                running.pop(aid)
        time.sleep(2)
    # 收尾：可能有臂被终止没写完
    results = []
    for a in arms:
        r = _read_arm(a)
        if r is None:
            r = {"arm_id": a["arm_id"], "status": "INCONCLUSIVE_BUDGET", "steps_done": 0}
        results.append(r)
    write_summary(results, gates, gate_pass, pre)
    return 0


def summary_fail(obj) -> None:
    (OUT_DIR / "summary.json").write_text(
        json.dumps({"status": "STOPPED", "detail": obj}, ensure_ascii=False, indent=2),
        encoding="utf-8")


def write_summary(results, gates, gate_pass, pre) -> None:
    def agg(split, model, N, key="rank_acc"):
        rs = [r for r in results if r.get("split") == split and r.get("model") == model
              and r.get("N") == N and "test" in r]
        if not rs:
            return None
        vals = [r["test"][key] for r in rs]
        return {"n_seeds": len(rs), "mean": round(float(np.mean(vals)), 6),
                "std": round(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, 6),
                "seeds": [{"seed": r["seed"], "status": r["status"], "steps": r["steps_done"],
                           **{k: r["test"][k] for k in
                              ("rank_acc", "rank_ci95_boot", "em_free", "follow")}}
                          for r in rs]}
    table = []
    for split in T.SPLITS:
        for model in MODELS:
            for N in T.N_LIST:
                a = agg(split, model, N)
                table.append({"split": split, "model": model, "N": N, **(a or {"n_seeds": 0})})
    out = {"status": "DONE" if gate_pass else "POSITIVE_CONTROL_FAILED",
           "gate_pass": gate_pass, "gates": gates,
           "global_cap_s": GLOBAL_CAP_S, "prereg": pre.get("commit", ""),
           "env": env_meta(), "table": table, "n_arms": len(results),
           "n_complete": sum(1 for r in results if str(r.get("status", "")).startswith("COMPLETE")),
           "n_inconclusive": sum(1 for r in results if r.get("status") == "INCONCLUSIVE_BUDGET"),
           "n_crashed": sum(1 for r in results if r.get("status") == "ARM_CRASHED"),
           "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    atomic_json(OUT_DIR / "summary.json", out)
    log(f"summary: complete={out['n_complete']} inconclusive={out['n_inconclusive']} "
        f"crashed={out['n_crashed']} gate={gate_pass}")


def main() -> int:
    cmd = sys.argv[1]
    if cmd == "gen":
        return cmd_gen()
    if cmd == "audit":
        return cmd_audit()
    if cmd == "bench":
        return cmd_bench()
    if cmd == "train":
        return cmd_train()
    if cmd == "worker":
        return cmd_worker(sys.argv[sys.argv.index("--cfg") + 1])
    raise SystemExit(f"unknown cmd {cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
