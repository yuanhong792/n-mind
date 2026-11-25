"""阶段 2.7 B：LR 网格 → 主运行（60 臂）→ N=8 扩展（余量不足则 NOT_RUN）。

修订（第二次启动）：并行 4（6 并发曾触发 cuDNN/OOM）、spawn 错峰 4s、
CUDA 类崩溃自动重试一次；已完成的 LR 臂不重跑。
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

import mqar_task as T  # noqa: E402
from mqar_run import (TOK, OUT_DIR, log, atomic_json,  # noqa: E402
                      _spawn, _read_arm)

PREREG = HERE / "preregister_phase27.json"
MODELS = ("transformer", "gru", "delta")
LRS = (0.0003, 0.001, 0.003)
SEEDS5 = (20270101, 20270102, 20270103, 20270104, 20270105)
NS = (2, 4)
SPLITS = ("heldout_seq", "heldout_name")
GLOBAL_CAP_S = 3600.0
WORKERS = 4
STAGGER_S = 4.0


def fingerprints(ds) -> set:
    out = set()
    for arr in (ds["train"], ds["dev"], ds["test"]):
        for row in arr:
            s = T.parse_seq(TOK, row)
            ks = [k for k, _ in s["pairs"]]
            vs = [v for _, v in s["pairs"]]
            out.add(tuple(zip(ks, vs)) + (ks.index(s["query_key"]),))
    return out


def build_seal_name(N: int):
    pools = T.build_pools()
    used = set()
    for k in pools:
        used.update(pools[k])
    rng = random.Random(20271203)
    nkey = T.gen_names(rng, T.POOL_SIZE, used)
    nval = T.gen_names(rng, T.POOL_SIZE, used)
    ds, _ = T.load_dataset(N, "heldout_name")
    exclude = fingerprints(ds)
    seqs = T.gen_seqs(rng, nkey, nval, 1000, N, exclude)

    def enc(seqs):
        ids, plens = [], []
        for ks, vs, qi in seqs:
            full, pl = T.encode_seq(TOK, ks, vs, qi)
            ids.append(full)
            plens.append(pl)
        assert len(set(plens)) == 1
        return np.asarray(ids, np.int32), plens[0]

    te, plen = enc(seqs)
    c = np.asarray([[TOK.encode(v) for v in vs] for ks, vs, qi in seqs],
                   dtype=np.int32).reshape(len(seqs), N, T.NAME_LEN)
    t = np.asarray([qi for ks, vs, qi in seqs], dtype=np.int64)
    ids2, ans = [], []
    for ks, vs, qi in seqs:
        full, _ = T.encode_seq(TOK, vs, ks, qi)
        ids2.append(full)
        ans.append(TOK.encode(ks[qi]))
    arrs = {"test": te, "test_cand": c, "test_true": t,
            "test_swap": np.asarray(ids2, np.int32),
            "test_swap_ans": np.asarray(ans, np.int32)}
    path = T.DATA_DIR / f"mqar_N{N}_heldout_name_seal3.npz"
    np.savez_compressed(path, **arrs)
    meta = {"N": N, "split": "heldout_name_seal3", "prompt_len": plen, "L": int(te.shape[1]),
            "sha256": T.sha256_file(path), "seed": 20271203, "n": len(te),
            "name_overlap_with_all_pools": 0, "seq_overlap_with_train_dev_test": 0}
    (T.DATA_DIR / f"mqar_N{N}_heldout_name_seal3.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    assert not (fingerprints({"train": te, "dev": np.zeros((0, te.shape[1]), np.int32),
                              "test": np.zeros((0, te.shape[1]), np.int32)}) & exclude)
    print(f"seal_name N={N} sha={meta['sha256'][:12]} n={len(te)}", flush=True)
    return path


def run_pool(arms, t_end):
    """≤WORKERS 并行、错峰 spawn、CUDA 类崩溃重试一次。"""
    queue = list(arms)
    running = {}
    retried = set()
    while queue or running:
        if time.time() > t_end:
            for p in running.values():
                p.terminate()
            break
        while queue and len(running) < WORKERS and time.time() < t_end:
            a = queue.pop(0)
            running[a["arm_id"]] = _spawn(a, t_end)
            time.sleep(STAGGER_S)
        for aid, p in list(running.items()):
            if p.poll() is not None:
                running.pop(aid)
                r = _read_arm({"arm_id": aid})
                if (r and r.get("status") == "ARM_CRASHED" and aid not in retried
                        and any(k in str(r.get("error", "")) for k in ("CUDA", "cuDNN", "cublas"))):
                    retried.add(aid)
                    log(f"retry {aid}: {str(r.get('error'))[:80]}")
                    for a in arms:
                        if a["arm_id"] == aid:
                            queue.append(a)
                            break
        time.sleep(2)


def wait_pool(arms, t_end):
    run_pool(arms, t_end)
    return [_read_arm(a) for a in arms]


def main():
    pre = json.loads(PREREG.read_text(encoding="utf-8"))
    assert pre["status"].startswith("PREREGISTERED")
    (OUT_DIR / "arms").mkdir(parents=True, exist_ok=True)
    t_end = time.time() + GLOBAL_CAP_S
    log(f"phase27 B restart, workers={WORKERS} stagger={STAGGER_S}s, deadline {time.ctime(t_end)}")

    # --- LR 选择（已完成的臂不重跑） ---
    lr_arms = []
    for model in MODELS:
        for lr in LRS:
            aid = f"lr_{model}_{lr}_N4seq"
            r = _read_arm({"arm_id": aid})
            if r and str(r.get("status")).startswith("COMPLETE") and r.get("levels"):
                continue
            lr_arms.append({"arm_id": aid, "model": model, "lr": lr,
                            "N": 4, "split": "heldout_seq", "seed": 20270101,
                            "max_steps": 3000, "gate_all": False})
    log(f"LR 待跑 {len(lr_arms)}（其余沿用既有结果）")
    if lr_arms:
        wait_pool(lr_arms, t_end)
    lrs = {}
    lr_detail = {}
    for model in MODELS:
        best = None
        for lr in LRS:
            r = _read_arm({"arm_id": f"lr_{model}_{lr}_N4seq"})
            if not r or not r.get("levels"):
                lr_detail[f"{model}_{lr}"] = "MISSING_AFTER_RETRY"
                continue
            lv = r["levels"][-1]
            lr_detail[f"{model}_{lr}"] = {"dev_rank": lv["dev_rank"],
                                           "dev_ce": lv.get("dev_ce"),
                                           "status": r.get("status")}
            log(f"LR {model} {lr} rank={lv['dev_rank']} ce={lv.get('dev_ce')}")
            key = (lv["dev_rank"], -lv.get("dev_ce", 9e9))
            if best is None or key > best[0]:
                best = (key, lr)
        if best is None:
            raise SystemExit(f"{model} 三个 LR 全部失败，停止")
        lrs[model] = best[1]
        log(f"SELECTED lr {model}={lrs[model]}")
    atomic_json(OUT_DIR / "phase27_lr_select.json",
                {"lrs": lrs, "detail": lr_detail, "eval_sanity": "PASS 8/8 @ 54af4dc",
                 "note": "若网格点崩溃重试后仍失败，则在该模型其余有效网格点中按预注册规则选择，并在报告披露"})

    # --- seal_name ---
    seal_name = {N: build_seal_name(N) for N in NS}
    seal_seq_path = {2: T.DATA_DIR / "mqar_N2_heldout_seq_seal2.npz",
                     4: T.DATA_DIR / "mqar_N4_heldout_seq_seal2.npz"}

    # --- 主运行 60 臂 ---
    main_arms = []
    for split in SPLITS:
        for N in NS:
            for model in MODELS:
                for seed in SEEDS5:
                    main_arms.append({
                        "arm_id": f"p27_{model}_N{N}_{split}_s{seed}",
                        "model": model, "lr": lrs[model], "N": N, "split": split,
                        "seed": seed, "gate_all": True,
                        "seal_path": str(seal_seq_path[N] if split == "heldout_seq"
                                          else seal_name[N])})
    log(f"main {len(main_arms)} 臂")
    run_pool(main_arms, t_end)

    # --- N=8 扩展 ---
    n8 = []
    remain = t_end - time.time()
    if remain >= 1200:
        log(f"N=8 扩展，剩余 {remain:.0f}s")
        n8_arms = []
        for model in MODELS:
            for seed in SEEDS5:
                n8_arms.append({
                    "arm_id": f"p27_{model}_N8_heldout_seq_s{seed}",
                    "model": model, "lr": lrs[model], "N": 8, "split": "heldout_seq",
                    "seed": seed, "gate_all": True, "seal_path": ""})
        run_pool(n8_arms, t_end)
        n8 = [(_read_arm({"arm_id": a["arm_id"]}) or {"arm_id": a["arm_id"],
                                                      "status": "INCONCLUSIVE_BUDGET"})
              for a in n8_arms]
    else:
        log(f"N=8 NOT_RUN，剩余 {remain:.0f}s < 1200s")
        n8 = [{"status": "NOT_RUN_BUDGET", "note": "主运行+LR 后余量不足"}]

    # --- 汇总 ---
    results = []
    for a in main_arms:
        r = _read_arm(a)
        if r is None:
            r = {"arm_id": a["arm_id"], "status": "INCONCLUSIVE_BUDGET", "steps_done": 0}
        results.append(r)
    table = []
    for split in SPLITS:
        for model in MODELS:
            for N in NS:
                rs = [r for r in results if r.get("split") == split and r.get("model") == model
                      and r.get("N") == N and "test" in r]
                n_gate, firsts = 0, []
                for r in rs:
                    los = [lv.get("dev_wilson_lo", 0) for lv in r.get("levels", [])]
                    if any(x >= 0.90 for x in los):
                        n_gate += 1
                        firsts.append(next(lv["step"] for lv in r["levels"]
                                           if lv["dev_wilson_lo"] >= 0.90))
                ranks = [r["test"]["rank_acc"] for r in rs]
                ems = [r["test"]["em_free"] for r in rs]
                fols = [r["test"]["follow"] for r in rs]
                table.append({
                    "split": split, "model": model, "N": N, "lr": lrs[model],
                    "n_seeds_done": len(rs), "n_gate_seeds": n_gate,
                    "first_gate_steps": firsts,
                    "rank_mean": round(float(np.mean(ranks)), 4) if ranks else None,
                    "em_mean": round(float(np.mean(ems)), 4) if ems else None,
                    "follow_mean": round(float(np.mean(fols)), 4) if fols else None,
                    "seeds": [{"seed": r.get("seed"), "status": r.get("status"),
                               "steps": r.get("steps_done"),
                               "rank": r.get("test", {}).get("rank_acc"),
                               "em": r.get("test", {}).get("em_free"),
                               "follow": r.get("test", {}).get("follow"),
                               "n_clusters": r.get("test", {}).get("n_clusters")}
                              for r in rs]})
    summary = {"status": "DONE" if all(not _read_arm(a) or
                                       str((_read_arm(a) or {}).get("status", "")).startswith("COMPLETE")
                                       for a in main_arms) else "PARTIAL",
               "eval_sanity": "PASS 8/8 @ 54af4dc",
               "lrs": lrs, "lr_detail": lr_detail, "table": table,
               "n8_status": "DONE" if n8 and n8[0].get("status") and str(n8[0].get("status")).startswith("COMPLETE") else
                            ("NOT_RUN_BUDGET" if n8 and n8[0].get("status") == "NOT_RUN_BUDGET" else "INCONCLUSIVE_BUDGET"),
               "n8_ranks": [r.get("test", {}).get("rank_acc") for r in n8 if r.get("test")],
               "n_arms": len(results),
               "n_complete": sum(1 for r in results if str(r.get("status", "")).startswith("COMPLETE")),
               "n_inconclusive": sum(1 for r in results if r.get("status") == "INCONCLUSIVE_BUDGET"),
               "n_crashed": sum(1 for r in results if r.get("status") == "ARM_CRASHED"),
               "wall_s": round(GLOBAL_CAP_S - (t_end - time.time()), 1),
               "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    atomic_json(OUT_DIR / "phase27_summary.json", summary)
    log(f"done complete={summary['n_complete']} incl={summary['n_inconclusive']} "
        f"crash={summary['n_crashed']} lrs={lrs} n8={summary['n8_status']}")


if __name__ == "__main__":
    main()
