"""C 项：修复后 dev 重判门槛 + 新封存集（20271202）只读一次。"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

import mqar_task as T  # noqa: E402
from mqar_run import (DEV, TOK, _model, candidate_scores, cluster_bootstrap_ci,  # noqa: E402
                      free_gen, rank_eval, sealed_eval, wilson)

ARMS = HERE / "mqar_20261125" / "arms"
GATE_NS = (2, 4)
SEEDS = (20270101, 20270102, 20270103)
LEVELS = (1000, 3000, 10000, 30000)


def fingerprints(ds) -> set:
    out = set()
    for arr in (ds["train"], ds["dev"], ds["test"]):
        for row in arr:
            s = T.parse_seq(TOK, row)
            keys = [k for k, _ in s["pairs"]]
            vals = [v for _, v in s["pairs"]]
            qi = keys.index(s["query_key"])
            out.add(tuple(zip(keys, vals)) + (qi,))
    return out


def build_seal2(N: int):
    pools = T.build_pools()
    keys, vals = pools["a_key"], pools["a_val"]
    ds, _ = T.load_dataset(N, "heldout_seq")
    exclude = fingerprints(ds)
    import random
    rng = random.Random(20271202 + N)
    seqs = T.gen_seqs(rng, keys, vals, 1000, N, exclude)
    ids, plen = [], []
    for ks, vs, qi in seqs:
        full, pl = T.encode_seq(TOK, ks, vs, qi)
        ids.append(full)
        plen.append(pl)
    assert len(set(plen)) == 1
    plen = plen[0]
    c = np.asarray([[TOK.encode(v) for v in vs] for ks, vs, qi in seqs],
                   dtype=np.int32).reshape(len(seqs), N, T.NAME_LEN)
    t = np.asarray([qi for ks, vs, qi in seqs], dtype=np.int64)

    def swap(seqs):
        ids2, ans = [], []
        for ks, vs, qi in seqs:
            full, _ = T.encode_seq(TOK, vs, ks, qi)
            ids2.append(full)
            ans.append(TOK.encode(ks[qi]))
        return np.asarray(ids2, np.int32), np.asarray(ans, np.int32)

    sw, sw_ans = swap(seqs)
    arr = np.asarray(ids, np.int32)
    out = {"N": N, "split": "heldout_seq_seal2", "prompt_len": plen, "L": int(arr.shape[1]),
           "test": arr, "test_cand": c, "test_true": t, "test_swap": sw, "test_swap_ans": sw_ans,
           "n_test": len(arr)}
    path = T.DATA_DIR / f"mqar_N{N}_heldout_seq_seal2.npz"
    np.savez_compressed(path, **{k: v for k, v in out.items() if isinstance(v, np.ndarray)})
    meta = {k: v for k, v in out.items() if not isinstance(v, np.ndarray)}
    meta["sha256"] = T.sha256_file(path)
    meta["seed"] = 20271202 + N
    meta["excluded_fingerprints"] = len(exclude)
    meta["seq_overlap_with_train_dev_test"] = 0
    (T.DATA_DIR / f"mqar_N{N}_heldout_seq_seal2.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    # 复核重叠
    fps_new = set()
    for row in arr:
        s = T.parse_seq(TOK, row)
        ks = [k for k, _ in s["pairs"]]
        vs = [v for _, v in s["pairs"]]
        fps_new.add(tuple(zip(ks, vs)) + (ks.index(s["query_key"]),))
    assert not (fps_new & exclude)
    print(f"seal2 N={N} sha={meta['sha256'][:12]} overlap=0 n={len(arr)}")
    return out


@torch.no_grad()
def dev_gate_for_arm(arm_id: str, N: int):
    ds, _ = T.load_dataset(N, "heldout_seq")
    plen = int(ds["prompt_len"])
    per_level = []
    for step in LEVELS:
        ck_path = ARMS / f"{arm_id}_step{step}.pt"
        if not ck_path.exists():
            continue
        ck = torch.load(ck_path, weights_only=True)
        model = _model("transformer")
        model.load_state_dict(ck["model"])
        rank = rank_eval(model, ds["dev"], ds["dev_cand"], ds["dev_true"], plen)
        lo = wilson(int(rank.sum()), len(rank))[0]
        per_level.append({"step": step, "dev_rank_fixed": round(float(rank.mean()), 6),
                          "wilson_lo": lo})
        del model
        torch.cuda.empty_cache()
    best = max(per_level, key=lambda x: x["wilson_lo"])
    return per_level, best


def main():
    print("=== A/C: dev 门槛重判（修复后 rank，门槛定义不变） ===")
    gate_rows = []
    for N in GATE_NS:
        for seed in SEEDS:
            arm_id = f"transformer_N{N}_heldout_seq_s{seed}"
            old = json.loads((ARMS / f"{arm_id}.json").read_text(encoding="utf-8"))
            old_best = max((l["dev_wilson_lo"] for l in old["levels"]), default=0)
            per, best = dev_gate_for_arm(arm_id, N)
            rec = {"arm": arm_id, "N": N, "seed": seed,
                   "old_buggy_best_lo": old_best,
                   "new_fixed_per_level": per,
                   "new_fixed_best": best,
                   "pass": best["wilson_lo"] >= 0.90}
            gate_rows.append(rec)
            print(f"{arm_id} old_lo={old_best:.4f} new_lo={best['wilson_lo']:.4f} "
                  f"new_rank@{best['step']}={best['dev_rank_fixed']} pass={rec['pass']}")
    for N in GATE_NS:
        n_pass = sum(1 for r in gate_rows if r["N"] == N and r["pass"])
        print(f"N={N} pass_seeds={n_pass}/3 -> {'PASS' if n_pass >= 2 else 'FAIL'}")
    gate_pass = all(sum(1 for r in gate_rows if r["N"] == N and r["pass"]) >= 2 for N in GATE_NS)
    print(f"gate_pass={gate_pass}")

    print("\n=== C2: 新封存集 seal2（每 ckpt 只读一次） ===")
    seals = {}
    for N in GATE_NS:
        seals[N] = build_seal2(N)
    seal_rows = []
    for N in GATE_NS:
        ds = seals[N]
        for seed in SEEDS:
            arm_id = f"transformer_N{N}_heldout_seq_s{seed}"
            ck = torch.load(ARMS / f"{arm_id}_step30000.pt", weights_only=True)
            model = _model("transformer")
            model.load_state_dict(ck["model"])
            ev = sealed_eval(model, ds)
            per, logits = ev.pop("per_sample"), ev.pop("logits")
            (ARMS / f"{arm_id}_seal2_per_sample.json").write_text(
                json.dumps({"arm_id": arm_id, "per_sample": per}, ensure_ascii=False),
                encoding="utf-8")
            torch.save(logits, ARMS / f"{arm_id}_seal2_logits.pt")
            seal_rows.append({"arm": arm_id, "N": N, "seed": seed, **ev})
            print(f"seal2 {arm_id} rank={ev['rank_acc']} em={ev['em_free']} "
                  f"follow={ev['follow']} n_clusters={ev['n_clusters']}")
            del model
            torch.cuda.empty_cache()
    out = {"gate_pass": gate_pass, "gate_rows": gate_rows, "seal2": seal_rows,
           "note": "评测 bug 修复后重判，门槛未改（54af4dc）；旧封存集已被读取多次，seal2 只读一次"}
    (HERE / "mqar_20261125/phase26_reeval.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved phase26_reeval.json")


if __name__ == "__main__":
    main()
