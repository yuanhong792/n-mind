"""阶段 2.7 B 汇总：按单元出表（有效臂 vs 截断臂）+ 图。"""
import json
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE / "mqar_20261125"
ARMS = OUT / "arms"
MODELS = ("transformer", "gru", "delta")
NS = (2, 4)
SPLITS = ("heldout_seq", "heldout_name")
SEEDS = (20270101, 20270102, 20270103, 20270104, 20270105)


def load(aid):
    p = ARMS / f"{aid}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


rows = []
for split in SPLITS:
    for model in MODELS:
        for N in NS:
            valid, trunc = [], []
            for seed in SEEDS:
                aid = f"p27_{model}_N{N}_{split}_s{seed}"
                r = load(aid)
                if r is None:
                    trunc.append({"seed": seed, "steps": 0, "status": "NOT_STARTED"})
                    continue
                st = r.get("status")
                if st in ("COMPLETE", "COMPLETE_GATE") and r.get("test"):
                    los = [lv.get("dev_wilson_lo", 0) for lv in r.get("levels", [])]
                    first = next((lv["step"] for lv in r["levels"]
                                  if lv.get("dev_wilson_lo", 0) >= 0.90), None)
                    valid.append({"seed": seed, "status": st, "steps": r.get("steps_done"),
                                  "first_gate": first, "gate": first is not None,
                                  "rank": r["test"]["rank_acc"], "em": r["test"]["em_free"],
                                  "follow": r["test"]["follow"],
                                  "n_clusters": r["test"].get("n_clusters")})
                else:
                    trunc.append({"seed": seed, "steps": r.get("steps_done", 0),
                                  "status": st})
            rows.append({"split": split, "model": model, "N": N, "valid": valid, "trunc": trunc})

out = {"eval_sanity": "PASS 8/8 @ 54af4dc", "rows": rows,
       "seal3": {n: json.loads((HERE / "mqar_data" / f"mqar_N{n}_heldout_name_seal3.meta.json")
                               .read_text(encoding="utf-8"))["sha256"][:12] for n in NS},
       "seal2": {n: json.loads((HERE / "mqar_data" / f"mqar_N{n}_heldout_seq_seal2.meta.json")
                               .read_text(encoding="utf-8"))["sha256"][:12] for n in NS}}
(OUT / "phase27_table.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

for r in rows:
    v = r["valid"]
    gate = sum(1 for x in v if x["gate"])
    if not v:
        print(f'{r["split"]:13s} {r["model"]:11s} N={r["N"]:2d} valid=0')
        continue
    rk = [x["rank"] for x in v]
    em = [x["em"] for x in v]
    fo = [x["follow"] for x in v]
    print(f'{r["split"]:13s} {r["model"]:11s} N={r["N"]:2d} valid={len(v)} gate={gate} '
          f'rank={np.mean(rk):.3f} em={np.mean(em):.3f} follow={np.mean(fo):.3f}')
    print(f'    rank {np.round(rk,3)} em {np.round(em,3)} follow {np.round(fo,3)} '
          f'first_gate {[x["first_gate"] for x in v]} '
          f'trunc {[(t["seed"], t["steps"], t["status"]) for t in r["trunc"]]}')

# 图：两个划分并列，rank 与 EM
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
for ax, key, title in zip(axes, ("rank", "em"), ("sealed candidate rank", "free-gen EM")):
    width = 0.26
    for i, model in enumerate(MODELS):
        xs, ys = [], []
        for j, (split, N) in enumerate([(s, n) for s in SPLITS for n in NS]):
            r = [x for x in rows if x["split"] == split and x["model"] == model and x["N"] == N][0]
            if not r["valid"]:
                continue
            xs.append(j + (i - 1) * width)
            ys.append(float(np.mean([v[key] for v in r["valid"]])))
        ax.bar(xs, ys, width=width, label=model)
    ax.set_xticks(range(4))
    ax.set_xticklabels([f"{s.replace('heldout_','')}\nN={n}" for s in SPLITS for n in NS], fontsize=8)
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3, axis="y")
axes[0].axhline(0.90, color="r", ls=":", label="gate 0.90")
axes[0].legend(fontsize=8)
fig.suptitle("Phase 2.7 MQAR baselines (valid arms only; truncated arms excluded)")
fig.tight_layout()
fig.savefig(HERE.parents[1] / "reports/figs/phase27_baselines.png", dpi=150)
print("saved phase27_table.json + fig")
