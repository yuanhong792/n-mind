import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

arms = Path("experiments/phase2_baselines/mqar_20261125/arms")
rows = []
for p in sorted(arms.glob("transformer_*.json")):
    if p.name.endswith(".cfg.json") or "_per_sample" in p.name:
        continue
    r = json.loads(p.read_text(encoding="utf-8"))
    t = r["test"]
    rows.append(dict(arm=r["arm_id"], N=r["N"], seed=r["seed"], status=r["status"],
                     steps=r["steps_done"], rank=t["rank_acc"], rank_ci=t["rank_ci95_boot"],
                     em=t["em_free"], em_ci=t["em_free_ci95_boot"], follow=t["follow"],
                     follow_ci=t["follow_ci95_wilson"],
                     dev_curve=[(l["step"], l["dev_rank"]) for l in r["levels"]]))
    print(rows[-1]["arm"], "rank", rows[-1]["rank"], "em", rows[-1]["em"], "follow", rows[-1]["follow"])
out = Path("experiments/phase2_baselines/mqar_20261125/gate_arms.json")
out.write_text(json.dumps(rows, indent=2), encoding="utf-8")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ax, key, title in zip(axes, ["rank", "em"], ["candidate rank acc", "free-gen EM"]):
    for N in (2, 4):
        rs = [r for r in rows if r["N"] == N]
        ax.plot([str(r["seed"])[-2:] for r in rs], [r[key] for r in rs], "o", label=f"N={N}")
        ax.axhline(1.0 / N, ls=":", color="gray")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
axes[0].legend()
axes[0].set_ylabel("heldout_seq test")
fig.suptitle("Transformer positive control (gate on rank CI lower >= 0.90: FAIL)")
fig.tight_layout()
fig.savefig("reports/figs/phase25_gate_seeds.png", dpi=150)

fig2, ax = plt.subplots(figsize=(6.5, 4))
for N in (2, 4):
    for r in [x for x in rows if x["N"] == N]:
        lab = f"N={N} s{str(r['seed'])[-2:]}"
        ax.plot([s for s, _ in r["dev_curve"]], [v for _, v in r["dev_curve"]],
                marker="o", alpha=0.6, label=lab)
ax.axhline(0.90, color="r", ls=":", label="gate 0.90")
ax.set_xscale("log")
ax.set_xlabel("steps")
ax.set_ylabel("dev rank acc")
ax.legend(fontsize=7)
ax.grid(alpha=0.3)
ax.set_title("dev rank curves (transformer heldout_seq)")
fig2.tight_layout()
fig2.savefig("reports/figs/phase25_gate_curves.png", dpi=150)
print("plots ok")
