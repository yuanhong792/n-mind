"""A3：用簇级 bootstrap 重算 2.4 的 EM 与跟随率 CI（不改历史 JSON）。簇=事实对。"""
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from mqar_run import cluster_bootstrap_ci  # noqa: E402

LAD = HERE / "ladder_20261125"
_FACT_RE = re.compile(r"([^：。\n]+)在([1-4]号位置)")


def fact_pair(ctx: str) -> tuple:
    fs = tuple(sorted(_FACT_RE.findall(ctx)[:2]))
    return fs


def main():
    out = {"note": "簇级 bootstrap 重算 2.4 EM/跟随 CI；历史 JSON 未改。簇=事实对（ctx 前两条陈述）。",
           "eval_sanity": "PASS 8/8 @ 54af4dc", "arms": []}
    for p in sorted(LAD.glob("seal_per_sample_*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        per = d["per_sample"]
        em = np.asarray([x["em"] for x in per], float)
        # 跟随对：(ctx, ack) 组内 ≥2 查询实体，全 EM 才 1
        groups = {}
        for x in per:
            groups.setdefault((x["ctx"], x["ack"]), {}).setdefault(x["query_entity"], []).append(x)
        pairs = []
        pair_cluster = []
        for gkey, g in groups.items():
            if len(g) < 2:
                continue
            hit = int(all(it["em"] for items in g.values() for it in items))
            pairs.append(hit)
            any_x = next(iter(g.values()))[0]
            pair_cluster.append(fact_pair(any_x["ctx"]))
        em_clusters = [fact_pair(x["ctx"]) for x in per]
        em_ci = cluster_bootstrap_ci(em, em_clusters, seed=0)
        fl_ci = cluster_bootstrap_ci(pairs, pair_cluster, seed=1)
        rec = {"file": p.name, "model": d["model"], "seed": d["seed"],
               "em_point": d["em"], "em_ci_cluster": em_ci,
               "n_clusters_em": len(set(em_clusters)),
               "follow_point": d["follow_rate"], "follow_hits": d["follow_hits"],
               "n_follow_pairs": d["n_follow_pairs"],
               "follow_ci_cluster": fl_ci,
               "n_clusters_follow": len(set(pair_cluster))}
        out["arms"].append(rec)
        print(rec["file"], "em", rec["em_point"], em_ci, "clu", rec["n_clusters_em"],
              "follow", rec["follow_point"], fl_ci, "clu", rec["n_clusters_follow"])
    (HERE / "phase24_cluster_ci.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved phase24_cluster_ci.json")


if __name__ == "__main__":
    main()
