"""数据泄漏审计：钉住 M2 复核时发现的切分缺陷，给出可复算的证据。

背景（旧代码，commit afc0d09）：
    train: default_rng(seed*1000 + 1 + i)
    test : default_rng(seed*1000 + 2 + i)
⇒ test[i] 与 train[i+1] 是同一个 RNG 流、同一个任务模板 ⇒ **逐位相同的轨迹**。
旧注释写的是"互不相交"，实际全相交。0.933 的稠密准确率是在这个切分上测的，
因此不能作为泛化证据。

修复后（v2-disjoint-pairs）：
  * 流 = SeedSequence([seed, split, i])，三组互不相干；
  * (start, goal) 组合全局去重，retry 直到拿到没见过的一对；
  * dataset_hash 带 split_version，权重文件能对上数据版本。

本脚本只依赖当前代码，可随时重跑：
    PYTHONPATH=src python experiments/2026-11-21_m2_leak_audit.py
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from nmind.config import default_config
from nmind.learning import datasets as D
from nmind.learning.trainer import make_data


def sig(row: dict) -> str:
    h = hashlib.sha256()
    h.update(np.asarray(row["obs"], dtype=np.float32).tobytes())
    h.update(np.asarray(row["actions"], dtype=np.int64).tobytes())
    return h.hexdigest()[:16]


def pairs(rows: list[dict]) -> set[tuple[int, int]]:
    """从第 0 步观测解出 (start, goal)：布局见 datasets._encode。

    [x 0:6][y 6:12][gx 12:18][gy 18:24][遮挡 24][碰撞 25]；第 0 步目标一定可见。
    """
    out = set()
    for r in rows:
        o = np.asarray(r["obs"])
        start = (int(np.argmax(o[0, 0:6])), int(np.argmax(o[0, 6:12])))
        goal = (int(np.argmax(o[0, 12:18])), int(np.argmax(o[0, 18:24])))
        out.add((start, goal))
    return out


def audit(seed: int) -> dict:
    cfg = default_config()
    data = make_data(cfg, D.GridTask(goal_visible_steps=4, seed=seed), seed=seed)
    tr, va, te = data["train"], data["val"], data["test"]
    s_tr, s_va, s_te = {sig(r) for r in tr}, {sig(r) for r in va}, {sig(r) for r in te}
    out = {
        "seed": seed, "dataset_hash": data["dataset_hash"],
        "split_version": data.get("split_version"),
        "n": {"train": len(tr), "val": len(va), "test": len(te)},
        "duplicate_train_trajectories": len(tr) - len(s_tr),
        "overlap": {"train_test": len(s_tr & s_te), "train_val": len(s_tr & s_va),
                    "val_test": len(s_va & s_te)},
    }

    p_tr, p_va, p_te = pairs(tr), pairs(va), pairs(te)
    out["pairs"] = {"train": len(p_tr), "val": len(p_va), "test": len(p_te),
                    "train_test": len(p_tr & p_te), "train_val": len(p_tr & p_va),
                    "val_test": len(p_va & p_te)}

    # 捷径基线：多数动作 / 重复上一步动作（遮挡段）
    cnt = Counter(int(a) for r in tr for a in np.asarray(r["actions"]).reshape(-1))
    maj = cnt.most_common(1)[0][0]
    hit = tot = rep_hit = rep_n = 0
    for r in te:
        o, a = np.asarray(r["obs"]), np.asarray(r["actions"])
        occ = o[:, 24] > 0.5
        hit += int((a[occ] == maj).sum())
        tot += int(occ.sum())
        # 第 0 步没有"上一步"，不计入
        rep = np.concatenate([[a[0]], a[:-1]])
        mask = occ.copy()
        mask[0] = False
        rep_hit += int((rep[mask] == a[mask]).sum())
        rep_n += int(mask.sum())
    out["shortcuts"] = {"majority_action": maj,
                        "majority_acc_occluded": round(hit / max(1, tot), 4),
                        "repeat_last_action_acc_occluded": round(rep_hit / max(1, rep_n), 4),
                        "n_occluded": tot}
    out["ok"] = (out["overlap"]["train_test"] == 0 and out["overlap"]["train_val"] == 0
                 and out["overlap"]["val_test"] == 0 and out["pairs"]["train_test"] == 0
                 and out["pairs"]["train_val"] == 0 and out["pairs"]["val_test"] == 0)
    return out


def main() -> None:
    results = [audit(s) for s in (20261121, 7)]
    for r in results:
        print(f"seed={r['seed']} hash={r['dataset_hash']} ok={r['ok']} "
              f"轨迹重叠={r['overlap']} 配对重叠={r['pairs']} 内部重复={r['duplicate_train_trajectories']}")
        print(f"  捷径基线：多数动作 {r['shortcuts']['majority_action']} "
              f"遮挡段 acc={r['shortcuts']['majority_acc_occluded']}，"
              f"重复上一步 acc={r['shortcuts']['repeat_last_action_acc_occluded']}")
    out = Path(__file__).parent / "m2_ablation" / "leak_audit.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"写入 {out}")
    if not all(r["ok"] for r in results):
        raise SystemExit("切分仍有重叠，不得继续用这份数据出结论")


if __name__ == "__main__":
    main()
