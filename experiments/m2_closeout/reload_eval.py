"""M2.1 收尾：checkpoint 全量登记 + 逐个重载评分 + 冻结 holdout 评测。

产出（experiments/m2_closeout/）：
  registry.json    每个已有 checkpoint 的 arm/seed/配置/sha256/记录值/重载值/一致性与 holdout
  reload_eval.json 每臂的重载评测明细（原始数字，供报告引用）

评分口径与原训练完全一致（learning/trainer.py 的 evaluate + fit_posthoc_probe），
冻结 holdout 是训练时从未见过的四个任务族：
  H1_fresh_pairs  同规则、全新 (start,goal) 对（seed 777001）
  H2_len48        同规则、序列长 48（训练长 24）—— 长度外推
  H3_obstacles    同规则、新增障碍布局（seed 777002）—— 空间规则 OOD
  H4_rule_swap    reversed_control（oracle 反向）—— 预期失败的对照，检验标签没漏

用法：
  python experiments/m2_closeout/reload_eval.py            # 全量
  python experiments/m2_closeout/reload_eval.py --skip-posthoc
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from nmind.config import default_config  # noqa: E402
from nmind.learning import datasets as D  # noqa: E402
from nmind.learning.checkpoint import load_checkpoint  # noqa: E402
from nmind.learning.trainer import (  # noqa: E402
    _make_probe, evaluate, fit_posthoc_probe, make_data)

SRC = ROOT / "experiments" / "m2_ablation"
OUT = ROOT / "experiments" / "m2_closeout"

# 记录值来源：这些 JSON 的 runs[] 里存了训练当时的评分
RECORD_JSONS = ["m2_2x2_smoke.json", "m2_2x2_s1.json", "m2_2x2_iso.json",
                "m2_2x2_fix.json", "m2_2x2_fix2.json"]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def recorded_rows() -> dict[tuple[str, int], dict]:
    """从所有实验 JSON 收集 (arm, seed) → 训练当时的评分记录。"""
    out: dict[tuple[str, int], dict] = {}
    for name in RECORD_JSONS:
        p = SRC / name
        if not p.exists():
            continue
        blob = json.loads(p.read_text(encoding="utf-8"))
        for r in blob.get("runs", []):
            key = (r["arm"], int(r["seed"]))
            out[key] = {k: r.get(k) for k in
                        ("acc_all", "acc_occ", "probe_posthoc_val", "probe_posthoc_test",
                         "loss_head", "loss_tail", "steps", "wall_s", "checkpoint", "dataset_hash")}
            out[key]["recorded_in"] = name
    return out


def load_model_and_probe(ckpt: Path):
    cfg = default_config()
    model, meta = load_checkpoint(ckpt, cfg=cfg, device="cpu", allow_config_mismatch=True)
    blob = torch.load(ckpt, map_location="cpu", weights_only=True)
    probe = _make_probe(model.cfg.block_hidden, 36).to("cpu")
    probe_state = blob.get("probe")
    if probe_state is not None:
        probe.load_state_dict(probe_state)
        probe_loaded = True
    else:
        probe_loaded = False
    return model, probe, probe_loaded, meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-posthoc", action="store_true")
    args = ap.parse_args()

    ckpts = sorted(SRC.glob("*.pt"))
    recs = recorded_rows()
    cfg = default_config()

    frozen: dict[str, list] = {}
    # H1 同规则全新 (start,goal) 对
    frozen["H1_fresh_pairs"] = make_data(cfg, D.GridTask(goal_visible_steps=4, seed=777001),
                                         seed=777001)["test"]
    # H2 长度外推：T=48（训练 T=24）
    d2 = make_data(cfg, D.GridTask(goal_visible_steps=4, seed=777001), seed=777001)
    cfg48 = cfg.model_copy(update={"learning": cfg.learning.model_copy(update={"traj_len": 48})})
    frozen["H2_len48"] = make_data(cfg48, D.GridTask(goal_visible_steps=4, seed=777001),
                                   seed=777001)["test"]
    del d2
    # H3 空间规则 OOD：障碍布局
    frozen["H3_obstacles"] = make_data(cfg, D.GridTask(goal_visible_steps=4, seed=777002,
                                                       obstacles=((2, 2), (2, 3), (3, 2))),
                                       seed=777002)["test"]
    # H4 规则反转对照（预期失败）
    frozen["H4_rule_swap"] = make_data(cfg, D.GridTask(goal_visible_steps=4, seed=777003,
                                                       reversed_control=True),
                                       seed=777003)["test"]

    registry: list[dict] = []
    reload_detail: list[dict] = []
    t_all = time.perf_counter()
    for ckpt in ckpts:
        stem = ckpt.stem
        seed = int(stem.rsplit("_s", 1)[1])
        arm = stem[: stem.rfind("_s")]
        rec = recs.get((arm, seed), {})
        entry: dict = {"arm": arm, "seed": seed, "checkpoint": str(ckpt),
                       "sha256": sha256_of(ckpt),
                       "recorded": rec, "reload": {}, "status": ""}
        try:
            model, probe, probe_loaded, meta = load_model_and_probe(ckpt)
            data = make_data(cfg, D.GridTask(goal_visible_steps=4, seed=seed), seed=seed)
            ev = evaluate(model, probe if probe_loaded else None, data["test"], device="cpu")
            reload = {"acc_all": round(ev["accuracy"], 4),
                      "acc_occ": round(ev.get("accuracy_occluded", float("nan")), 4),
                      "n_occ": int(ev.get("n_steps_occluded", 0)),
                      "dataset_hash_match": data["dataset_hash"] == rec.get("dataset_hash"),
                      "probe_in_checkpoint": probe_loaded}
            if not args.skip_posthoc:
                pos = fit_posthoc_probe(model, data["train"], data["test"], device="cpu", seed=seed)
                reload["probe_posthoc_test"] = round(pos["probe_posthoc_accuracy"], 4)
            # 冻结 holdout：决策指标（不碰训练，不调参）
            for hname, hrows in frozen.items():
                hev = evaluate(model, None, hrows, device="cpu")
                reload[hname] = {"acc": round(hev["accuracy"], 4),
                                 "acc_occ": round(hev.get("accuracy_occluded", float("nan")), 4)}
            entry["reload"] = reload
            d_occ = None
            if rec.get("acc_occ") is not None:
                d_occ = round(reload["acc_occ"] - float(rec["acc_occ"]), 4)
            entry["delta_acc_occ_vs_record"] = d_occ
            entry["status"] = "verified" if rec else "no_record"
            line = (f"[{arm} s{seed}] reload occ={reload['acc_occ']} "
                    f"(记录 {rec.get('acc_occ')}，Δ{d_occ}) "
                    f"H1={reload['H1_fresh_pairs']['acc_occ']} "
                    f"H2={reload['H2_len48']['acc_occ']} "
                    f"H3={reload['H3_obstacles']['acc_occ']} "
                    f"H4={reload['H4_rule_swap']['acc_occ']}")
        except Exception as exc:  # noqa: BLE001 - 单个 checkpoint 坏了不能拖垮全量登记
            entry["status"] = f"load_failed: {type(exc).__name__}: {exc}"
            line = f"[{arm} s{seed}] FAILED: {exc}"
        registry.append(entry)
        reload_detail.append(entry)
        print(line, flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "registry.json").write_text(
        json.dumps({"generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "n_checkpoints": len(registry),
                    "n_verified": sum(1 for e in registry if e["status"] == "verified"),
                    "n_failed": sum(1 for e in registry if e["status"].startswith("load_failed")),
                    "frozen_holdouts": {k: len(v) for k, v in frozen.items()},
                    "env": {"torch": torch.__version__, "device": "cpu",
                            "threads": torch.get_num_threads()},
                    "checkpoints": registry}, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "reload_eval.json").write_text(
        json.dumps(reload_detail, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"registry → {OUT / 'registry.json'}（{len(registry)} 项，"
          f"用时 {time.perf_counter() - t_all:.0f}s）", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
