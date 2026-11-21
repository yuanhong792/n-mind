"""M2.2-B 多步闭环 rollout：next 模式预测器的长程误差增长与可解码性。

任务书 4.1/4.2 要求：不能每步偷偷喂 oracle 真实状态然后称多步预测。
本脚本在**潜空间闭环**：ẑ_0 = z_true_0，此后 ẑ_{k+1} = Predictor(ẑ_k, emb(a_k), 0)——
工作空间读出置零（纯 (z,a)→z' 核的测试，如实标注口径），全程不接触未来真实观测。
真实 z_true[t] 仅用于计算误差与冻结读出器的拟合/解码对照。

指标（horizon ∈ {1,3,5,10}，全部在 test episode 上聚合）：
  - 潜空间相对误差 E(z_true[t+h], ẑ[t+h])（模型自身的归一化误差口径）
  - 解码位置准确率：冻结读出 MLP（train episode 拟合）把 ẑ 解码成格子位置
  - 负对照：copy-current（位置不动）、majority-delta（全局最常见位移）、
    shuffled-action（同一 rollout 但动作随机置换——破坏动作条件应显著变差）

用法：
  python experiments/m2_2_validation/pred_rollout.py --ckpts F10_next_s20261121,F10_next_s20261122,F10_ste_s20261121
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from nmind.config import default_config  # noqa: E402
from nmind.learning import datasets as D  # noqa: E402
from nmind.learning.checkpoint import load_checkpoint  # noqa: E402
from nmind.learning.trainer import _assemble, make_data  # noqa: E402

SRC = ROOT / "experiments" / "m2_ablation"
OUT = ROOT / "experiments" / "m2_2_validation"
HORIZONS = (1, 3, 5, 10)


def decode_pos(obs_row: np.ndarray) -> tuple[int, int]:
    return int(np.argmax(obs_row[0:6])), int(np.argmax(obs_row[6:12]))


def collect(model, rows: list[dict], device) -> dict[str, np.ndarray]:
    """真实轨迹的 z_true[t]（encoder 输出）、位置序列、动作序列。"""
    model.eval()
    zs, poss, act = [], [], []
    lengths = []
    with torch.no_grad():
        for s in range(0, len(rows), 8):
            b = _assemble(rows[s:s + 8], device)
            obs = b["obs"].cpu().numpy()
            B, T, _ = obs.shape
            z = model.encoder(b["obs"].reshape(-1, b["obs"].shape[-1]), None)
            zs.append(z.view(B, T, -1).cpu().numpy())
            act.append(b["actions"].cpu().numpy())
            for bi in range(B):
                poss.extend([decode_pos(obs[bi, t]) for t in range(T)])   # 平铺，与 z 顺序一致
                lengths.append(T)
    return {"z": np.concatenate([z_.reshape(-1, z_.shape[-1]) for z_ in zs]),
            "pos": np.array(poss), "act": np.concatenate([a.reshape(-1) for a in act]),
            "lengths": lengths}


def fit_pos_readout(model, train_rows, device, *, epochs: int = 60, seed: int = 0):
    """冻结读出：z_true → 位置（36 类）。用 train episode 拟合。"""
    tr = collect(model, train_rows, device)
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(model.cfg.z_dim, 64), torch.nn.GELU(),
                              torch.nn.Linear(64, 36))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-2, weight_decay=1e-4)
    y = torch.tensor([p[0] * 6 + p[1] for p in tr["pos"]], dtype=torch.long)
    x = torch.tensor(tr["z"], dtype=torch.float32)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        F.cross_entropy(net(x), y).backward()
        opt.step()
    net.eval()
    return net


def rollout_metrics(model, readout, test_rows, device, *, shuffled_actions: bool = False,
                    seed: int = 7) -> dict:
    """潜空间闭环 rollout，逐 horizon 报误差与解码位置准确率。"""
    cfg = default_config()
    te = collect(model, test_rows, device)
    W0 = torch.zeros(1, model.cfg.workspace_dim, device=device)   # 口径：workspace 读出置零
    inv6 = 36
    rng = np.random.default_rng(seed)
    err_by_h = {h: [] for h in HORIZONS}
    pos_hit_by_h = {h: 0 for h in HORIZONS}
    pos_n_by_h = {h: 0 for h in HORIZONS}
    pos_by_episode = te["pos"]
    idx = 0
    with torch.no_grad():
        for ep_len in te["lengths"]:
            pos_ep = pos_by_episode[idx:idx + ep_len]
            act_ep = te["act"][idx:idx + ep_len]
            z_true_ep = torch.tensor(te["z"][idx:idx + ep_len], dtype=torch.float32, device=device)
            if shuffled_actions:
                act_ep = rng.permutation(act_ep)
            idx += ep_len
            zh = z_true_ep[0:1]
            for k in range(ep_len - 1):
                a_t = torch.tensor([int(act_ep[k])], device=device)
                zh = model.predictor(zh, model.action_emb(a_t), W0)
                h = k + 1
                if h in err_by_h:
                    # 潜空间相对误差（模型自身口径）
                    diff = (z_true_ep[k + 1] - zh).pow(2).mean(dim=-1).sqrt()
                    scale = z_true_ep[k + 1].pow(2).mean(dim=-1).sqrt() + 1e-6
                    err_by_h[h].append(float(diff / scale))
                    # 解码位置
                    pred_cls = int(readout(zh).argmax(-1))
                    true_pos = pos_ep[k + 1]
                    pos_hit_by_h[h] += int(pred_cls == true_pos[0] * 6 + true_pos[1])
                    pos_n_by_h[h] += 1
    out = {}
    for h in HORIZONS:
        out[h] = {"latent_rel_error": round(float(np.mean(err_by_h[h])), 4) if err_by_h[h] else None,
                  "decoded_pos_acc": round(pos_hit_by_h[h] / max(1, pos_n_by_h[h]), 4),
                  "n": pos_n_by_h[h]}
    return out


def baselines(test_rows) -> dict:
    """copy-current 与 majority-delta 的逐 horizon 位置准确率。"""
    poss = []
    lengths = []
    for r in test_rows:
        poss.append([decode_pos(r["obs"][t]) for t in range(r["len"])])
        lengths.append(r["len"])
    deltas_all = []
    for p in poss:
        for t in range(len(p) - 1):
            deltas_all.append((p[t + 1][0] - p[t][0], p[t + 1][1] - p[t][1]))
    keys = [d[0] * 8 + d[1] for d in deltas_all]
    vals, counts = np.unique(keys, return_counts=True)
    mj = vals[int(np.argmax(counts))]
    mj_delta = (int(mj // 8), int(mj % 8))
    res = {}
    for h in HORIZONS:
        copy_hit = copy_n = mj_hit = mj_n = 0
        for p in poss:
            for t in range(len(p) - h):
                true = p[t + h]
                copy_hit += int(p[t] == true); copy_n += 1
                mjp = (min(max(p[t][0] + mj_delta[0] * h, 0), 5),
                       min(max(p[t][1] + mj_delta[1] * h, 0), 5))
                mj_hit += int(mjp == true); mj_n += 1
        res[h] = {"copy_current_acc": round(copy_hit / max(1, copy_n), 4),
                  "majority_delta_acc": round(mj_hit / max(1, mj_n), 4), "n": copy_n}
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", default="F10_next_s20261121,F10_next_s20261122,F10_ste_s20261121")
    ap.add_argument("--seed", type=int, default=20261121)
    args = ap.parse_args()

    cfg = default_config()
    device = torch.device("cpu")
    data = make_data(cfg, D.GridTask(goal_visible_steps=4, seed=args.seed), seed=args.seed)
    base = baselines(data["test"])

    runs = []
    for name in [x.strip() for x in args.ckpts.split(",") if x.strip()]:
        ckpt = SRC / f"{name}.pt"
        if not ckpt.exists():
            print(f"[skip] {ckpt}", flush=True)
            continue
        model, _meta = load_checkpoint(ckpt, cfg=cfg, device="cpu", allow_config_mismatch=True)
        model.eval()
        readout = fit_pos_readout(model, data["train"], device)
        # 读出天花板：z_true → 位置（test 上）
        te = collect(model, data["test"], device)
        y = np.array([p[0] * 6 + p[1] for p in te["pos"]])
        with torch.no_grad():
            acc_true = float((readout(torch.tensor(te["z"], dtype=torch.float32))
                              .argmax(-1).numpy() == y).mean())
        normal = rollout_metrics(model, readout, data["test"], device)
        shuf = rollout_metrics(model, readout, data["test"], device, shuffled_actions=True)
        runs.append({"arm": name, "readout_ceiling_on_z_true": round(acc_true, 4),
                     "rollout": normal, "rollout_shuffled_actions_negctl": shuf,
                     "pred_mode": getattr(model, "pred_mode", "current")})
        print(f"[{name}] mode={runs[-1]['pred_mode']} readout_ceiling={acc_true:.3f} "
              + " ".join(f"h{h}:err={normal[h]['latent_rel_error']}/pos={normal[h]['decoded_pos_acc']}"
                         for h in HORIZONS), flush=True)

    payload = {"captured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
               "git_commit": os.popen("git rev-parse HEAD").read().strip(),
               "device": "cpu", "seed": args.seed,
               "protocol": "潜空间闭环：ẑ0=z_true0，ẑ[k+1]=P(ẑ[k],a[k],w=0)；全程不喂未来真实观测；"
                           "workspace 读出置零（纯 (z,a)→z' 核口径，如实标注）",
               "baselines": base, "runs": runs}
    dest = OUT / "pred_rollout.json"
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"→ {dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
