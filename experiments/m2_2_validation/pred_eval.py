"""M2.2-B：世界模型预测分支的直接有效性评测（提示词第 4 节）。

背景缺陷（代码核实，2026-11-21）：legacy 模式下 z_hat[t] = Predictor(z[t], a[t-1], w[t])，
损失 MSE(z_hat[t], z[t]) —— **预测器把要预测的目标当输入**。这是重建目标；
文档声称的"由 t-1 的 z 预测 z[t]"与实现不符。本脚本用可证伪的测试量化它：

  T1 动作敏感性：固定观测、变化 a_prev，z_hat 应有可感变化（世界模型对动作响应）。
  T2 拷贝检验：cosine(z_hat, z) 若≈1，说明预测器学到的是恒等映射（读目标当输入）。
  T3 未来状态读出阶梯（冻结 MLP 在 train episode 特征上拟合，test episode 上评测）：
       R1: z[t]                     —— 无动作的拷贝基线
       R2: z[t]+onehot(a[t])        —— 确定性动力学的真值等价（上限）
       R3: z_hat[t]                 —— 预测器输出是否编码了"动作后的未来位置"
       R4: z_hat[t]+onehot(a[t])
     目标变量：物体在 t+1 的位置（36 格分类，位置从 obs one-hot 解码，oracle 一致）。
  T4 负对照：多数类位移（全局最常见的 (dx,dy) 平移）、随机打乱 a 配对。

判定规则（预登记）：
  world_model_validated=true 当且仅当 R3/R4 在留出 episode 上显著优于 R1 与 T4 基线，
  且 T1 显示动作敏感性（Δz_hat / |z_hat| ≥ 0.05）。否则 world_model_validated=false。

用法：
  python experiments/m2_2_validation/pred_eval.py --ckpts C_sparse_ce_s20261121,F1_detach_s20261121,F10_ste_s20261121
"""

from __future__ import annotations

import argparse
import json
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
DIRS = D.DIRS  # (0,-1),(1,0),(0,1),(-1,0)


def decode_pos(obs_row: np.ndarray) -> tuple[int, int]:
    """从 obs one-hot 解码物体位置（[0:6]=x, [6:12]=y）。"""
    x = int(np.argmax(obs_row[0:6]))
    y = int(np.argmax(obs_row[6:12]))
    return x, y


def collect_features(model, rows: list[dict], device) -> dict[str, np.ndarray]:
    """跑一遍 forward_sequence，收集 z / z_hat / a / pos(t) / pos(t+1)。"""
    model.eval()
    zs, hats, acts, pos_t, pos_t1 = [], [], [], [], []
    with torch.no_grad():
        for s in range(0, len(rows), 8):
            b = _assemble(rows[s:s + 8], device)
            out = model.forward_sequence(b["obs"], b["actions"], mem=b["mem"])
            obs_np = b["obs"].cpu().numpy()
            B, T, _ = obs_np.shape
            zs.append(out["z"].cpu().numpy())
            hats.append(out["z_hat"].cpu().numpy())
            acts.append(b["actions"].cpu().numpy())
            for bi in range(B):
                for t in range(T):
                    pos_t.append(decode_pos(obs_np[bi, t]))
                    pos_t1.append(decode_pos(obs_np[bi, min(t + 1, T - 1)]))
    Z = np.concatenate([z.reshape(-1, z.shape[-1]) for z in zs])
    H = np.concatenate([h.reshape(-1, h.shape[-1]) for h in hats])
    A = np.concatenate([a.reshape(-1) for a in acts])
    P = np.array(pos_t)
    P1 = np.array(pos_t1)
    # 末步没有 t+1（pos_t1 复制了末位），读出训练时丢弃每条轨迹最后一步
    valid = np.ones(len(P), dtype=bool)
    idx = 0
    for s in range(0, len(rows), 8):
        chunk = rows[s:s + 8]
        for r in chunk:
            n = r["len"]
            valid[idx + n - 1] = False
            idx += n
    return {"z": Z[valid], "z_hat": H[valid], "a": A[valid],
            "pos": P[valid], "pos1": P1[valid]}


def pos_to_class(pos: np.ndarray) -> np.ndarray:
    return pos[:, 0] * 6 + pos[:, 1]


def fit_readout(x_train, y_train, x_test, y_test, *, hidden: int = 64,
                epochs: int = 60, seed: int = 0, n_classes: int = 36) -> float:
    torch.manual_seed(seed)
    din = x_train.shape[1]
    net = torch.nn.Sequential(torch.nn.Linear(din, hidden), torch.nn.GELU(),
                              torch.nn.Linear(hidden, n_classes))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-2, weight_decay=1e-4)
    xt = torch.tensor(x_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.long)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(net(xt), yt)
        loss.backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        pred = net(torch.tensor(x_test, dtype=torch.float32)).argmax(-1).numpy()
    return float((pred == y_test).mean())


def onehot_a(a: np.ndarray) -> np.ndarray:
    return np.eye(4, dtype=np.float32)[a]


def t1_action_sensitivity(model, rows: list[dict], device, *, n_obs: int = 16) -> dict:
    """固定观测、扫 4 个 a_prev，量 Δz_hat 与 cosine(z_hat, z)。"""
    model.eval()
    b = _assemble(rows[:n_obs], device)
    sens = []
    cos_zhat_z = []
    with torch.no_grad():
        for t in range(min(6, b["obs"].shape[1])):
            obs_t = b["obs"][:, t]
            state = model.initial_state(b["obs"].shape[0], device)
            # 重放 t-1 步建立真实状态
            for tt in range(t):
                out_t, state = model.step(b["obs"][:, tt], state, b["actions"][:, tt], depth=1)
            base = model.encoder(obs_t, None)
            z_hats = []
            for a in range(4):
                a_idx = torch.full((b["obs"].shape[0],), a, dtype=torch.long, device=device)
                out_a, _st = model.step(obs_t, state, a_idx, depth=1)
                z_hats.append(out_a.z_hat)
            H = torch.stack(z_hats)                       # [4, B, z]
            spread = (H.max(0).values - H.min(0).values).norm(dim=-1)
            ref = H[0].norm(dim=-1).clamp_min(1e-6)
            sens += (spread / ref).tolist()
            # cosine(z_hat, z)：同一输入下与目标的相关（拷贝检验）
            zc = torch.nn.functional.normalize(base, dim=-1)
            hc = torch.nn.functional.normalize(H.mean(0), dim=-1)
            cos_zhat_z += (zc * hc).sum(-1).tolist()
    return {"action_sensitivity_mean": round(float(np.mean(sens)), 4),
            "action_sensitivity_p95": round(float(np.percentile(sens, 95)), 4),
            "cosine_zhat_vs_z_mean": round(float(np.mean(cos_zhat_z)), 4),
            "n_points": len(sens)}


def eval_ckpt(name: str, ckpt: Path, *, seed: int, skip_posthoc: bool = False) -> dict:
    cfg = default_config()
    model, _meta = load_checkpoint(ckpt, cfg=cfg, device="cpu", allow_config_mismatch=True)
    model.eval()
    device = torch.device("cpu")
    data = make_data(cfg, D.GridTask(goal_visible_steps=4, seed=seed), seed=seed)
    tr = collect_features(model, data["train"], device)
    te = collect_features(model, data["test"], device)
    y_tr, y_te = pos_to_class(tr["pos1"]), pos_to_class(te["pos1"])

    # T1/T2
    sens = t1_action_sensitivity(model, data["test"][:16], device)
    cos = float(np.mean(np.sum(
        tr["z_hat"] / np.maximum(np.linalg.norm(tr["z_hat"], axis=1, keepdims=True), 1e-6)
        * tr["z"] / np.maximum(np.linalg.norm(tr["z"], axis=1, keepdims=True), 1e-6), axis=1)))

    # T3 读出阶梯
    za = np.concatenate([tr["z"], onehot_a(tr["a"])], axis=1)
    ha = np.concatenate([tr["z_hat"], onehot_a(tr["a"])], axis=1)
    za_te = np.concatenate([te["z"], onehot_a(te["a"])], axis=1)
    ha_te = np.concatenate([te["z_hat"], onehot_a(te["a"])], axis=1)
    readouts = {
        "R1_z_only(copy_baseline)": fit_readout(tr["z"], y_tr, te["z"], y_te),
        "R2_z_plus_action(oracle_equiv)": fit_readout(za, y_tr, za_te, y_te),
        "R3_zhat_only": fit_readout(tr["z_hat"], y_tr, te["z_hat"], y_te),
        "R4_zhat_plus_action": fit_readout(ha, y_tr, ha_te, y_te),
        "R5_zhat_shuffled_action(neg_control)": fit_readout(
            np.concatenate([tr["z_hat"], onehot_a(np.random.default_rng(1).permutation(tr["a"]))],
                           axis=1), y_tr,
            np.concatenate([te["z_hat"], onehot_a(te["a"])], axis=1), y_te),
    }

    # T4 多数类位移基线：全局最常见 (dx,dy)
    deltas = te["pos1"] - te["pos"]
    keys = deltas[:, 0] * 8 + deltas[:, 1]
    vals, counts = np.unique(keys, return_counts=True)
    majority_delta = vals[np.argmax(counts)]
    base_xy = np.array([int(majority_delta // 8), int(majority_delta % 8)])
    pred_majority = np.clip(te["pos"] + base_xy[None, :], 0, 5)
    majority_acc = float((pos_to_class(pred_majority) == y_te).mean())
    copy_acc = float((pos_to_class(te["pos"]) == y_te).mean())     # 永远抄当前位置

    validated = (readouts["R4_zhat_plus_action"] > max(readouts["R1_z_only(copy_baseline)"],
                                                       copy_acc, majority_acc) + 0.05
                 and sens["action_sensitivity_mean"] >= 0.05)
    return {
        "arm": name, "checkpoint": str(ckpt), "seed": seed,
        "n_train_points": int(len(y_tr)), "n_test_points": int(len(y_te)),
        "T1_action_sensitivity": sens,
        "T2_cosine_zhat_vs_z": round(cos, 4),
        "T3_readout_next_pos_acc": readouts,
        "T4_negative_controls": {"majority_delta_acc": round(majority_acc, 4),
                                 "copy_current_acc": round(copy_acc, 4)},
        "world_model_validated": bool(validated),
        "validation_rule": "R4 > max(R1, copy, majority)+0.05 且 T1 敏感性 ≥0.05（预登记）",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", default="C_sparse_ce_s20261121,F1_detach_s20261121,F10_ste_s20261121")
    ap.add_argument("--seed", type=int, default=20261121)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for name in [x.strip() for x in args.ckpts.split(",") if x.strip()]:
        ckpt = SRC / f"{name}.pt"
        if not ckpt.exists():
            print(f"[skip] {ckpt} 不存在", flush=True)
            continue
        t0 = time.perf_counter()
        r = eval_ckpt(name, ckpt, seed=args.seed)
        r["wall_s"] = round(time.perf_counter() - t0, 1)
        results.append(r)
        print(f"[{name}] T1 sens={r['T1_action_sensitivity']['action_sensitivity_mean']} "
              f"cos(z_hat,z)={r['T2_cosine_zhat_vs_z']} "
              f"R1={r['T3_readout_next_pos_acc']['R1_z_only(copy_baseline)']} "
              f"R2={r['T3_readout_next_pos_acc']['R2_z_plus_action(oracle_equiv)']} "
              f"R3={r['T3_readout_next_pos_acc']['R3_zhat_only']} "
              f"R4={r['T3_readout_next_pos_acc']['R4_zhat_plus_action']} "
              f"→ validated={r['world_model_validated']} ({r['wall_s']}s)", flush=True)

    import os
    payload = {"captured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
               "git_commit": os.popen("git rev-parse HEAD").read().strip(),
               "device": "cpu", "seed": args.seed,
               "verdict_rule": "见各条目 validation_rule；预登记于脚本 docstring",
               "runs": results}
    dest = OUT / "pred_eval.json"
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"→ {dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
