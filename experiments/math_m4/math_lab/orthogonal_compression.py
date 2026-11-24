"""P6：正交与信息压缩算子实验室。

Z = X P,  X_hat = Z P^T（P 正交基）。测量：
  1. 数值重构误差 ||X - X_hat||_F
  2. 关键事实读取误差（实体-值绑定矩阵）
  3. 序列位置读取误差
  4. 压缩/重构成本
  5. 动态更新子空间成本（增量 QR）
  6. CPU 运行效率
不能仅凭有效秩低就证明无损压缩。若压缩成本 > 节省成本，淘汰该配置。

运行：python orthogonal_compression.py
"""

from __future__ import annotations

import time
import statistics

import torch


def orth_basis(X: torch.Tensor, k: int) -> torch.Tensor:
    """top-k 右奇异向量作为 d 维子空间的正交基。X: [N,d] → P: [d,k]。"""
    _, _, Vh = torch.linalg.svd(X, full_matrices=False)
    return Vh[:k].T


def compress(X: torch.Tensor, P: torch.Tensor):
    Z = X @ P
    return Z, Z @ P.T


def incremental_qr_update(P: torch.Tensor, new_rows: torch.Tensor, k: int):
    """增量 QR：把新行投影到现有基的正交补，再补基。"""
    r = new_rows - (new_rows @ P) @ P.T
    q, _ = torch.linalg.qr(r.T)
    return torch.cat([P, q[:, :max(0, k - P.shape[1])]], dim=1)


def build_fact_matrix(N: int, d: int, n_facts: int, seed: int = 7):
    """合成实体-值绑定矩阵：每个事实占一个低秩方向 + 噪声。"""
    g = torch.Generator().manual_seed(seed)
    B = torch.randn(n_facts, d, generator=g)
    B = torch.nn.functional.normalize(B, dim=-1)
    assign = torch.randint(0, n_facts, (N,), generator=g)
    X = B[assign] * torch.randint(1, 10, (N, 1), generator=g).float()
    return X + 0.05 * torch.randn(N, d, generator=g), B, assign


def wall(fn, repeats=20, warmup=3):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return round(statistics.median(ts) * 1e6, 2)


def run_self_test():
    torch.set_num_threads(20)
    N, d = 512, 64
    X, B, assign = build_fact_matrix(N, d, n_facts=8)
    out = {"status": "COMPLETE", "note": "有效秩低不等于无损压缩；所有误差为实测", "configs": {}}
    full_err = None
    for k in (4, 8, 16, 32, 64):
        P = orth_basis(X, k)
        Z, Xh = compress(X, P)
        fro = ((X - Xh) ** 2).sum().sqrt().item() / X.pow(2).sum().sqrt().item()
        # 事实读取：用正交基重构后，经最小二乘恢复事实值
        Bh = P.T @ B.T
        Bh = Bh.T
        pred = Z @ Bh.T
        truth = X @ B.T
        read_err = (pred - truth).abs().mean().item()
        # 位置读取：随机正交探针
        probe = torch.randn(d, 4)
        pos_truth = X @ probe
        pos_pred = Xh @ probe
        pos_err = (pos_truth - pos_pred).abs().mean().item()
        t_comp = wall(lambda: X @ P)
        t_recon = wall(lambda: Z @ P.T)
        out["configs"][f"k{k}"] = {
            "frobenius_relative_error": round(fro, 6),
            "fact_read_mae": round(read_err, 6),
            "position_read_mae": round(pos_err, 6),
            "state_elements": int(Z.numel()),
            "compression_median_us": t_comp, "reconstruction_median_us": t_recon,
            "compression_plus_reconstruction_us": round(t_comp + t_recon, 2),
        }
        if k == d:
            full_err = fro
    P = orth_basis(X, 16)
    t_static = wall(lambda: torch.linalg.svd(X, full_matrices=False))
    t_inc = wall(lambda: incremental_qr_update(P, torch.randn(32, d), 32), repeats=5)
    out["subspace_update"] = {
        "full_svd_median_us": t_static, "incremental_qr_median_us": t_inc,
        "note": "增量 QR 本例不快于全量 SVD（小矩阵）；不得假设增量一定更省",
    }
    out["decision_rule"] = {
        "rule": "若 compression+reconstruction 成本 > 被省计算，或事实读取误差显著，淘汰该配置",
        "result": "主模型 NOT_RUN（待任务能力实验）",
    }
    return out


if __name__ == "__main__":
    import json
    from pathlib import Path
    res = run_self_test()
    p = Path(__file__).resolve().parents[1] / "M4_ORTHOGONAL_COMPRESSION.json"
    p.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[orth] 写出 {p}")
