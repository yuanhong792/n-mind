"""训练损失：监督动作 + 潜在状态预测 + 防塌缩 + 路由辅助项。

各项都可单独关闭（权重置 0），消融实验靠权重而不是靠改代码。

z_hat[t] 的含义在 architecture.step 里定死了：**由 t-1 时刻的 z 和当时刻执行的
动作预测出来的、对 z[t] 的估计**。所以预测损失对齐的是 z_hat[:, 1:] 与 z[:, 1:]，
第 0 步没有前驱，必须掩掉，否则等于让网络去预测一个无条件的先验。

防塌缩用 latent 方差正则（对每个维度求标准度，低于阈值才罚），
因为只最小化 MSE 的最优解是"所有输入编码成同一个向量"—— 那会让预测损失为 0
而能力为零。这条不是可选项，是默认开的。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def action_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """logits [B,T,A]，targets [B,T] 的交叉熵。"""
    b, t, a = logits.shape
    if targets.shape != (b, t):
        raise ValueError(f"targets 应为 {(b, t)}，实际 {tuple(targets.shape)}")
    return F.cross_entropy(logits.reshape(-1, a), targets.reshape(-1))


def prediction_loss(z_hat: torch.Tensor, z: torch.Tensor, *, detach_target: bool = False) -> torch.Tensor:
    """MSE(z_hat[:,1:], z[:,1:])；T<2 时返回 0 并保持计算图。"""
    if z_hat.shape != z.shape:
        raise ValueError(f"形状不匹配：z_hat {tuple(z_hat.shape)} vs z {tuple(z.shape)}")
    if z.shape[1] < 2:
        return z.new_zeros(())
    target = z[:, 1:].detach() if detach_target else z[:, 1:]
    return F.mse_loss(z_hat[:, 1:], target)


def variance_regularizer(z: torch.Tensor, *, target_std: float = 0.25) -> torch.Tensor:
    """只惩罚"过于平坦"的维度：relu(target_std - std)。塌缩时该项迅速变大。"""
    std = z.reshape(-1, z.shape[-1]).std(dim=0)
    return F.relu(target_std - std).mean()


def cosine_error(z_hat: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    if z.shape[1] < 2:
        return z.new_zeros(())
    a = F.normalize(z_hat[:, 1:], dim=-1)
    b = F.normalize(z[:, 1:], dim=-1)
    return (1.0 - (a * b).sum(-1)).mean()


def balance_loss(probs: torch.Tensor, *, n_blocks: int) -> torch.Tensor:
    """Shazeer 式负载均衡：整体负载方差 × 块数，越小越均衡。probs [.., N]。"""
    load = probs.mean(dim=tuple(range(probs.dim() - 1)))
    return load.var(unbiased=False) * float(n_blocks)


def entropy_loss(probs: torch.Tensor) -> torch.Tensor:
    """取负熵的相反数：最小化本函数 = 最大化路由熵 = 抑制"永远只选同几块"。"""
    p = probs.clamp_min(1e-9)
    ent = -(p * p.log()).sum(-1).mean()
    return -ent / float(torch.log(torch.tensor(float(p.shape[-1]))))


@dataclass
class LossWeights:
    pred: float = 1.0
    var: float = 0.05
    balance: float = 0.1
    entropy: float = 0.01
    probe: float = 0.25
    target_std: float = 0.25
    detach_pred_target: bool = False

    @classmethod
    def from_cfg(cls, cfg) -> "LossWeights":
        l = cfg.learn
        return cls(pred=l.pred_weight, var=l.var_reg_weight, balance=l.aux_balance_weight,
                   entropy=l.entropy_weight, probe=l.probe_weight, target_std=l.target_std,
                   detach_pred_target=l.detach_pred_target)


def compute_total(model_out: dict, actions: torch.Tensor, weights: LossWeights,
                  pred_shift: bool = False) -> dict[str, torch.Tensor]:
    """把 forward_sequence 的输出 + 目标动作打包成一个可反传总损失与各分量。

    pred_shift=True（pred_mode="next"）时，预测对齐改为 z_hat[:, :-1] → z[:, 1:]，
    即"用当前观测与当前动作预测下一潜状态"的真世界模型目标；False 保持历史
    对齐（z_hat[t]→z[t]，预测器把目标当输入，已实测学成恒等拷贝）。

    动作标签由调用方显式传入，不从 model_out 里"顺便"取 —— 避免把预测输出
    当成标签的自监督假象。探针损失不在此处（它需要探针模块与跨任务目标码，
    由 train.py 负责加权后并入 total）。
    """
    acts = action_loss(model_out["action_logits"], actions)
    if pred_shift:
        preds = prediction_loss(model_out["z_hat"][:, :-1], model_out["z"][:, 1:],
                                detach_target=weights.detach_pred_target)
    else:
        preds = prediction_loss(model_out["z_hat"], model_out["z"],
                                detach_target=weights.detach_pred_target)
    var = variance_regularizer(model_out["z"], target_std=weights.target_std)
    probs = model_out["route_probs"]
    bal = balance_loss(probs, n_blocks=probs.shape[-1])
    ent = entropy_loss(probs)
    total = (acts + weights.pred * preds + weights.var * var
             + weights.balance * bal + weights.entropy * ent)
    return {"total": total, "action": acts.detach(), "pred": preds.detach(),
            "var": var.detach(), "balance": bal.detach(), "entropy": ent.detach()}
