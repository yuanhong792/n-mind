"""算子 M2-SM：固定容量序列记忆（Sequential Memory）与神经读写。

状态 H_seq: [B, L, d]，L 为固定容量（如 16/32）。每个时间步：
  - 写入指针 p_w（来自 differentiable_pointer）决定写到哪些槽位；
  - 读取指针 p_r 决定从哪里有序读取；
  - 提供 shift 关联：读取指针随生成步向后移动，形成连续有序输出。
写入：H_new = H*(1 - p_w⊙erase) + p_w ⊗ v
读取：r = p_r · H
L2 归一与门控保证数值有界；逐样本独立。
"""
from __future__ import annotations

import torch
from torch import nn

from differentiable_pointer import pointer_read, pointer_write, shift_pointer, l2_normalize


def seq_write(H: torch.Tensor, p_w: torch.Tensor, v: torch.Tensor,
              erase: torch.Tensor | None = None) -> torch.Tensor:
    return pointer_write(H, p_w, v, erase)


def seq_read(H: torch.Tensor, p_r: torch.Tensor) -> torch.Tensor:
    return pointer_read(H, p_r)


def advance_pointer(p: torch.Tensor, step_gate: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    """按 step_gate 强度把读取指针向后平移一位（循环），用于有序连续读取。"""
    K = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    k = torch.zeros(p.size(0), K, device=p.device, dtype=p.dtype)
    mid = K // 2
    # 读取指针向后推进一位（槽 0→1→2...）：torch.roll(shifts=1) 使新 p[j]=旧 p[j-1]
    shifted = torch.roll(p, shifts=1, dims=-1)
    return (1.0 - step_gate[:, None]) * p + step_gate[:, None] * shifted


class SequentialMemoryOperator(nn.Module):
    """最小可训练序列记忆算子封装。"""

    def __init__(self, L: int, d: int):
        super().__init__()
        self.L, self.d = L, d
        self.erase_logit = nn.Parameter(torch.zeros(()))

    def initial(self, batch: int, device):
        return torch.zeros(batch, self.L, self.d, device=device)

    def forward(self, H, p_w, v, p_r):
        erase = torch.sigmoid(self.erase_logit).expand(v.size(0))
        H_new = seq_write(H, p_w, v, erase)
        return H_new, seq_read(H_new, p_r)
