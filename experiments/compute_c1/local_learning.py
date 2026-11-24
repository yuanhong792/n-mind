"""受控局部学习接口。

模式 A（监督学习）：解释器标准答案 → 神经权重更新（本轮可用）。
模式 B（执行反馈学习）：模型程序生成 → 执行奖励 → 权重更新（仅接口/测试，
未经额外授权不运行开放式自动训练循环）。
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from model_compute import ComputeLM
from feedback_buffer import FeedbackBuffer


@dataclass
class LearnConfig:
    lr: float = 1e-4
    max_steps: int = 32
    batch_size: int = 8
    clip: float = 1.0


class LocalLearner:
    def __init__(self, model: ComputeLM, tok, cfg: LearnConfig | None = None):
        self.model = model
        self.tok = tok
        self.cfg = cfg or LearnConfig()
        self.opt = torch.optim.AdamW(model.parameters(), lr=self.cfg.lr)
        self.steps = 0

    def supervised_step(self, batch: list[dict]) -> float | None:
        """模式 A：一批监督样本更新一次。样本 prompt/answer 必须在 buffer 中标记来源。"""
        if not batch:
            return None
        seqs = [self.tok.encode(b["prompt"]) + self.tok.encode(b["answer"]) + [2] for b in batch]
        n = max(len(s) for s in seqs)
        x = torch.zeros((len(seqs), n), dtype=torch.long)
        mask = torch.zeros((len(seqs), n - 1), dtype=torch.bool)
        for j, s in enumerate(seqs):
            x[j, :len(s)] = torch.tensor(s)
            start = len(self.tok.encode(batch[j]["prompt"])) - 1
            for p in range(start, len(s) - 1):
                mask[j, p] = True
        self.model.train()
        logits, _ = self.model(x[:, :-1])
        loss = torch.nn.functional.cross_entropy(
            logits[mask].reshape(-1, logits.size(-1)), x[:, 1:][mask].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.clip)
        self.opt.step(); self.opt.zero_grad(set_to_none=True)
        self.steps += 1
        return float(loss.detach())

    def feedback_update(self, buffer: FeedbackBuffer) -> dict:
        """模式 B 接口：只计算奖励统计，不执行任何权重更新（需额外授权）。"""
        return {"status": "INTERFACE_ONLY_NO_UPDATE",
                "reason": "执行反馈学习需单独授权；本轮仅监督闭环",
                "buffer_stats": buffer.stats(),
                "optimizer_steps": self.steps}
