"""反馈缓冲区：收集模型自主输出与解释器反馈，形成后续学习样本。

注意：监督样本使用解释器标准答案，但绝不把解释器答案伪装成模型输出。
执行反馈学习（奖励/错误信号）接口预留，本轮不做开放式自动训练。
"""
from __future__ import annotations

import json
import random
from collections import deque
from pathlib import Path

from environment import Feedback


class FeedbackBuffer:
    def __init__(self, capacity=4096):
        self.supervised: deque = deque(maxlen=capacity)
        self.feedback: deque[Feedback] = deque(maxlen=capacity)

    def add_supervised(self, prompt: str, answer: str, source: str = "generator"):
        """标准答案由确定性生成器/解释器给出（source 记录来源，可审计）。"""
        self.supervised.append({"prompt": prompt, "answer": answer, "source": source})

    def add_feedback(self, fb: Feedback):
        self.feedback.append(fb)

    def sample_supervised(self, n: int, rng: random.Random):
        if not self.supervised:
            return []
        return rng.sample(list(self.supervised), min(n, len(self.supervised)))

    def stats(self) -> dict:
        n = max(1, len(self.feedback))
        return {
            "supervised_count": len(self.supervised),
            "feedback_count": len(self.feedback),
            "correct_rate": round(sum(f.correct for f in self.feedback) / n, 4),
            "syntax_valid_rate": round(sum(f.syntactically_valid for f in self.feedback) / n, 4),
            "executed_rate": round(sum(f.executed for f in self.feedback) / n, 4),
        }

    def save(self, path):
        Path(path).write_text(json.dumps({
            "supervised": list(self.supervised),
            "feedback": [f.__dict__ for f in self.feedback],
            "stats": self.stats(),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
