"""受控计算环境：观察任务 → 神经输出 → 解释器验证 → 反馈。

严格隔离：解释器绝不代替神经网络产生预测；只在验证阶段给出标准答案/奖励。
"""
from __future__ import annotations

from dataclasses import dataclass
from minilang import MiniLang, MiniLangError


@dataclass
class Feedback:
    task: str
    prompt: str
    neural_answer: str
    expected: str
    correct: bool
    reward: float
    syntactically_valid: bool
    executed: bool
    error: str | None = None


class ComputeEnvironment:
    def __init__(self, max_steps=256):
        self.interp = MiniLang(max_steps=max_steps)

    def verify(self, row: dict, neural_answer: str) -> Feedback:
        """对比模型自由生成与标准答案；程序类任务额外做语法/执行验证。"""
        expected = row.get("answer", "")
        valid = True
        executed = False
        error = None
        task = row.get("task", "")
        if task.startswith("C5") or task.startswith("C6") or task.startswith("C7"):
            valid = self.interp.validate(neural_answer)
            if valid:
                res = self.interp.run(neural_answer)
                executed = res.error is None
                error = res.error
                correct = executed and res.output == expected
            else:
                correct = False
        else:
            correct = neural_answer.strip() == expected.strip()
        return Feedback(task=task, prompt=row.get("prompt", ""),
                        neural_answer=neural_answer, expected=expected,
                        correct=bool(correct), reward=1.0 if correct else 0.0,
                        syntactically_valid=valid, executed=executed, error=error)

    def batch_feedback(self, rows, answers):
        return [self.verify(r, a) for r, a in zip(rows, answers)]
