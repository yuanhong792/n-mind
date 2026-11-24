"""跨重启共享的训练预算台账。

每次 optimizer.step 前登记；异常中断无法确认时保守计入已消耗。
预算：累计 2048 step / 1800s 墙钟，所有子进程与重启共享。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

LEDGER = Path(__file__).resolve().parent / "C11_BUDGET_LEDGER.jsonl"
MAX_STEPS = 2048
MAX_WALL = 1800
RESERVE = 60


class BudgetManager:
    def __init__(self, ledger=LEDGER, max_steps=MAX_STEPS, max_wall=MAX_WALL):
        self.ledger = Path(ledger)
        self.max_steps = max_steps
        self.max_wall = max_wall
        self.start_wall = None
        self._load()

    def _load(self):
        self.steps_used = 0
        self.wall_used = 0.0
        self.sessions = 0
        if self.ledger.exists():
            for line in self.ledger.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    self.steps_used += 1  # 保守计入无法解析的条目
                    continue
                if e.get("type") == "session_start":
                    self.sessions += 1
                    self.wall_used += float(e.get("carried_wall", 0.0))
                elif e.get("type") == "step":
                    self.steps_used += int(e.get("steps", 1))
                elif e.get("type") == "uncertain":
                    self.steps_used += int(e.get("steps", 1))  # 保守
                elif e.get("type") == "session_end":
                    self.wall_used = max(self.wall_used, float(e.get("total_wall", self.wall_used)))

    def start_session(self):
        self.start_wall = time.perf_counter()
        self.sessions += 1
        with open(self.ledger, "a", encoding="utf-8") as f:
            f.write(json.dumps({"type": "session_start", "ts": time.time(),
                                "carried_steps": self.steps_used,
                                "carried_wall": round(self.wall_used, 2)}) + "\n")

    def current_wall(self):
        if self.start_wall is None:
            return self.wall_used
        return self.wall_used + (time.perf_counter() - self.start_wall)

    def can_step(self) -> tuple[bool, str]:
        if self.steps_used >= self.max_steps:
            return False, "step_budget"
        if self.current_wall() > self.max_wall - RESERVE:
            return False, "wall_budget"
        return True, "ok"

    def record_step(self, step_index, loss=None):
        self.steps_used += 1
        with open(self.ledger, "a", encoding="utf-8") as f:
            f.write(json.dumps({"type": "step", "global_step": self.steps_used,
                                "local_step": step_index,
                                "wall": round(self.current_wall(), 2),
                                "loss": round(float(loss), 5) if loss is not None else None}) + "\n")

    def record_uncertain(self, n=1, reason=""):
        """异常中断保守计入。"""
        self.steps_used += n
        with open(self.ledger, "a", encoding="utf-8") as f:
            f.write(json.dumps({"type": "uncertain", "steps": n, "reason": reason,
                                "ts": time.time()}) + "\n")

    def end_session(self, reason="done"):
        total = self.current_wall()
        self.wall_used = total
        self.start_wall = None  # 防止 summary() 再次叠加本次会话耗时
        with open(self.ledger, "a", encoding="utf-8") as f:
            f.write(json.dumps({"type": "session_end", "reason": reason,
                                "total_wall": round(total, 2),
                                "total_steps": self.steps_used, "ts": time.time()}) + "\n")

    def summary(self):
        return {"steps_used": self.steps_used, "max_steps": self.max_steps,
                "steps_remaining": max(0, self.max_steps - self.steps_used),
                "wall_used": round(self.current_wall(), 2), "max_wall": self.max_wall,
                "sessions": self.sessions}
