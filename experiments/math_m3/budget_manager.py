"""M3 全局共享预算台账：4096 optimizer.step / 3600s，跨重启不重置。

异常中断保守计入不确定步骤。所有初筛/失败试训/梯度探针产生的 step 均累计。
"""

import json
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LEDGER = HERE / "M3_BUDGET_LEDGER.jsonl"
MAX_STEPS = 4096
MAX_WALL = 3600
RESERVE = 120


class BudgetManager:
    def __init__(self, ledger=LEDGER, max_steps=MAX_STEPS, max_wall=MAX_WALL):
        self.ledger = Path(ledger)
        self.max_steps, self.max_wall = max_steps, max_wall
        self.start_wall = None
        self._load()

    def _load(self):
        self.steps_used = 0; self.wall_used = 0.0; self.sessions = 0; self.runs = []
        if not self.ledger.exists():
            return
        for line in self.ledger.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                self.steps_used += 1; continue
            t = e.get("type")
            if t == "session_start":
                self.sessions += 1
                self.wall_used += float(e.get("carried_wall", 0.0))
            elif t in ("step", "uncertain"):
                self.steps_used += int(e.get("steps", 1))
            elif t == "run_end":
                self.runs.append({k: e.get(k) for k in
                                  ("run_id", "architecture", "seed", "steps", "wall", "stop_reason")})
            elif t == "session_end":
                self.wall_used = max(self.wall_used, float(e.get("total_wall", self.wall_used)))

    def start_session(self):
        self.start_wall = time.perf_counter()
        self.sessions += 1
        self._append({"type": "session_start", "ts": time.time(),
                      "carried_steps": self.steps_used, "carried_wall": round(self.wall_used, 2)})

    def current_wall(self):
        return self.wall_used if self.start_wall is None else self.wall_used + (time.perf_counter() - self.start_wall)

    def can_step(self):
        if self.steps_used >= self.max_steps:
            return False, "step_budget"
        if self.current_wall() > self.max_wall - RESERVE:
            return False, "wall_budget"
        return True, "ok"

    def record_step(self, run_id, arch, seed, local_step, loss=None):
        self.steps_used += 1
        self._append({"type": "step", "global_step": self.steps_used, "run_id": run_id,
                      "architecture": arch, "seed": seed, "local_step": local_step,
                      "wall": round(self.current_wall(), 2),
                      "loss": round(float(loss), 5) if loss is not None else None})

    def record_uncertain(self, n=1, reason=""):
        self.steps_used += n
        self._append({"type": "uncertain", "steps": n, "reason": reason, "ts": time.time()})

    def record_run_end(self, run_id, arch, seed, steps, wall, stop_reason, best_score=None):
        self._append({"type": "run_end", "run_id": run_id, "architecture": arch, "seed": seed,
                      "steps": steps, "wall": round(wall, 2), "stop_reason": stop_reason,
                      "best_score": best_score, "ts": time.time()})

    def end_session(self, reason="done"):
        total = self.current_wall(); self.wall_used = total; self.start_wall = None
        self._append({"type": "session_end", "reason": reason,
                      "total_wall": round(total, 2), "total_steps": self.steps_used, "ts": time.time()})

    def _append(self, obj):
        with open(self.ledger, "a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def summary(self):
        return {"steps_used": self.steps_used, "max_steps": self.max_steps,
                "steps_remaining": max(0, self.max_steps - self.steps_used),
                "wall_used": round(self.current_wall(), 2), "max_wall": self.max_wall,
                "sessions": self.sessions, "runs": self.runs}
