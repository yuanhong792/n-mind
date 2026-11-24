"""M4 全局预算台账：4096 optimizer.step / 3600s，跨重启不重置。

正控制 256 步 + 四架构双种子正式对照 3840 步 = 4096 步。
失败训练、探针训练、性能测试中的实际梯度更新和重启共享预算。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[0]
LEDGER = HERE / "M4_BUDGET_LEDGER.jsonl"
MAX_STEPS = 4096
MAX_WALL = 3600
RESERVE = 180

STAGE_POSITIVE = 256
STAGE_MAIN = 3840


class BudgetManager:
    def __init__(self, ledger=LEDGER, max_steps=MAX_STEPS, max_wall=MAX_WALL):
        self.ledger = Path(ledger)
        self.max_steps, self.max_wall = max_steps, max_wall
        self.start_wall = None
        self._load()

    def _load(self):
        self.steps_used = 0
        self.wall_used = 0.0
        self.sessions = 0
        self.runs = []
        self.stages = {}
        if not self.ledger.exists():
            return
        for line in self.ledger.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                self.steps_used += 1
                continue
            t = e.get("type")
            if t == "session_start":
                self.sessions += 1
                self.wall_used += float(e.get("carried_wall", 0.0))
            elif t in ("step", "uncertain"):
                n = int(e.get("steps", 1))
                self.steps_used += n
                stage = e.get("stage", "unknown")
                self.stages[stage] = self.stages.get(stage, 0) + n
            elif t == "run_end":
                self.runs.append({k: e.get(k) for k in
                                  ("run_id", "architecture", "seed", "stage", "steps",
                                   "wall", "stop_reason", "best_score")})
            elif t == "session_end":
                self.wall_used = max(self.wall_used, float(e.get("total_wall", self.wall_used)))

    def start_session(self, note=""):
        self.start_wall = time.perf_counter()
        self.sessions += 1
        self._append({"type": "session_start", "ts": time.time(),
                      "carried_steps": self.steps_used,
                      "carried_wall": round(self.wall_used, 2), "note": note})

    def current_wall(self):
        return self.wall_used if self.start_wall is None else \
            self.wall_used + (time.perf_counter() - self.start_wall)

    def can_step(self, stage=None):
        if self.steps_used >= self.max_steps:
            return False, "step_budget"
        if stage == "positive" and self.stages.get("positive", 0) >= STAGE_POSITIVE:
            return False, "positive_stage_budget"
        if stage == "main" and self.stages.get("main", 0) >= STAGE_MAIN:
            return False, "main_stage_budget"
        if self.current_wall() > self.max_wall - RESERVE:
            return False, "wall_budget"
        return True, "ok"

    def record_step(self, run_id, arch, seed, local_step, loss=None, stage="main",
                    batch_info=None, macs=None):
        self.steps_used += 1
        self.stages[stage] = self.stages.get(stage, 0) + 1
        self._append({"type": "step", "global_step": self.steps_used, "run_id": run_id,
                      "architecture": arch, "seed": seed, "local_step": local_step,
                      "stage": stage, "wall": round(self.current_wall(), 2),
                      "loss": round(float(loss), 5) if loss is not None else None,
                      "batch": batch_info, "macs": macs})

    def record_uncertain(self, n=1, reason="", stage="unknown"):
        self.steps_used += n
        self.stages[stage] = self.stages.get(stage, 0) + n
        self._append({"type": "uncertain", "steps": n, "reason": reason,
                      "stage": stage, "ts": time.time()})

    def record_run_end(self, run_id, arch, seed, steps, wall, stop_reason, stage="main",
                       best_score=None, checkpoint_sha=None):
        self.runs.append({"run_id": run_id, "architecture": arch, "seed": seed,
                          "stage": stage, "steps": steps, "stop_reason": stop_reason})
        self._append({"type": "run_end", "run_id": run_id, "architecture": arch,
                      "seed": seed, "stage": stage, "steps": steps, "wall": round(wall, 2),
                      "stop_reason": stop_reason, "best_score": best_score,
                      "checkpoint_sha": checkpoint_sha, "ts": time.time()})

    def end_session(self, reason="done"):
        total = self.current_wall()
        self.wall_used = total
        self.start_wall = None
        self._append({"type": "session_end", "reason": reason,
                      "total_wall": round(total, 2), "total_steps": self.steps_used,
                      "ts": time.time()})

    def _append(self, obj):
        with self.ledger.open("a", encoding="utf-8") as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def summary(self):
        return {"steps_used": self.steps_used, "max_steps": self.max_steps,
                "steps_remaining": max(0, self.max_steps - self.steps_used),
                "wall_used": round(self.current_wall(), 2), "max_wall": self.max_wall,
                "stages": dict(self.stages), "sessions": self.sessions, "runs": self.runs}
