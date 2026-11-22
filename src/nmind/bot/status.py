"""机器可读状态快照（供 CLI bot status 与本地面板）。"""

from __future__ import annotations

import time
from typing import Any


def build_snapshot(*, state: str, pid: int | None, run_id: str, stable_sha: str,
                   candidate_status: str, cycles_done: int, queue_len: int,
                   budget_remaining: int, last_result: str | None,
                   capabilities: dict[str, Any], heartbeat_ts: str | None = None) -> dict[str, Any]:
    return {"state": state, "pid": pid, "run_id": run_id,
             "heartbeat_ts": heartbeat_ts or time.strftime("%Y-%m-%dT%H:%M:%S%z"),
             "stable_model_sha256": stable_sha, "candidate_status": candidate_status,
             "idle_cycles_done": cycles_done, "user_queue_len": queue_len,
             "budget_remaining_steps": budget_remaining,
             "last_learning_result": last_result, "capabilities": capabilities}
