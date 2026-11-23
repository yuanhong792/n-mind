from pathlib import Path

from nmind.v3.events import EventLog
from nmind.v3.tasks import TaskService
from nmind.v3.facade import V3Facade


def test_task_state_machine_persists(tmp_path: Path):
    events = EventLog(tmp_path / "events.db")
    tasks = TaskService(tmp_path / "tasks.db", events)
    task = tasks.create("demo", [{"id": "a", "depends_on": []}])
    assert tasks.transition(task.task_id, "running").status == "running"
    assert tasks.transition(task.task_id, "paused").status == "paused"
    assert tasks.transition(task.task_id, "running").status == "running"
    assert tasks.transition(task.task_id, "completed").status == "completed"
    assert tasks.get(task.task_id).status == "completed"
    assert len(events.list()) >= 4


def test_v3_learning_is_hard_disabled(tmp_path: Path):
    facade = V3Facade(tmp_path)
    result = facade.learn_once()
    assert result["optimizer_steps"] == 0
    assert result["network_requests"] == 0
    assert facade.status()["training_enabled"] is False


def test_invalid_task_transition_is_rejected(tmp_path: Path):
    events = EventLog(tmp_path / "events.db")
    tasks = TaskService(tmp_path / "tasks.db", events)
    task = tasks.create("demo", [])
    try:
        tasks.transition(task.task_id, "completed")
    except ValueError:
        pass
    else:
        raise AssertionError("queued task must not skip execution state")
