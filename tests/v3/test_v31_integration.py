from nmind.v3.facade import V3Facade

def test_learning_cycle_has_oracle_and_zero_training(tmp_path):
    f = V3Facade(tmp_path); out = f.learn_once()
    assert out["state"] == "completed"
    assert out["oracle_result"] == "verified"
    assert out["source"] == "tool/oracle"
    assert out["neural_attempt"] is None
    assert out["optimizer_steps"] == 0
    assert f.growth()["sources"]["neural"] == 0

def test_permission_and_event_audit(tmp_path):
    f = V3Facade(tmp_path)
    assert f.permissions()["network_enabled"] is False
    assert f.set_learning(True)["enabled"] is True
    assert any(e["event_type"] == "learning.permission_changed" for e in f.events.list())

def test_task_cancel_is_terminal(tmp_path):
    f = V3Facade(tmp_path); task = f.tasks.create("safe", [{"id": "one"}])
    f.tasks.transition(task.task_id, "cancelled")
    try: f.tasks.transition(task.task_id, "running")
    except ValueError: pass
    else: raise AssertionError("cancelled task must remain terminal")
