from nmind.v3.facade import V3Facade


def test_v33_permission_and_explainable_goal(tmp_path):
    f = V3Facade(tmp_path)
    assert f.v33_status()["enabled"] is False
    f.set_learning(True)
    goals = f.v33_goals()["goals"]
    assert goals and goals[0]["verification_method"] == "deterministic_oracle"
    assert goals[0]["constraints"]["offline"] is True


def test_v33_run_persists_review_and_zero_training(tmp_path):
    f = V3Facade(tmp_path); f.set_learning(True)
    out = f.v33_run_once()
    assert out["status"] == "completed"
    assert out["verification_source"] == "deterministic_oracle"
    assert out["optimizer_steps"] == 0
    assert f.v33_reviews()["reviews"]
    f.close()
    f2 = V3Facade(tmp_path)
    assert f2.v33_reviews()["reviews"]
    assert f2.v33_status()["optimizer_steps"] == 0


def test_v33_pause_resume_stop_are_persistent_controls(tmp_path):
    f = V3Facade(tmp_path); f.set_learning(True)
    assert f.v33_pause()["state"] == "paused"
    assert f.v33_resume()["state"] == "idle"
    assert f.v33_stop()["state"] == "idle"
