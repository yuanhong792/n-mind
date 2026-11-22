from nmind.memory import MemoryQuery, MemoryService, MemoryStore


def test_v2_observe_recall_survives_restart(tmp_path):
    db = tmp_path / "memory.db"
    store = MemoryStore(db)
    svc = MemoryService(store)
    mid = svc.observe(session_id="s1", task="files", observation="读取文件时先检查编码并使用 with open", memory_kind="procedural", confidence=.9, trust=.9)
    assert mid > 0
    assert svc.recall(MemoryQuery("文件 编码")).hits[0]["memory_kind"] == "procedural"
    store.close()
    store2 = MemoryStore(db)
    assert MemoryService(store2).recall("文件 编码").used


def test_v2_forget_is_tombstoned(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    svc = MemoryService(store)
    mid = svc.observe(session_id="s", observation="用户偏好短答案", memory_kind="user_preference", privacy_level="normal")
    assert svc.forget(kind="episode", memory_id=mid)
    assert not svc.recall("偏好 短答案").hits
    assert svc.observe(session_id="s", observation="用户偏好短答案", memory_kind="user_preference") == -1


def test_v2_sensitive_observation_not_persisted(tmp_path):
    store = MemoryStore(tmp_path / "memory.db")
    svc = MemoryService(store)
    assert svc.observe(session_id="s", observation="API key: secret-value") == -1
    assert store.stats()["episodes"] == 0
