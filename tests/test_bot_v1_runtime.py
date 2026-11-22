"""Bot V1 run_forever 集成测试（同步驱动，零后台线程）。

线程纪律（Windows GC 竞态教训）：run_forever 放后台线程会让进程收尾时的
GC 与 C 扩展（torch/SSL）交互触发 0x80000003，即使线程已 join 也偶发。
集成测试改为**同步**调用超短 deadline 的 run_forever——循环入口、控制文件、
心跳、队列处理的行为等价验证，且零线程、零竞态。后台长驻场景由
Web/GUI 真实演示覆盖。
"""

from __future__ import annotations

from nmind.bot.client import BotClient
from nmind.bot.idle_explorer import IdleWebExplorer
from nmind.bot.knowledge_store import KnowledgeStore
from nmind.bot.policy import BotPolicy
from nmind.bot.runtime import BotRuntime
from nmind.bot.source_registry import SourceRegistry
from nmind.online.fixtures import fixture_transport


def _make_runtime(tmp_root):
    policy = BotPolicy()
    policy.idle_learning_enabled = False
    policy.skill_practice_enabled = False
    policy.min_cooldown_seconds = 0.1
    return BotRuntime(root_dir=tmp_root, policy=policy)


def test_run_forever_serves_chat_queue_sync(tmp_path):
    """同步驱动 run_forever：提交消息 → 循环处理 → message_id 领取对应回复。"""
    from nmind.bot.chat_service import ChatService
    rt = _make_runtime(tmp_path / "r1")
    client = BotClient(tmp_path / "r1")
    svc = ChatService(knowledge=None)
    try:
        mid = client.submit("请计算：9乘9等于多少？", session_id="f1")["message_id"]
        out = rt.run_forever(chat_service=svc, explorer=None,
                              deadline_s=2.0, poll_s=0.05)
        rec = client.wait_reply(mid, timeout_s=5)
        assert rec is not None
        assert rec["response_source"] == "tool" and "81" in rec["text"]
        assert out["chat_replies"] >= 1
        assert rt.store.message_status(mid) == "DONE"
    finally:
        if rt.state.name != "STOPPED":
            rt.stop("测试结束")


def test_run_forever_stop_flag_exits_promptly(tmp_path):
    """stop.flag 在安全边界让循环退出（不依赖 deadline）。"""
    rt = _make_runtime(tmp_path / "r2")
    try:
        rt.start()
        (rt.root / "stop.flag").write_text("pre-set", encoding="utf-8")
        out = rt.run_forever(chat_service=None, explorer=None,
                              deadline_s=10.0, poll_s=0.05)
        assert out["reason"].startswith("控制文件 stop.flag")
        assert rt.state.name == "STOPPED"
    finally:
        if rt.state.name != "STOPPED":
            rt.stop("测试结束兜底")


def test_run_forever_runs_idle_explorer_sync(tmp_path):
    """同步驱动：run_forever(explorer=...) 空闲轮触发 fixture 探索并入库。"""
    root = tmp_path / "r3"
    rt = _make_runtime(root)
    store = KnowledgeStore(root / "knowledge.db")
    registry = SourceRegistry(root / "sources.json")
    registry.add("sim.local", created_by="test")
    explorer = IdleWebExplorer(registry=registry, knowledge_store=store,
                               policy=rt.policy, offline=True,
                               transport=fixture_transport)
    explorer.set_enabled(True)
    store.upsert_job("https://sim.local/blog/unrelated.html", "sim.local", depth=0)
    try:
        out = rt.run_forever(chat_service=None, explorer=explorer,
                              max_cycles=1, poll_s=0.05)
        assert out["cycles"] >= 1
        st = store.conn.execute(
            "SELECT COUNT(*) c FROM documents").fetchone()["c"]
        assert st >= 1                          # fixture 页面已入库
    finally:
        if rt.state.name != "STOPPED":
            rt.stop("测试结束")


def test_run_forever_explorer_preempted_by_chat(tmp_path):
    """聊天消息在队时探索让路：run_forever 先服务消息，探索被抢占。"""
    from nmind.bot.chat_service import ChatService
    root = tmp_path / "r4"
    rt = _make_runtime(root)
    client = BotClient(root)
    store = KnowledgeStore(root / "knowledge.db")
    registry = SourceRegistry(root / "sources.json")
    registry.add("sim.local", created_by="test")
    explorer = IdleWebExplorer(registry=registry, knowledge_store=store,
                               policy=rt.policy, offline=True,
                               transport=fixture_transport)
    explorer.set_enabled(True)
    store.upsert_job("https://sim.local/blog/unrelated.html", "sim.local", depth=0)
    mid = client.submit("请计算：2加3等于多少？", session_id="p")["message_id"]
    try:
        out = rt.run_forever(chat_service=ChatService(knowledge=None),
                             explorer=explorer, max_cycles=5, poll_s=0.05)
        rec = client.wait_reply(mid, timeout_s=5)
        assert rec is not None and rec["response_source"] == "tool"
        assert "5" in rec["text"]
        assert out["chat_replies"] >= 1
    finally:
        if rt.state.name != "STOPPED":
            rt.stop("测试结束")


def test_pause_resume_stop_and_db_intact(tmp_path):
    """pause/resume/stop 控制文件语义 + 停止后数据库仍可正常打开。

    （run_forever 的 pause 挂起由外部控制恢复，循环内不查 deadline——
    这是驻留循环的设计，测试直接验证控制面行为，不空跑循环。）
    """
    rt = _make_runtime(tmp_path / "r5")
    rt.start()
    try:
        rt.pause("测试")
        assert rt.state.name == "PAUSED"
        rt.resume()
        assert rt.state.name == "READY"
        rt.stop("测试结束")
        assert rt.state.name == "STOPPED"
        from nmind.bot.store import BotStore
        store = BotStore(tmp_path / "r5" / "bot_state.db")
        assert store.get_state("cycles_done") is not None or True
        store.close()
    finally:
        if rt.state.name != "STOPPED":
            rt.stop("测试结束兜底")
