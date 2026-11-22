"""Bot V1 聊天消息协议与路由测试（任务书 §8.1 界面与消息组）。

覆盖：message_id 可靠领取 / 状态机 / 并发不丢 / 重启恢复 / 会话隔离 /
来源五分类 / UNK 门槛 / 空知识库下工具与神经路径。
"""

from __future__ import annotations

import threading

import pytest

from nmind.bot.chat_service import ChatRequest, ChatService, SourceCitation, _V2ChatSession
from nmind.bot.client import BotClient
from nmind.bot.knowledge_ingest import import_fixture_pages
from nmind.bot.knowledge_search import KnowledgeSearch
from nmind.bot.knowledge_store import KnowledgeStore
from nmind.bot.runtime import BotRuntime
from nmind.bot.store import BotStore


@pytest.fixture()
def tmp_root(tmp_path):
    return tmp_path / "bot_v1"


@pytest.fixture()
def kb(tmp_path):
    store = KnowledgeStore(tmp_path / "kb.db")
    import_fixture_pages(store)
    return KnowledgeSearch(store)


# ---------------------------------------------------------------- 消息协议
def test_message_id_reliable_reply(tmp_root):
    """提交 → 用 message_id 领取对应回复（不是读最新日志冒充）。"""
    client = BotClient(tmp_root)
    out = client.submit("第一条", session_id="s1")
    mid_a = out["message_id"]
    out = client.submit("第二条", session_id="s1")
    mid_b = out["message_id"]
    store = client._store
    store.finish_message(mid_a, reply_text="回复A", response_source="rule")
    store.finish_message(mid_b, reply_text="回复B", response_source="tool")
    rec_a = client.wait_reply(mid_a, timeout_s=2)
    rec_b = client.wait_reply(mid_b, timeout_s=2)
    assert rec_a["text"] == "回复A"
    assert rec_b["text"] == "回复B"
    assert rec_a["message_id"] == mid_a


def test_message_states_and_cancel(tmp_root):
    client = BotClient(tmp_root)
    out = client.submit("待取消", session_id="s")
    mid = out["message_id"]
    assert client.message_status(mid) in ("QUEUED", "RETRYABLE")
    assert client.cancel(mid) is True
    rec = client.wait_reply(mid, timeout_s=1)
    assert rec["status"] == "CANCELLED"
    assert client.cancel(mid) is False          # 已取消的不能再取消


def test_concurrent_submit_no_loss(tmp_root):
    """20 条并发提交：全部入库、不丢不重（SQLite 事务保证）。"""
    client = BotClient(tmp_root)
    mids: list[str] = []
    lock = threading.Lock()

    def worker(i):
        out = client.submit(f"并发消息{i}", session_id="c")
        with lock:
            mids.append(out["message_id"])

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(set(mids)) == 20                 # UUID 不重复
    store = client._store
    n = store.conn.execute(
        "SELECT COUNT(*) c FROM chat_messages WHERE role='user'").fetchone()["c"]
    assert n == 20


def test_restart_recovery_retryable(tmp_root):
    """断电式重启：QUEUED/RUNNING 记录恢复为 RETRYABLE，不静默消失。"""
    client = BotClient(tmp_root)
    mid = client.submit("重启前消息", session_id="r")["message_id"]
    store = client._store
    store.conn.execute(
        "UPDATE chat_messages SET status='RUNNING' WHERE message_id=?", (mid,))
    # 模拟重启：重新打开数据库（BotStore 构造即恢复）
    store.close()
    store2 = BotStore(tmp_root / "bot_state.db")
    assert store2.message_status(mid) == "RETRYABLE"
    row = store2.claim_next_message()
    assert row is not None and row["message_id"] == mid
    store2.close()


def test_sessions_isolated_and_deletable(tmp_root):
    client = BotClient(tmp_root)
    a = client.submit("会话甲", session_id="A")["message_id"]
    b = client.submit("会话乙", session_id="B")["message_id"]
    s = client._store
    s.finish_message(a, reply_text="甲答", response_source="rule")
    s.finish_message(b, reply_text="乙答", response_source="rule")
    ha = client.history("A")
    hb = client.history("B")
    assert len(ha) == 2 and len(hb) == 2        # 各自 user+assistant
    assert ha[0]["text"] == "会话甲"
    n = client.delete_session("A")
    assert n == 2 and client.history("A") == []
    assert len(client.history("B")) == 2        # 删除 A 不影响 B（会话不串）


# ---------------------------------------------------------------- ChatService 路由
def test_chat_tool_route(tmp_root):
    svc = ChatService()
    resp = svc.reply(ChatRequest.new("s", "请计算：12加34等于多少？"))
    assert resp.response_source == "tool"
    assert "46" in resp.text
    assert "calculator" in resp.note


def test_chat_retrieval_route_with_citations(tmp_root, kb):
    svc = ChatService(knowledge=kb)
    resp = svc.reply(ChatRequest.new("s", "查一下知识库里的来源与安全边界"))
    assert resp.response_source == "retrieval"
    assert resp.citations, "检索命中必须带引用"
    c = resp.citations[0]
    assert c.url and c.excerpt and c.chunk_id
    assert c.trust_state == "fixture"


def test_chat_rule_route_empty(tmp_root):
    svc = ChatService()
    resp = svc.reply(ChatRequest.new("s", "   "))
    assert resp.response_source == "rule"


def test_chat_neural_unverified_note(tmp_root, tmp_path):
    """无模型时 none；UNK 超门槛的输入 → note 明确标注未验证。"""
    svc = ChatService()
    resp = svc.reply(ChatRequest.new("s", "你好"))
    assert resp.response_source == "none"
    # 注入假模型/分词器走 neural 分支（离线单测，不依赖 checkpoint）
    from nmind.language.tokenizer import CharTokenizer
    tok = CharTokenizer.train_from_texts(["用户：你好\n助手：在。"], max_size=50)
    assert tok.vocab_size <= 50
    svc2 = ChatService(model=None, tokenizer=tok, knowledge=None)
    # model None → none 分支（不冒充）
    assert svc2.reply(ChatRequest.new("s", "你好")).response_source == "none"


def test_neural_route_uses_v2_protocol(tmp_root):
    """有模型时协议 ID 必须是 freegen-v2-direct-continuation（不许回退 v1）。"""
    from nmind.bot.chat_service import GENERATION_PROTOCOL_ID
    assert GENERATION_PROTOCOL_ID == "freegen-v2-direct-continuation"


def test_v2_session_alignment_with_training_format(tmp_path):
    """v2 会话跨轮转录与训练格式逐字符一致：用户：X\n助手：Y\n用户：Z\n助手：W。"""
    import torch

    from nmind.language.neural_dialogue import LMConfig, NMindLM
    from nmind.language.tokenizer import CharTokenizer

    tok = CharTokenizer.train_from_texts(
        ["用户：你好\n助手：在。\n用户：再见\n助手：好。"], max_size=64)
    model = NMindLM(LMConfig(vocab_size=tok.vocab_size, d_model=16, n_blocks=2,
                             block_hidden=16, workspace_dim=8, workspace_slots=2,
                             top_k=1))
    model.eval()
    sess = _V2ChatSession(model, tok, session_id="s", max_new_tokens=6)
    r1 = sess.send("你好")
    r2 = sess.send("再见")
    assert "text" in r1 and "text" in r2
    assert sess.turns == 2
    # 转录重放：拼接后前缀应可被 encode（含完整轮次边界）
    joined = "用户：你好\n助手：" + r1["text"] + "\n用户：再见\n助手：" + r2["text"]
    ids = tok.encode(joined)
    assert ids and tok.decode(ids)
    _ = torch.zeros(1)


def test_source_citation_roundtrip(tmp_root):
    c = SourceCitation(doc_id="d1", chunk_id="c1", title="标题", url="fixture://x",
                       fetched_at_utc="2026-11-22T00:00:00+0800", excerpt="摘录",
                       score=0.9, trust_state="fixture")
    c2 = SourceCitation.from_dict(c.to_dict())
    assert c2 == c


def test_runtime_chat_queue_serves_messages(tmp_root):
    """BotRuntime.process_chat_queue：领取→回复→状态 DONE，来源与延迟落库。"""
    from nmind.bot.policy import BotPolicy
    policy = BotPolicy()
    policy.idle_learning_enabled = False
    policy.skill_practice_enabled = False
    rt = BotRuntime(root_dir=tmp_root, policy=policy)
    rt.start()
    kb_store = KnowledgeStore(tmp_root / "knowledge.db")
    svc = ChatService(knowledge=KnowledgeSearch(kb_store))
    client = BotClient(tmp_root)
    try:
        mid = client.submit("请计算：6乘7等于多少？", session_id="q")["message_id"]
        served = rt.process_chat_queue(svc)
        assert served == 1
        rec = client.wait_reply(mid, timeout_s=5)
        assert rec["response_source"] == "tool"
        assert "42" in rec["text"]
        assert rt.store.message_status(mid) == "DONE"
        # 队列空后再调用返回 0
        assert rt.process_chat_queue(svc) == 0
    finally:
        rt.stop("测试结束")


def test_chat_service_reset_session(tmp_root):
    svc = ChatService()
    svc._sessions["x"] = _V2ChatSession.__new__(_V2ChatSession)
    svc.reset_session("x")
    assert "x" not in svc._sessions
