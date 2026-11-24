"""Web UI 服务测试（HTTP 层冒烟 + 安全边界 + 消息协议）。

服务只绑定 127.0.0.1；所有请求来自本测试进程，零真实网络。
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

import pytest

from nmind.bot.chat_service import ChatService
from nmind.bot.client import BotClient
from nmind.bot.idle_explorer import IdleWebExplorer
from nmind.bot.knowledge_ingest import ensure_fixture_pages
from nmind.bot.knowledge_search import KnowledgeSearch
from nmind.bot.knowledge_store import KnowledgeStore
from nmind.bot.policy import BotPolicy
from nmind.bot.source_registry import SourceRegistry
from nmind.bot.web_server import WebBotServer


@pytest.fixture()
def server(tmp_path):
    root = tmp_path / "web"
    client = BotClient(root)
    kb = KnowledgeSearch(KnowledgeStore(root / "knowledge.db"))
    ensure_fixture_pages(kb.store)
    policy = BotPolicy()
    policy.idle_learning_enabled = False
    policy.skill_practice_enabled = False
    registry = SourceRegistry(root / "sources.json")
    explorer = IdleWebExplorer(registry=registry, knowledge_store=kb.store,
                               policy=policy, offline=True)
    svc = ChatService(knowledge=kb)
    srv = WebBotServer(root_dir=root, client=client, chat_service=svc,
                       knowledge=kb, explorer=explorer, offline=True)
    url = srv.start()
    try:
        yield srv, url
    finally:
        srv.stop()
        client.close()


def _get(url: str, path: str) -> dict:
    with urllib.request.urlopen(url + path.lstrip("/"), timeout=5) as r:
        return json.loads(r.read())


def _post(url: str, path: str, body: dict) -> dict:
    req = urllib.request.Request(url + path.lstrip("/"),
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read())


def test_binds_loopback_only(server):
    srv, url = server
    assert url.startswith("http://127.0.0.1:")
    # 服务地址不包含非回环 IP
    host = url[len("http://"):].split(":")[0]
    assert host == "127.0.0.1"


def test_serves_index_and_asset(server):
    _srv, url = server
    html = urllib.request.urlopen(url, timeout=5).read().decode("utf-8")
    assert "N-MIND" in html and "assets/nmind-brain.png" in html
    png = urllib.request.urlopen(url + "assets/nmind-brain.png", timeout=5)
    assert png.headers["Content-Type"] == "image/png"
    assert len(png.read()) > 1000


def test_static_path_traversal_blocked(server):
    _srv, url = server
    try:
        with urllib.request.urlopen(url + "assets/../web_server.py",
                                    timeout=5) as r:
            body = r.read()
    except urllib.error.HTTPError as exc:
        body = exc.read()
    assert b"error" in body or b"403" in body or b"404" in body


def test_status_endpoint(server):
    _srv, url = server
    st = _get(url, "/api/runtime/status")
    assert st["offline"] is True
    assert "state" in st and "quota" in st
    assert st["quota"]["offline"] is True


def test_knowledge_search_and_stats(server):
    _srv, url = server
    stats = _get(url, "/api/knowledge/stats")
    assert stats["documents"] == 3
    res = _post(url, "/api/knowledge/search", {"q": "安全边界", "limit": 3})
    assert res["results"]
    c = res["results"][0]["citation"]
    assert c["chunk_id"] and c["url"] and c["excerpt"]


def test_chat_message_full_protocol(server):
    """提交 → message_id → 轮询领取对应回复（Web 层同样不允许拿最新日志冒充）。"""
    _srv, url = server
    sid = _post(url, "/api/chat/sessions", {"title": "web测试"})["session_id"]
    out = _post(url, f"/api/chat/sessions/{sid}/messages",
                {"text": "请计算：7乘8等于多少？"})
    mid = out["message_id"]
    rec = None
    for _ in range(150):
        rr = _get(url, f"/api/chat/messages/{mid}")
        if rr.get("status") != "PENDING":
            rec = rr
            break
        time.sleep(0.02)
    assert rec is not None
    assert rec["response_source"] == "tool"
    assert "56" in rec["text"]


def test_web_site_authorization_flow(server):
    """站点授权写入本地列表；离线下读取真实 URL 被明确拒绝。"""
    _srv, url = server
    add = _post(url, "/api/web/sites", {"host": "docs.example.org",
                                        "path_scope": "/kb"})
    assert add["host"] == "docs.example.org"
    st = _get(url, "/api/web/status")
    assert any(s["host"] == "docs.example.org" for s in st["registry"]["sites"])
    fetch = _post(url, "/api/web/fetch", {"url": "https://docs.example.org/kb/x"})
    assert fetch["status"] == "blocked" and fetch["reason"] == "offline_mode"
    # fixture 读取在离线下允许（零真实网络）
    fetch2 = _post(url, "/api/web/fetch", {"url": "fixture://local/nmind-intro.html"})
    assert fetch2["status"] in ("fetched", "deduplicated")


def test_control_endpoint(server):
    _srv, url = server
    r = _post(url, "/api/control", {"action": "pause"})
    assert r["action"] == "pause"
    r = _post(url, "/api/control", {"action": "resume"})
    assert r["action"] == "resume"
    bad = _post(url, "/api/control", {"action": "hack"})
    assert bad["ok"] is False


def test_exploring_toggle_and_activity(server):
    _srv, url = server
    r = _post(url, "/api/web/explore", {"enabled": True})
    assert r["exploring_enabled"] is True
    act = _get(url, "/api/activity")
    assert "events" in act


def test_empty_message_rejected(server):
    _srv, url = server
    sid = _post(url, "/api/chat/sessions", {"title": "x"})["session_id"]
    try:
        _post(url, f"/api/chat/sessions/{sid}/messages", {"text": ""})
        pytest.fail("空文本应当被拒绝")
    except urllib.error.HTTPError as exc:
        assert exc.code == 400
