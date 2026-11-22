"""IdleWebExplorer 测试（任务书 §8.1 探索组 + §8.3 场景D）。

离线 fixture 模式：全部走注入 transport，零真实网络。
"""

from __future__ import annotations

import time

import pytest

from nmind.bot.idle_explorer import IdleWebExplorer
from nmind.bot.knowledge_search import KnowledgeSearch
from nmind.bot.knowledge_store import KnowledgeStore
from nmind.bot.policy import BotPolicy
from nmind.bot.runtime import BotRuntime
from nmind.bot.source_registry import SourceRegistry
from nmind.online.fixtures import SIM_WEB, fixture_transport


@pytest.fixture()
def env(tmp_path):
    root = tmp_path / "bot"
    store = KnowledgeStore(root / "knowledge.db")
    registry = SourceRegistry(root / "sources.json")
    policy = BotPolicy()
    policy.idle_learning_enabled = False
    policy.skill_practice_enabled = False
    rt = BotRuntime(root_dir=root, policy=policy)
    rt.start()
    explorer = IdleWebExplorer(registry=registry, knowledge_store=store, policy=policy,
                               offline=True, transport=fixture_transport,
                               clock=time.time)
    yield rt, explorer, registry, store
    rt.stop("测试结束")


def test_disabled_explorer_dormant(env):
    rt, explorer, _reg, _store = env
    out = explorer.run_idle_exploration(rt)
    assert out["explored"] is False and out["reason"] == "exploring_disabled"
    assert out["fetches"] == []


def test_no_allowed_sources_dormant(env):
    rt, explorer, _reg, _store = env
    explorer.set_enabled(True)
    out = explorer.run_idle_exploration(rt)
    assert out["reason"] == "no_allowed_sources"


def test_offline_fixture_exploration_ingests(env):
    """fixture 源探索：抓取→抽取→去重→入库，全部标注 fixture，台账可见。"""
    rt, explorer, registry, store = env
    registry.add("sim.local", created_by="test")
    explorer.set_enabled(True)
    for url in SIM_WEB:
        store.upsert_job(url, "sim.local", depth=0)
    out = explorer.run_idle_exploration(rt)
    assert out["explored"] is not None
    assert len(out["fetches"]) >= 1
    st = KnowledgeSearch(store).stats()
    assert st["documents"] >= 1
    # 所有文档 trust_state=fixture
    for row in store.conn.execute("SELECT trust_state FROM documents"):
        assert row["trust_state"] == "fixture"


def test_quota_exhausted_dormant(env):
    rt, explorer, registry, _store = env
    registry.add("sim.local", created_by="test")
    explorer.set_enabled(True)
    lim = registry.limits
    # 烧掉每日配额
    for _ in range(int(lim["requests_per_day"])):
        explorer._charge(10)
    out = explorer.run_idle_exploration(rt)
    assert out["reason"] == "daily_request_quota_exhausted"
    assert out["fetches"] == []


def test_user_message_preempts_exploration(env):
    """用户消息到达 → 探索立即让路（请求边界抢占）。"""
    rt, explorer, registry, _store = env
    registry.add("sim.local", created_by="test")
    explorer.set_enabled(True)
    for url in SIM_WEB:
        explorer.store.upsert_job(url, "sim.local", depth=0)
    rt.submit_user_task("用户来了")            # inbox 有消息
    out = explorer.run_idle_exploration(rt)
    assert out["preempted"] is True and out["explored"] is False
    assert out["fetches"] == []


def test_external_links_not_queued(env):
    """新网页里的外域链接不得进入任务队列（不能自动扩大 allowlist）。"""
    rt, explorer, registry, store = env
    registry.add("sim.local", created_by="test")
    n = explorer._queue_same_site_links(
        "https://sim.local/page", "sim.local",
        ["https://sim.local/sub/page.html", "https://evil.example.net/x",
         "http://sim.local/http.html"], depth=0)
    jobs = store.conn.execute("SELECT url FROM explore_jobs").fetchall()
    urls = [j["url"] for j in jobs]
    assert "https://sim.local/sub/page.html" in urls
    assert all("evil.example.net" not in u for u in urls)
    assert "http://sim.local/http.html" not in urls     # http 也不入队
    _ = n


def test_network_disabled_no_exploration(tmp_path):
    """真实网络未授权（network_enabled=False + 非离线夹具）→ 明确休眠。"""
    root = tmp_path / "bot2"
    store = KnowledgeStore(root / "knowledge.db")
    registry = SourceRegistry(root / "sources.json")
    policy = BotPolicy()
    policy.idle_learning_enabled = False
    policy.skill_practice_enabled = False
    policy.network_enabled = False
    rt = BotRuntime(root_dir=root, policy=policy)
    rt.start()
    explorer = IdleWebExplorer(registry=registry, knowledge_store=store, policy=policy,
                               offline=False, transport=None)
    explorer.set_enabled(True)
    registry.add("example.com", created_by="test")
    out = explorer.run_idle_exploration(rt)
    assert out["reason"] == "network_disabled" and out["fetches"] == []
    rt.stop("测试结束")


def test_fetch_for_user_blocked_offline(env):
    """离线下 GUI 读取真实 URL → 明确拒绝（不发送任何真实请求）。"""
    rt, explorer, registry, _store = env
    registry.add("example.com", created_by="test")
    rec = explorer.fetch_for_user("https://example.com/")
    assert rec["status"] == "blocked" and rec["reason"] == "offline_mode"


def test_pause_flag_stops_exploration(env):
    rt, explorer, registry, store = env
    registry.add("sim.local", created_by="test")
    explorer.set_enabled(True)
    for url in SIM_WEB:
        store.upsert_job(url, "sim.local", depth=0)
    rt.pause("测试暂停")
    out = explorer.run_idle_exploration(rt)
    assert out["paused"] is True or out["preempted"] is True


def test_stable_untouched_during_exploration(env):
    """探索全程 stable 权重 sha 逐位不变（知识增长不等于神经学习）。"""
    rt, explorer, registry, _store = env
    sha_before = rt.stable_sha
    registry.add("sim.local", created_by="test")
    explorer.set_enabled(True)
    for url in SIM_WEB:
        explorer.store.upsert_job(url, "sim.local", depth=0)
    explorer.run_idle_exploration(rt)
    assert rt.stable_sha == sha_before
    assert rt.steps_used == 0
