"""Bot V1 GUI 测试：Tkinter 可用性 + 视图逻辑（窗口 withdraw，不弹前台）。"""

from __future__ import annotations

import pytest

tk = pytest.importorskip("tkinter")


@pytest.fixture()
def tk_root():
    try:
        root = tk.Tk()
    except tk.TclError as exc:                      # 无显示环境：跳过（如实标注）
        pytest.skip(f"tkinter 无法初始化显示：{exc}")
    root.withdraw()
    yield root
    root.destroy()


def test_source_labels_cover_five_sources():
    from nmind.bot.ui.chat_view import SOURCE_LABELS
    for src in ("neural", "retrieval", "tool", "rule", "none"):
        assert src in SOURCE_LABELS
        label, color = SOURCE_LABELS[src]
        assert label and color.startswith("#")


def test_chat_view_renders_all_sources(tk_root):
    """五种来源 + 引用渲染不抛异常，且文本可见（来源不允许隐藏）。"""
    from nmind.bot.ui.chat_view import ChatView
    view = ChatView(tk_root, on_send=lambda t: None, on_cancel=lambda m: None)
    view.pack()
    view.append_user("你好", "m1")
    view.append_reply("m1", "神经原样输出", "neural",
                      note="未验证提示", latency_ms=12.0)
    view.append_reply("m1", "摘录内容", "retrieval",
                      citations=[{"doc_id": "d", "chunk_id": "c", "title": "T",
                                  "url": "fixture://u", "fetched_at_utc": "t",
                                  "excerpt": "摘录", "trust_state": "fixture"}])
    view.append_reply("m1", "计算结果：42。", "tool")
    view.append_reply("m1", "规则提示文本", "rule")
    view.append_reply("m1", "不能可靠回答", "none")
    text = view.display.get("1.0", "end")
    for token in ("你好", "神经原样输出", "摘录内容", "计算结果：42", "规则提示文本",
                  "不能可靠回答", "N-MIND神经生成", "本地知识库检索", "确定性工具",
                  "规则提示", "当前不能可靠回答", "fixture://u"):
        assert token in text


def test_chat_view_enter_vs_shift_enter(tk_root):
    """Enter 触发发送；Shift+Enter 保留换行（事件处理器分支断言）。"""
    from nmind.bot.ui.chat_view import ChatView
    sent: list[str] = []
    view = ChatView(tk_root, on_send=sent.append, on_cancel=lambda m: None)
    view.input.insert("1.0", "测试文本")

    class E:
        state = 0x0

    assert view._on_enter(E()) == "break"
    assert sent == ["测试文本"]

    class EShift:
        state = 0x1

    assert view._on_enter(EShift()) is None          # Shift+Enter 不发送


def test_status_view_and_health_line(tk_root):
    from nmind.bot.ui.status_view import StatusView, format_health_line
    snap = {"model_loaded": True, "model_sha": "abc", "vocab_size": 31,
            "generation_protocol_id": "freegen-v2-direct-continuation",
            "knowledge_docs": 3, "knowledge_chunks": 6, "offline": True,
            "network_enabled": False, "queue_len": 0, "state": "READY",
            "running": True}
    sv = StatusView(tk_root)
    sv.show_health(snap)
    text = sv.text.get("1.0", "end")
    assert "已加载" in text and "离线" in text and "31" in text
    line = format_health_line(snap)
    assert "离线" in line and "模型已加载" in line and "知识文档 3" in line


def test_bot_health_contract_fields():
    from nmind.bot.chat_service import BotHealth
    h = BotHealth(running=True, state="READY", model_loaded=True, model_sha="s",
                  vocab_size=31, generation_protocol_id="p", knowledge_docs=1,
                  knowledge_chunks=1, network_enabled=False, offline=True,
                  queue_len=0).to_dict()
    for field in ("response_source",) if False else (
            "running", "state", "model_loaded", "model_sha", "vocab_size",
            "generation_protocol_id", "knowledge_docs", "knowledge_chunks",
            "network_enabled", "offline", "queue_len"):
        assert field in h
