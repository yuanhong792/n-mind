"""聊天视图：消息渲染 + 回复来源标注 + 引用区 + 输入区。

线程规则（任务书 §3.1）：本类所有公开方法只能在 Tkinter 主线程调用；
工作线程通过 queue.Queue 投递 (kind, payload) 消息，由 desktop 轮询分发。
"""

from __future__ import annotations

import queue
import tkinter as tk
from tkinter import scrolledtext, ttk

# 五种回复来源的中文标签与配色——UI 上不允许隐藏源头
SOURCE_LABELS = {
    "neural": ("N-MIND神经生成（未验证）", "#1a5fa8"),
    "retrieval": ("本地知识库检索", "#1e7d3e"),
    "tool": ("确定性工具", "#b06d1a"),
    "rule": ("规则提示", "#5f5f5f"),
    "none": ("当前不能可靠回答", "#a02b2b"),
}

NEURAL_TAG = "src_neural"
USER_TAG = "msg_user"
ASSISTANT_TAG = "msg_assistant"
URL_TAG = "cite_url"


class ChatView(ttk.Frame):
    """聊天主区：只读渲染文本 + 多行输入框 + 发送/取消按钮。"""

    def __init__(self, master, *, on_send, on_cancel, **kw):
        super().__init__(master, **kw)
        self._on_send = on_send
        self._on_cancel = on_cancel
        self._pending: dict[str, str] = {}          # message_id -> 占位序号文本
        self._cancel_requested: set[str] = set()

        self.display = scrolledtext.ScrolledText(self, wrap="word", state="disabled",
                                                 font=("Microsoft YaHei UI", 10),
                                                 padx=8, pady=6)
        self.display.pack(fill="both", expand=True)
        self.display.tag_configure(USER_TAG, foreground="#123c6b",
                                   font=("Microsoft YaHei UI", 10, "bold"))
        self.display.tag_configure(ASSISTANT_TAG, foreground="#202020")
        self.display.tag_configure(NEURAL_TAG, font=("Microsoft YaHei UI", 8))
        self.display.tag_configure(URL_TAG, foreground="#1a5fa8", underline=True)
        self.display.tag_bind(URL_TAG, "<Button-1>", self._open_link)
        for src, (_label, color) in SOURCE_LABELS.items():
            self.display.tag_configure(f"badge_{src}", foreground=color,
                                       font=("Microsoft YaHei UI", 8, "bold"))

        input_bar = ttk.Frame(self)
        input_bar.pack(fill="x", pady=(4, 0))
        self.input = tk.Text(input_bar, height=3, wrap="word",
                             font=("Microsoft YaHei UI", 10))
        self.input.pack(side="left", fill="both", expand=True)
        self.input.bind("<Return>", self._on_enter)
        self.input.bind("<Shift-Return>", lambda _e: None)   # Shift+Enter 换行（默认行为）
        btns = ttk.Frame(input_bar)
        btns.pack(side="right", fill="y")
        self.send_btn = ttk.Button(btns, text="发送", command=self._send_clicked)
        self.send_btn.pack(fill="x", pady=(0, 4))
        self.cancel_btn = ttk.Button(btns, text="取消", command=self._cancel_clicked,
                                     state="disabled")
        self.cancel_btn.pack(fill="x")
        self._inflight: str | None = None

    # ------------------------------------------------------------ 输入
    def _on_enter(self, event):
        # Shift+Enter 换行；单独 Enter 发送
        if int(event.state) & 0x1:
            return None
        self._send_clicked()
        return "break"

    def _send_clicked(self):
        text = self.input.get("1.0", "end").strip()
        if not text:
            return
        self.input.delete("1.0", "end")
        self._on_send(text)

    def _cancel_clicked(self):
        if self._inflight and self._inflight not in self._cancel_requested:
            self._cancel_requested.add(self._inflight)
            self._on_cancel(self._inflight)
            self.append_system("已请求取消该消息（若正在生成将在本条完成后停止）。")

    # ------------------------------------------------------------ 渲染
    def append_user(self, text: str, message_id: str) -> None:
        self._enable()
        self.display.insert("end", f"我：{text}\n", USER_TAG)
        self._pending[message_id] = message_id
        self.display.insert("end", "  └ 处理中…\n", ("pending", message_id))
        self._disable()
        self._inflight = message_id
        self.send_btn.state(["!disabled"])
        self.cancel_btn.state(["!disabled"])

    def append_reply(self, message_id: str, text: str, response_source: str,
                     note: str = "", citations: list[dict] | None = None,
                     latency_ms: float = 0.0) -> None:
        label, _color = SOURCE_LABELS.get(response_source, (response_source, "#333"))
        self._enable()
        self.display.insert("end", f"N-MIND：{text}\n", ASSISTANT_TAG)
        self.display.insert("end", f"  [来源：{label}", f"badge_{response_source}")
        if latency_ms:
            self.display.insert("end", f" · {latency_ms:.0f}ms", f"badge_{response_source}")
        self.display.insert("end", "]\n", f"badge_{response_source}")
        for c in citations or []:
            head = f"  ↳ 《{c.get('title') or '未命名'}》 "
            self.display.insert("end", head, NEURAL_TAG)
            url = c.get("url") or ""
            if url:
                self.display.insert("end", url, (URL_TAG, f"u_{message_id}"))
            stamp = c.get("fetched_at_utc") or ""
            trust = "fixture夹具" if c.get("trust_state") == "fixture" else "已抓取"
            if c.get("stale"):
                trust += " · stale(已失效)"
            self.display.insert("end", f"  [{trust} {stamp}]\n", NEURAL_TAG)
        if note:
            self.display.insert("end", f"  （{note}）\n", NEURAL_TAG)
        self._disable()
        self._pending.pop(message_id, None)
        if not self._pending:
            self._inflight = None
            self.cancel_btn.state(["disabled"])

    def append_system(self, text: str) -> None:
        self._enable()
        self.display.insert("end", f"—— {text}\n", NEURAL_TAG)
        self._disable()

    def append_error(self, message_id: str, error: str) -> None:
        self._enable()
        self.display.insert("end",
                            f"N-MIND：处理该消息失败：{error}\n", ASSISTANT_TAG)
        self.display.insert("end", "  [来源：错误反馈（请求未完成，可重试）]\n",
                            "badge_none")
        self._disable()
        self._pending.pop(message_id, None)
        if not self._pending:
            self._inflight = None
            self.cancel_btn.state(["disabled"])

    def _open_link(self, event):
        url = self.display.get(event.widget.index(f"@{event.x},{event.y}") +
                               " linestart", f"{event.widget.index(f'@{event.x},{event.y}')} lineend")
        import re
        m = re.search(r"https?://\S+", url)
        if not m:
            return
        import webbrowser
        try:
            webbrowser.open(m.group(0))
        except Exception:                              # noqa: BLE001 - 打不开不崩 UI
            self.append_system(f"浏览器打开失败：{m.group(0)}")

    def _enable(self):
        self.display.configure(state="normal")

    def _disable(self):
        self.display.configure(state="disabled")
        self.display.see("end")

    def set_history(self, rows: list[dict]) -> None:
        self.display.configure(state="normal")
        self.display.delete("1.0", "end")
        self.display.configure(state="disabled")
        for r in rows:
            if r["role"] == "user":
                self.append_user_no_pending(r["text"])
            else:
                import json as _json
                try:
                    cites = _json.loads(r.get("citations_json") or "[]")
                except ValueError:
                    cites = []
                note = ""
                if r.get("error"):
                    try:
                        note = _json.loads(r["error"]).get("note", "")
                    except (ValueError, AttributeError):
                        note = ""
                self.append_reply(r["message_id"], r["text"],
                                  r.get("response_source") or "none", note=note,
                                  citations=cites, latency_ms=r.get("latency_ms") or 0.0)

    def append_user_no_pending(self, text: str) -> None:
        self._enable()
        self.display.insert("end", f"我：{text}\n", USER_TAG)
        self._disable()

    def history_drain(self) -> dict[str, str]:
        return dict(self._pending)
