"""状态视图：顶部状态条 + 健康快照页（模型/网络/队列/资源/最后错误）。"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import json


class StatusView(ttk.Frame):
    def __init__(self, master, **kw):
        super().__init__(master, **kw)
        self.text = tk.Text(self, wrap="word", font=("Microsoft YaHei UI", 10),
                            padx=8, pady=6, height=18)
        self.text.pack(fill="both", expand=True)
        self.text.configure(state="disabled")

    def show_health(self, snap: dict) -> None:
        lines = []
        model_ok = snap.get("model_loaded")
        lines.append(("模型状态", "已加载（本地 NMindLM，仅本地推理）" if model_ok
                      else "缺失（神经侧不可用：界面与缓存知识仍可用，如实标注）"))
        lines.append(("运行状态", snap.get("state", "?")))
        if snap.get("model_sha"):
            lines.append(("模型SHA", str(snap.get("model_sha"))[:16]))
        if snap.get("vocab_size"):
            lines.append(("词表大小", f"{snap.get('vocab_size')} token（中文闭集词表）"))
        lines.append(("生成协议", str(snap.get("generation_protocol_id") or "-")))
        lines.append(("知识库", f"文档 {snap.get('knowledge_docs', 0)} 篇 / "
                                f"片段 {snap.get('knowledge_chunks', 0)} 条"))
        net = "离线（--offline 硬断，0 真实请求）" if snap.get("offline") else (
            "受控联网（仅白名单站点，限额+审计）" if snap.get("network_enabled")
            else "已禁用（默认安全）")
        lines.append(("网络", net))
        lines.append(("等待队列", str(snap.get("queue_len", 0))))
        if snap.get("last_error"):
            lines.append(("最后错误", str(snap.get("last_error"))))
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        for k, v in lines:
            self.text.insert("end", f"{k}：", ("k",))
            self.text.insert("end", f"{v}\n")
        self.text.tag_configure("k", foreground="#444",
                                font=("Microsoft YaHei UI", 10, "bold"))
        self.text.configure(state="disabled")


class TopStatusBar(ttk.Frame):
    """主窗口底部状态条：离线/联网、探索、模型、队列、最后错误。"""

    def __init__(self, master, **kw):
        super().__init__(master, **kw)
        self.var = tk.StringVar(value="初始化…")
        lbl = ttk.Label(self, textvariable=self.var, anchor="w",
                        font=("Microsoft YaHei UI", 9))
        lbl.pack(fill="x")

    def update_text(self, text: str) -> None:
        self.var.set(text)


def format_health_line(snap: dict) -> str:
    mode = "离线" if snap.get("offline") else ("受控联网" if snap.get("network_enabled")
                                              else "网络禁用")
    model = "模型已加载" if snap.get("model_loaded") else "模型缺失"
    state = snap.get("state", "?")
    q = snap.get("queue_len", 0)
    docs = snap.get("knowledge_docs", 0)
    return f"{mode} · {model} · {state} · 待处理 {q} · 知识文档 {docs}"
