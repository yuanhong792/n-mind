"""N-MIND Bot V1 Windows 桌面主窗口（Tkinter 标准库）。

架构与线程规则（任务书 §2/§3.1）：
  * 一个 Bot 进程、一套状态与数据：本进程内的唯一 BotRuntime 实例由服务线程
    驱动 run_forever(chat_service=..., explorer=...)；若已有外部 Bot 实例在跑
    （lock.json 存活），本窗口只作为客户端连接，不启动第二份运行时、不碰锁；
  * Tkinter 控件只能在主线程更新；服务/抓取/搜索线程一律通过 ui_queue 投递；
  * 主线程不做模型推理、不做 HTTP、不做 FTS 大查询——只有轻量 SQLite 读；
  * 关闭窗口：本进程服务 → stop() 释放锁再退出；外部服务 → 只关窗口不误杀。
"""

from __future__ import annotations

import queue
import threading
import time
import traceback
from pathlib import Path
from typing import Any

from ..chat_service import BotHealth, ChatService
from ..client import BotClient
from ..runtime import BotRuntime, read_status
from .chat_view import ChatView
from .status_view import StatusView, TopStatusBar, format_health_line


class NMindBotApp:
    def __init__(self, root, *, runtime: BotRuntime | None, client: BotClient,
                 chat_service: ChatService, knowledge=None, explorer=None,
                 offline: bool = True, owned_runtime: bool = True,
                 session_id: str = "default") -> None:
        import tkinter as tk
        from tkinter import messagebox, simpledialog, ttk
        self.tk, self.ttk = tk, ttk
        self.messagebox, self.simpledialog = messagebox, simpledialog
        self.root = root
        self.runtime = runtime
        self.client = client
        self.chat_service = chat_service
        self.knowledge = knowledge          # KnowledgeSearch 实例（P3；可为 None）
        self.explorer = explorer            # IdleWebExplorer 实例（P4；可为 None）
        self.offline = bool(offline)
        self.owned_runtime = bool(owned_runtime)
        self.session_id = session_id
        self.ui_queue: queue.Queue = queue.Queue()
        self._inflight: dict[str, float] = {}
        self._closing = False

        root.title("N-MIND Bot V1 · 中文聊天（本地模型 · 离线优先）")
        root.geometry("1120x740")
        root.minsize(900, 600)

        self._build_menus()
        pane = ttk.PanedWindow(root, orient="horizontal")
        pane.pack(fill="both", expand=True)

        left = ttk.Frame(pane)
        pane.add(left, weight=3)
        self._build_chat_pane(left)

        right = ttk.Frame(pane)
        pane.add(right, weight=2)
        self._build_notebook(right)

        self.status_bar = TopStatusBar(root)
        self.status_bar.pack(fill="x", side="bottom")

        # 恢复历史会话
        try:
            self.chat_view.set_history(self.client.history(self.session_id))
        except Exception as exc:                        # noqa: BLE001 - 历史读不到不崩 UI
            print(f"[bot-ui] 历史会话加载失败：{type(exc).__name__}: {exc}")

        if self.runtime is not None:
            self._service_thread = threading.Thread(target=self._service_main,
                                                    daemon=True, name="nmind-bot-service")
            self._service_thread.start()
        else:
            self._service_thread = None
        root.after(120, self._poll_ui_queue)
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._refresh_status()

    # ------------------------------------------------------------ UI 构建
    def _build_menus(self) -> None:
        menubar = self.tk.Menu(self.root)
        m_session = self.tk.Menu(menubar, tearoff=0)
        m_session.add_command(label="新建会话", command=self._new_session)
        m_session.add_command(label="删除当前会话", command=self._delete_session)
        m_session.add_separator()
        m_session.add_command(label="退出", command=self._on_close)
        menubar.add_cascade(label="会话", menu=m_session)
        self.root.config(menu=menubar)

    def _build_chat_pane(self, parent) -> None:
        top = self.ttk.Frame(parent)
        top.pack(fill="x", pady=(2, 0))
        self.ttk.Label(top, text="会话：").pack(side="left")
        self.session_var = self.tk.StringVar(value=self.session_id)
        self.session_box = self.ttk.Combobox(top, textvariable=self.session_var,
                                             state="readonly", width=28)
        self.session_box.pack(side="left", padx=4)
        self.session_box.bind("<<ComboboxSelected>>", lambda _e: self._switch_session())
        self.ttk.Button(top, text="新建", width=6, command=self._new_session).pack(side="left")
        self.ttk.Button(top, text="删除", width=6, command=self._delete_session).pack(side="left")
        self.chat_view = ChatView(parent, on_send=self._send, on_cancel=self._cancel)
        self.chat_view.pack(fill="both", expand=True)

    def _build_notebook(self, parent) -> None:
        nb = self.ttk.Notebook(parent)
        nb.pack(fill="both", expand=True)
        # ---- 状态页 ----
        page_status = self.ttk.Frame(nb)
        nb.add(page_status, text="状态")
        self.status_view = StatusView(page_status)
        self.status_view.pack(fill="both", expand=True)
        btns = self.ttk.Frame(page_status)
        btns.pack(fill="x", pady=4)
        self.ttk.Button(btns, text="暂停探索", width=10,
                        command=lambda: self._control("pause")).pack(side="left", padx=2)
        self.ttk.Button(btns, text="恢复", width=8,
                        command=lambda: self._control("resume")).pack(side="left", padx=2)
        self.ttk.Button(btns, text="停止服务", width=10,
                        command=lambda: self._control("stop")).pack(side="left", padx=2)
        # ---- 知识页 ----
        page_kb = self.ttk.Frame(nb)
        nb.add(page_kb, text="知识库")
        row = self.ttk.Frame(page_kb)
        row.pack(fill="x", pady=4)
        self.ttk.Label(row, text="搜索：").pack(side="left")
        self.kb_query = self.ttk.StringVar()
        self.ttk.Entry(row, textvariable=self.kb_query).pack(side="left", fill="x",
                                                             expand=True, padx=4)
        self.ttk.Button(row, text="搜索知识", command=self._kb_search).pack(side="left")
        self.kb_text = self.tk.Text(page_kb, wrap="word", font=("Microsoft YaHei UI", 9),
                                    padx=8, pady=6, height=16)
        self.kb_text.pack(fill="both", expand=True)
        kb_btns = self.ttk.Frame(page_kb)
        kb_btns.pack(fill="x", pady=4)
        self.ttk.Button(kb_btns, text="导入测试资料（fixture，标注来源）",
                        command=self._kb_import_fixture).pack(side="left", padx=2)
        self.ttk.Button(kb_btns, text="刷新统计", command=self._kb_stats).pack(side="left",
                                                                              padx=2)
        self.kb_stats_var = self.tk.StringVar(value="文档 0 篇 / 片段 0 条")
        self.ttk.Label(kb_btns, textvariable=self.kb_stats_var).pack(side="left", padx=8)
        self._kb_stats()
        # ---- 设置页 ----
        page_cfg = self.ttk.Frame(nb)
        nb.add(page_cfg, text="设置")
        self.ttk.Label(page_cfg, text="允许站点（真实联网白名单；默认为空）",
                       font=("Microsoft YaHei UI", 10, "bold")).pack(anchor="w", pady=(6, 2))
        self.site_list = self.tk.Listbox(page_cfg, height=6, font=("Microsoft YaHei UI", 9))
        self.site_list.pack(fill="x", padx=8)
        site_row = self.ttk.Frame(page_cfg)
        site_row.pack(fill="x", pady=4)
        self.site_var = self.ttk.StringVar()
        self.ttk.Entry(site_row, textvariable=self.site_var).pack(side="left", fill="x",
                                                                  expand=True, padx=8)
        self.ttk.Button(site_row, text="添加允许站点", command=self._site_add).pack(side="left")
        self.ttk.Button(site_row, text="删除选中", command=self._site_del).pack(side="left",
                                                                                padx=4)
        self.ttk.Separator(page_cfg).pack(fill="x", pady=6)
        self.explore_var = self.tk.BooleanVar(value=False)
        self.ttk.Checkbutton(page_cfg, text="允许空闲自主探索（仅已授权站点，可随时停止）",
                             variable=self.explore_var,
                             command=self._toggle_explore).pack(anchor="w", padx=8)
        self.net_note = self.ttk.Label(
            page_cfg, text="网络：本窗口运行在离线模式（--offline）。真实联网需在配置中显式"
                           "启用 network_enabled 且逐站点授权；网页内容永远无权修改本页设置。",
            wraplength=420, justify="left", foreground="#555")
        self.net_note.pack(anchor="w", padx=8, pady=8)
        url_row = self.ttk.Frame(page_cfg)
        url_row.pack(fill="x", pady=4)
        self.ttk.Label(url_row, text="读取网页URL：").pack(side="left", padx=(8, 0))
        self.fetch_var = self.ttk.StringVar()
        self.ttk.Entry(url_row, textvariable=self.fetch_var).pack(side="left", fill="x",
                                                                  expand=True, padx=4)
        self.ttk.Button(url_row, text="读取", command=self._user_fetch).pack(side="left")
        self._refresh_sites()

    # ------------------------------------------------------------ 服务线程
    def _service_main(self) -> None:
        try:
            self.runtime.run_forever(chat_service=self.chat_service,
                                     explorer=self.explorer, poll_s=0.5)
        except Exception as exc:                        # noqa: BLE001 - 服务崩溃必须上屏
            self.ui_queue.put(("service_error",
                               f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-1500:]}"))

    def _control(self, action: str) -> None:
        from ..runtime import request_control
        try:
            if self.runtime is not None and self.owned_runtime:
                if action == "pause":
                    self.runtime.pause("GUI 暂停")
                elif action == "resume":
                    self.runtime.resume()
                elif action == "stop":
                    self.ui_queue.put(("service_stopping", None))
                    return
            request_control(self.client.root, action)
            self.chat_view.append_system(f"已发送控制指令：{action}")
        except Exception as exc:                        # noqa: BLE001
            self.chat_view.append_system(f"控制指令失败：{exc}")
        self._refresh_status()

    # ------------------------------------------------------------ 发送/接收
    def _send(self, text: str) -> None:
        try:
            out = self.client.submit(text, session_id=self.session_id)
        except Exception as exc:                        # noqa: BLE001
            self.chat_view.append_system(f"发送失败：{exc}")
            return
        self.chat_view.append_user(text, out["message_id"])
        self._inflight[out["message_id"]] = time.monotonic()

    def _cancel(self, message_id: str) -> None:
        try:
            self.client.cancel(message_id)
        except Exception as exc:                      # noqa: BLE001
            print(f"[bot-ui] 取消消息失败：{type(exc).__name__}: {exc}")

    def _poll_ui_queue(self) -> None:
        try:
            while True:
                kind, payload = self.ui_queue.get_nowait()
                if kind == "reply":
                    self.chat_view.append_reply(payload["message_id"], payload["text"],
                                                payload.get("response_source") or "none",
                                                note=payload.get("note", ""),
                                                citations=payload.get("citations"),
                                                latency_ms=payload.get("latency_ms") or 0.0)
                    self._inflight.pop(payload["message_id"], None)
                    self._refresh_status()
                elif kind == "failed":
                    self.chat_view.append_error(payload["message_id"],
                                                payload.get("error", ""))
                    self._inflight.pop(payload["message_id"], None)
                elif kind == "service_error":
                    self.chat_view.append_system(f"服务线程异常：{payload}")
                elif kind == "service_stopping":
                    self._stop_owned_runtime()
                    self.chat_view.append_system("服务已停止（stable 权重未变）。")
                    self._refresh_status()
                elif kind == "kb_result":
                    self._render_kb_results(payload)
                elif kind == "fetch_result":
                    self._render_fetch_result(payload)
        except queue.Empty:
            pass
        # 未完成消息轮询（message_id 对应回复，绝不拿最新日志冒充）
        stale: list[str] = []
        for mid, t0 in list(self._inflight.items()):
            rec = self.client.message(mid)
            if rec is not None:
                if rec.get("status") == "FAILED":
                    self.ui_queue.put(("failed", {"message_id": mid,
                                                  "error": rec.get("note") or "处理失败"}))
                else:
                    self.ui_queue.put(("reply", rec))
            elif time.monotonic() - t0 > 180:
                stale.append(mid)
        for mid in stale:
            self.ui_queue.put(("failed", {"message_id": mid, "error": "等待超时（180s）"}))
            self._inflight.pop(mid, None)
        if not self._closing:
            self.root.after(150, self._poll_ui_queue)
            if int(time.monotonic()) % 3 == 0:
                self._refresh_status()

    # ------------------------------------------------------------ 会话
    def _session_ids(self) -> list[str]:
        try:
            return [s["session_id"] for s in self.client.sessions()]
        except Exception:                              # noqa: BLE001
            return [self.session_id]

    def _refresh_sessions(self) -> None:
        ids = self._session_ids()
        if self.session_id not in ids:
            ids.insert(0, self.session_id)
        self.session_box["values"] = ids
        self.session_var.set(self.session_id)

    def _switch_session(self) -> None:
        sid = self.session_var.get()
        if sid == self.session_id:
            return
        self.session_id = sid
        self.chat_service.reset_session(sid)
        self.chat_view.set_history(self.client.history(self.session_id))
        self._refresh_sessions()

    def _new_session(self) -> None:
        sid = self.client.new_session_id()
        self.client.create_session(sid, "新会话")
        self.session_id = sid
        self.chat_service.reset_session(sid)
        self.chat_view.set_history([])
        self._refresh_sessions()
        self.chat_view.append_system(f"已新建会话 {sid}")

    def _delete_session(self) -> None:
        sid = self.session_id
        if not self.messagebox.askyesno("删除会话", f"确定删除会话 {sid} 及其全部聊天记录？"):
            return
        self.client.delete_session(sid)
        self.session_id = self.client.new_session_id()
        self.client.create_session(self.session_id, "新会话")
        self.chat_service.reset_session(self.session_id)
        self.chat_view.set_history([])
        self._refresh_sessions()

    # ------------------------------------------------------------ 知识页
    def _kb_search(self) -> None:
        q = self.kb_query.get().strip()
        if not q or self.knowledge is None:
            self.chat_view.append_system("请输入关键词（知识库模块未加载时不可用）")
            return

        def worker():
            try:
                res = self.knowledge.search(q, limit=5)
                self.ui_queue.put(("kb_result", {"query": q, "results": res}))
            except Exception as exc:                    # noqa: BLE001
                self.ui_queue.put(("kb_result", {"query": q, "results": [],
                                                 "error": str(exc)}))
        threading.Thread(target=worker, daemon=True).start()

    def _render_kb_results(self, payload: dict) -> None:
        self.kb_text.configure(state="normal")
        self.kb_text.delete("1.0", "end")
        if payload.get("error"):
            self.kb_text.insert("end", f"搜索失败：{payload['error']}\n")
        res = payload.get("results") or []
        if not res:
            self.kb_text.insert("end", f"未找到与「{payload['query']}」相关的已保存资料。\n")
        for i, r in enumerate(res, 1):
            c = r["citation"]
            self.kb_text.insert("end", f"【{i}】《{c['title']}》 相关度 {r.get('score', 0):.2f}\n",
                                ("t",))
            self.kb_text.insert("end", f"    {c['url']}  [{c.get('trust_state')}] "
                                       f"{c.get('fetched_at_utc')}\n", ("u",))
            self.kb_text.insert("end", f"    {c['excerpt']}\n\n")
        self.kb_text.tag_configure("t", font=("Microsoft YaHei UI", 9, "bold"))
        self.kb_text.tag_configure("u", foreground="#666", font=("Microsoft YaHei UI", 8))
        self.kb_text.configure(state="disabled")
        self._kb_stats()

    def _kb_stats(self) -> None:
        if self.knowledge is None:
            self.kb_stats_var.set("知识库模块未加载")
            return
        try:
            st = self.knowledge.stats()
            self.kb_stats_var.set(f"文档 {st['documents']} 篇 / 片段 {st['chunks']} 条"
                                  f"（去重 {st.get('deduplicated', 0)}）")
        except Exception as exc:                        # noqa: BLE001
            self.kb_stats_var.set(f"统计失败：{exc}")

    def _kb_import_fixture(self) -> None:
        if self.knowledge is None:
            self.chat_view.append_system("知识库模块未加载")
            return
        try:
            from ..knowledge_ingest import import_fixture_pages
            stats = import_fixture_pages(self.knowledge)
            self.chat_view.append_system(
                f"已导入测试资料（fixture）：新增 {stats['new_documents']} 篇，"
                f"去重 {stats['deduplicated']} 篇，索引 {stats['indexed_chunks']} 条片段。")
            self._kb_stats()
        except Exception as exc:                        # noqa: BLE001
            self.chat_view.append_system(f"导入失败：{exc}")

    # ------------------------------------------------------------ 设置页
    def _refresh_sites(self) -> None:
        self.site_list.delete(0, "end")
        if self.explorer is None:
            return
        for s in self.explorer.registry.list():
            scope = f" · 路径 {s['path_scope']}" if s.get("path_scope") else ""
            self.site_list.insert("end", f"{s['host']}{scope} · {s['state']}")

    def _site_add(self) -> None:
        host = self.site_var.get().strip()
        if not host or self.explorer is None:
            return
        ok = self.messagebox.askyesno(
            "授权确认", f"确认允许 N-MIND 只读访问站点 {host}？\n"
                       "（真实联网还需配置文件启用 network_enabled；网页内容永远无权修改此列表）")
        if not ok:
            return
        self.explorer.registry.add(host, created_by="gui-user")
        self.site_var.set("")
        self._refresh_sites()
        self.chat_view.append_system(f"已添加允许站点 {host}（默认仍不联网，需配置启用）")

    def _site_del(self) -> None:
        sel = self.site_list.curselection()
        if not sel or self.explorer is None:
            return
        text = self.site_list.get(sel[0])
        host = text.split(" · ")[0]
        self.explorer.registry.remove(host)
        self._refresh_sites()

    def _toggle_explore(self) -> None:
        if self.explorer is None:
            self.explore_var.set(False)
            return
        self.explorer.set_enabled(bool(self.explore_var.get()))
        state = "开启" if self.explore_var.get() else "关闭"
        self.chat_view.append_system(f"空闲自主探索已{state}（仍然只访问已授权站点）")

    def _user_fetch(self) -> None:
        url = self.fetch_var.get().strip()
        if not url or self.explorer is None:
            return
        self.chat_view.append_system(f"请求读取 {url}（检查白名单后才发真实请求）…")

        def worker():
            try:
                rec = self.explorer.fetch_for_user(url)
                self.ui_queue.put(("fetch_result", rec))
            except Exception as exc:                    # noqa: BLE001
                self.ui_queue.put(("fetch_result", {"url": url, "status": "error",
                                                    "error": str(exc)}))
        threading.Thread(target=worker, daemon=True).start()

    def _render_fetch_result(self, rec: dict) -> None:
        status = rec.get("status")
        if status in ("fetched", "duplicate"):
            self.chat_view.append_system(
                f"读取完成：{rec.get('final_url') or rec.get('url')} "
                f"（HTTP {rec.get('http_status')}，{rec.get('bytes_read', 0)}B）\n"
                f"标题：{rec.get('title') or '-'}\n摘录：{(rec.get('excerpt') or '')[:200]}")
        else:
            self.chat_view.append_system(
                f"读取被拒绝/失败：{rec.get('url')} 状态={status} "
                f"原因={rec.get('reason') or rec.get('error') or '-'}")

    # ------------------------------------------------------------ 状态与退出
    def _refresh_status(self) -> None:
        try:
            snap = self.health_snapshot()
            self.status_bar.update_text(format_health_line(snap))
            self.status_view.show_health(snap)
            self._refresh_sessions()
        except Exception:                              # noqa: BLE001 - 状态刷新不崩 UI
            pass

    def health_snapshot(self) -> dict[str, Any]:
        kb_docs = kb_chunks = 0
        if self.knowledge is not None:
            try:
                st = self.knowledge.stats()
                kb_docs, kb_chunks = st["documents"], st["chunks"]
            except Exception as exc:                  # noqa: BLE001
                print(f"[bot-ui] 知识统计失败：{type(exc).__name__}")
        state = "?"
        queue_len = self.client.queue_len()
        if self.runtime is not None:
            state = self.runtime.state.name
            queue_len += len(self.runtime.user_queue)
        else:
            st = read_status(self.client.root)
            state = str(st.get("state", "?"))
        return BotHealth(
            running=True, state=state,
            model_loaded=self.chat_service.model_loaded,
            model_sha=self.chat_service.model_sha,
            vocab_size=self.chat_service.tok.vocab_size if self.chat_service.tok else None,
            generation_protocol_id=self.chat_service.protocol_id if hasattr(
                self.chat_service, "protocol_id") else "freegen-v2-direct-continuation",
            knowledge_docs=kb_docs, knowledge_chunks=kb_chunks,
            network_enabled=bool(self.runtime.policy.network_enabled) if self.runtime else False,
            offline=self.offline,
            queue_len=queue_len).to_dict()

    def _stop_owned_runtime(self) -> None:
        if self.runtime is not None and self.runtime.state.name != "STOPPED":
            try:
                self.runtime.stop("GUI 关闭")
            except Exception:                          # noqa: BLE001
                pass

    def _on_close(self) -> None:
        self._closing = True
        try:
            if self.explorer is not None:
                self.explorer.set_enabled(False)
            if self.owned_runtime and self.runtime is not None:
                self._stop_owned_runtime()
        finally:
            try:
                self.client.close()
            except Exception as exc:                  # noqa: BLE001
                print(f"[bot-ui] 客户端关闭失败：{type(exc).__name__}")
            self.root.destroy()


def launch_desktop(args) -> int:
    """`nmind bot ui` 入口：组装运行时/服务/知识/探索器并打开窗口。"""
    import json
    try:
        import tkinter as tk
    except ImportError as exc:
        print(json.dumps({"verdict": "TKINTER_UNAVAILABLE", "error": str(exc),
                          "hint": "当前 Python 缺少 tkinter，需要带 tk 支持的解释器"},
                         ensure_ascii=False))
        return 2

    from ..cli import _policy, _root, _stable_pair
    from ..knowledge_ingest import ensure_fixture_pages
    from ..knowledge_search import KnowledgeSearch
    from ..knowledge_store import KnowledgeStore
    from ..source_registry import SourceRegistry
    from ..store import BotStore

    root_dir = _root(args)
    policy = _policy(args)
    offline = bool(getattr(args, "offline", False)) or not policy.network_enabled
    if offline:
        policy.network_enabled = False
    if getattr(args, "no_idle", False):
        policy.idle_learning_enabled = False

    client = BotClient(root_dir)
    store = BotStore(root_dir / "bot_state.db")

    # 与 CLI run 同一套 stable 权重解析（副本优先）
    stable, tok_path = _stable_pair(args, root_dir)
    runtime = None
    chat_service = ChatService()
    if stable and tok_path and Path(stable).exists() and Path(tok_path).exists():
        from ..language.neural_dialogue import load_lm
        from ..language.tokenizer import CharTokenizer
        from ..online.promotion import file_sha256
        tokenizer = CharTokenizer.load(tok_path)
        model, meta = load_lm(stable, tokenizer)
        chat_service = ChatService(model=model, tokenizer=tokenizer,
                                   model_sha=file_sha256(stable))
    # 外部 Bot 已在跑 → 只连接，不建第二个运行时（任务书 §3.4）
    external_running = read_status(root_dir).get("running", False) if \
        (root_dir / "lock.json").exists() else False
    if not external_running:
        runtime = BotRuntime(root_dir=root_dir, policy=policy,
                             stable_checkpoint=stable, tokenizer_path=tok_path)
        if chat_service.model_loaded:
            chat_service.knowledge = KnowledgeSearch(
                KnowledgeStore(root_dir / "knowledge.db"))
        else:
            chat_service.knowledge = KnowledgeSearch(
                KnowledgeStore(root_dir / "knowledge.db"))

    knowledge = KnowledgeSearch(KnowledgeStore(root_dir / "knowledge.db"))
    registry = SourceRegistry(root_dir / "sources.json")
    from ..idle_explorer import IdleWebExplorer
    explorer = IdleWebExplorer(registry=registry, knowledge_store=knowledge.store,
                               policy=policy, offline=offline)
    if chat_service.knowledge is None:
        chat_service.knowledge = knowledge
    ensure_fixture_pages(knowledge.store)          # 幂等：首次给三篇标注 fixture 的中文样例

    root = tk.Tk()
    app = NMindBotApp(root, runtime=runtime, client=client, chat_service=chat_service,
                      knowledge=knowledge, explorer=explorer, offline=offline,
                      owned_runtime=not external_running)
    app.chat_view.append_system(
        "N-MIND Bot V1 已启动。来源标注说明：神经生成（未验证）/ 本地知识库检索 / "
        "确定性工具 / 规则提示 / 不能可靠回答——每条回复都会标明来源。" if
        chat_service.model_loaded else
        "N-MIND Bot V1 已启动（无本地语言权重）：聊天、检索、工具功能可用；"
        "神经侧如实标注为不可用，不调用外部大模型。")
    root.mainloop()
    return 0
