"""N-MIND Bot V1 本地 Web 服务：HTTP UI 的唯一后端。

安全边界（任务书 §5/§7，Web 服务同样必须遵守）：
  * **只绑定 127.0.0.1 回环地址**——浏览器通过本地 HTTP 访问，零公网暴露；
    任何本机其他进程都可访问，故所有写操作（站点授权/控制/开关）都要求
    来自本机用户主动提交，且权限仍由 SourceRegistry/policy 二次校验；
  * 浏览器的 POST 只进入本服务的 JSON 接口，**绝不转发**到外部站点；
    联网动作（抓取/探索）仍走 explorer，受白名单+授权+配额+安全传输约束；
  * 服务线程驱动 BotRuntime.run_forever(chat_service, explorer)：与
    Tkinter 模式同一运行时、同一数据库，不存在第二套调度器。

接口契约（与 UI README 建议一致）：
  GET  /                     静态页面
  GET  /api/runtime/status   真实状态（状态/模型/网络/队列/配额）
  POST /api/chat/sessions        创建会话 {title}
  POST /api/chat/sessions/{id}/messages  {text} → message_id
  GET  /api/chat/messages/{message_id}  领取对应回复
  GET  /api/chat/sessions/{id}/messages 历史
  DELETE /api/chat/sessions/{id}
  GET  /api/knowledge/stats  POST /api/knowledge/search {q}
  POST /api/knowledge/import-fixture
  GET  /api/web/status       POST /api/web/sites {host,path_scope}
  DELETE /api/web/sites/{host}
  POST /api/web/explore {enabled}  POST /api/web/fetch {url}
  POST /api/control {action}  GET /api/activity
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from .chat_service import ChatRequest

WEB_DIR = Path(__file__).resolve().parent / "ui" / "web"
MAX_BODY = 64 * 1024                              # 请求体上限（防超大 POST）


class _Handler(BaseHTTPRequestHandler):
    server_version = "NMindBotV1/1.0"
    protocol_version = "HTTP/1.1"

    # ---------------------------------------------------------------- 工具
    @property
    def api(self) -> WebBotServer:
        return self.server.api                      # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:
        # 日志最小化：只记方法/路径/码，不记请求体（隐私链）；格式化交给 stdlib
        msg = fmt % args if args else fmt
        print(f"[web] {msg}")

    def _json(self, code: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            raise ValueError(f"请求体超过 {MAX_BODY} 字节上限")
        raw = self.rfile.read(length) if length else b""
        if not raw:
            return {}
        try:
            data = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise ValueError(f"JSON 解析失败：{exc}") from exc
        if not isinstance(data, dict):
            raise TypeError("JSON 必须是对象")
        return data

    # ---------------------------------------------------------------- 路由
    def do_GET(self) -> None:
        try:
            parsed = urlparse(self.path)
            self._route_get(parsed.path, parse_qs(parsed.query))
        except Exception as exc:                    # noqa: BLE001 - API 错误完整返回
            self._json(500, {"error": f"{type(exc).__name__}: {exc}"})

    def _route_get(self, path: str, query: dict[str, list[str]] | None = None) -> None:
        if path == "/" or path == "/index.html":
            self._static("index.html", "text/html; charset=utf-8")
            return
        if path.startswith("/assets/"):
            name = path[len("/assets/"):].replace("/", "")
            if ".." in name or "\\" in name:
                self._json(403, {"error": "非法资源路径"})
                return
            self._static(f"assets/{name}", None)
            return
        if path == "/api/runtime/status":
            self._json(200, self.api.health_payload())
            return
        # ---- 真实 8B 后端只读接口（数据全部来自本地推理/日志，绝不虚构）----
        if path == "/api/model/info":
            self._json(200, self.api.model_info())
            return
        if path == "/api/neural/state":
            self._json(200, self.api.neural_state())
            return
        if path == "/api/resource/state":
            self._json(200, self.api.resource_state())
            return
        if path == "/api/learn/state":
            self._json(200, self.api.learn_state())
            return
        if path == "/api/teachers/status":
            self._json(200, self.api.teachers_status())
            return
        if path == "/api/cloud/status":
            self._json(200, self.api.cloud_status())
            return
        if path == "/api/v3/status":
            self._json(200, self.api.v3.status())
            return
        if path == "/api/v3/events":
            self._json(200, {"events": self.api.v3.events.list()})
            return
        if path == "/api/v3/tasks":
            self._json(200, {"tasks": self.api.v3.tasks.list()})
            return
        if path.startswith("/api/v33/learning/"):
            action = path[len("/api/v33/learning/"):]
            handlers = {"status": self.api.v3.v33_status, "goals": self.api.v3.v33_goals, "decision-explanation": self.api.v3.v33_decision, "reviews": self.api.v3.v33_reviews, "failures": self.api.v3.v33_failures, "growth": self.api.v3.growth, "events": self.api.v3.v33_events, "policy": self.api.v3.v33_policy}
            if action in handlers: self._json(200, handlers[action]()); return
        if path == "/api/v34/knowledge/status": self._json(200, self.api.v3.v34_status()); return
        if path == "/api/v34/knowledge/entities": self._json(200, {"entities": self.api.v3.graph.entities((query or {}).get("q", [""])[0])}); return
        if path.startswith("/api/v34/knowledge/entities/") and path.endswith("/relations"):
            eid=path[len("/api/v34/knowledge/entities/"):-len("/relations")]; self._json(200,{"entity_id":eid,"relations":self.api.v3.graph.relations(eid,2)}); return
        if path == "/api/v34/knowledge/assertions": self._json(200,{"assertions":self.api.v3.graph.assertions()}); return
        if path == "/api/v34/knowledge/conflicts": self._json(200,{"conflicts":self.api.v3.graph.conflicts()}); return
        if path == "/api/v34/knowledge/query": self._json(200,self.api.v3.graph.query((query or {}).get("q", [""])[0])); return
        if path == "/api/v34/experiences": self._json(200,{"experiences":self.api.v3.graph.experiences()}); return
        if path == "/api/v34/knowledge/events": self._json(200,{"events":self.api.v3.events.list()}); return
        if path == "/api/v35/plans": self._json(200,{"plans":self.api.v35_store.list()}); return
        if path.startswith("/api/v35/plans/") and path.endswith("/events"):
            pid=path[len("/api/v35/plans/"):-len("/events")]; self._json(200,{"plan_id":pid,"events":self.api.planning_store.events(pid)}); return
        if path == "/api/status":
            self._json(200, self.api.v3.status()); return
        if path == "/api/events":
            self._json(200, {"events": self.api.v3.events.list()}); return
        if path == "/api/growth":
            self._json(200, self.api.v3.growth()); return
        if path == "/api/permissions":
            self._json(200, self.api.v3.permissions()); return
        if path == "/api/learning/goals":
            self._json(200, {"goals": [{"goal_id": "goal-arithmetic", "reason": "能力台账缺口/可由独立Oracle验证", "source": "rule"}]}); return
        if path == "/api/learning/cycles":
            self._json(200, {"cycles": self.api.v3.learning_cycles()}); return
        if path == "/api/capabilities":
            self._json(200, self.api.v3.growth()); return
        if path == "/api/knowledge/stats":
            self._json(200, self.api.knowledge_stats())
            return
        if path == "/api/memories":
            q = (query or {}).get("query", [""])[0]
            self._json(200, self.api.memory_search(q))
            return
        if path.startswith("/api/planning/") and path.endswith("/events"):
            task_id = path[len("/api/planning/"):-len("/events")]
            self._json(200, {"task_id": task_id, "events": self.api.planning_store.events(task_id)})
            return
        if path == "/api/web/status":
            self._json(200, self.api.web_status())
            return
        if path == "/api/activity":
            self._json(200, self.api.activity())
            return
        if path == "/api/chat/sessions":
            self._json(200, {"sessions": self.api.sessions_list()})
            return
        if path.startswith("/api/chat/sessions/"):
            tail = path[len("/api/chat/sessions/"):]
            if tail.endswith("/messages"):
                sid = tail[: -len("/messages")]
                self._json(200, {"session_id": sid,
                                 "messages": self.api.session_history(sid)})
                return
            sid = tail
            self._json(200, {"sessions": self.api.sessions_list()})
            return
        if path.startswith("/api/chat/messages/"):
            mid = path[len("/api/chat/messages/"):]
            rec = self.api.poll_reply(mid)
            if rec is None:
                self._json(200, {"status": "PENDING", "message_id": mid,
                                 "note": "消息正在处理，稍后重试"})
            else:
                self._json(200, rec)
            return
        self._json(404, {"error": "未知路径"})

    def _static(self, rel: str, ctype: str | None) -> None:
        p = (WEB_DIR / rel)
        try:
            resolved = p.resolve()
            web_root = WEB_DIR.resolve()
        except OSError:
            self._json(404, {"error": "资源不存在"})
            return
        if resolved != web_root and web_root not in resolved.parents:
            self._json(403, {"error": "非法资源路径"})
            return
        if not p.is_file():
            self._json(404, {"error": "资源不存在"})
            return
        data = p.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype or _guess_ctype(p.name))
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:
        try:
            if urlparse(self.path).path == "/api/chat/stream":
                self._chat_stream()
                return
            self._route_post(urlparse(self.path).path, self._read_body())
        except ValueError as exc:
            self._json(400, {"error": str(exc)})
        except Exception as exc:                    # noqa: BLE001
            self._json(500, {"error": f"{type(exc).__name__}: {exc}"})

    def _chat_stream(self) -> None:
        """真实 SSE：消费 8B chat_stream 的结构化事件，逐 chunk 推送。

        与 8765 独立服务同口径：think/answer/done/truncated 状态机在神经侧完成，
        这里只做传输层，不改写内容、不伪装思考链、不虚构答案。
        """
        body = self._read_body()
        sid = str(body.get("session_id") or "stream-default")
        request_id = str(body.get("request_id") or "")
        text = str(body.get("text") or "").strip()
        supplied_messages = body.get("messages") if isinstance(body.get("messages"), list) else None
        max_new_tokens = max(1, int(body.get("max_new_tokens") or 2048))
        keep_thinking = bool(body.get("keep_thinking", False))
        if not text:
            self._json(400, {"error": "text 不能为空"})
            return
        svc = getattr(self.api, "chat_service", None)
        if svc is None or not hasattr(svc, "stream_reply"):
            self._json(503, {"error": "当前后端不支持真实流式推理"})
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        try:
            queued = {"state":"queued","source":"neural","request_id":request_id}
            self.wfile.write(f"data: {json.dumps(queued, ensure_ascii=False)}\n\n".encode("utf-8"))
            self.wfile.flush()
            for ev in svc.stream_reply(sid, text, keep_thinking=keep_thinking,
                                       supplied_messages=supplied_messages,
                                       max_new_tokens=max_new_tokens):
                ev = {**ev, "request_id": request_id}
                payload = json.dumps(ev, ensure_ascii=False)
                self.wfile.write(f"data: {payload}\n\n".encode("utf-8"))
                self.wfile.flush()
            closed = {"state": "closed", "request_id": request_id}
            self.wfile.write(f"data: {json.dumps(closed, ensure_ascii=False)}\n\n".encode("utf-8"))
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as exc:                    # noqa: BLE001
            err = json.dumps({"error": f"{type(exc).__name__}: {exc}"[:300], "state":"error",
                              "finish_reason":"error"}, ensure_ascii=False)
            try:
                self.wfile.write(f"data: {err}\n\n".encode("utf-8"))
                self.wfile.flush()
            except Exception:
                pass

    def _route_post(self, path: str, body: dict[str, Any]) -> None:
        # 正式产品唯一聊天入口是 /api/chat/stream（DeepSeek 8B）。
        # 旧 N-Cell 队列和旧 v3 直聊保留历史代码但从活动 API 硬隔离，避免旧页面再次调用。
        if path == "/api/v3/chat" or (path.startswith("/api/chat/sessions/") and path.endswith("/messages")):
            self._json(410, {"error": "旧 N-Cell 聊天路径已停用，请使用 /api/chat/stream", "active_model": "N-MIND"})
            return
        if path == "/api/cloud/configure":
            self._json(200, self.api.configure_cloud(body)); return
        if path == "/api/cloud/test":
            self._json(200, self.api.test_cloud()); return
        if path == "/api/cloud/model":
            self._json(200, self.api.select_cloud_model(str(body.get("model") or ""))); return
        if path == "/api/v3/chat":
            sid = str(body.get("session_id") or "v3-default")
            text = str(body.get("text") or "").strip()
            if not text:
                self._json(400, {"error": "text 不能为空"}); return
            self._json(200, self.api.v3.chat_reply(sid, text)); return
        if path.startswith("/api/chat/sessions/") and path.endswith("/direct"):
            sid = path[len("/api/chat/sessions/"):-len("/direct")]
            text = str(body.get("text") or "").strip()
            if not text:
                self._json(400, {"error": "text cannot be empty"}); return
            self._json(200, self.api.direct_message(sid, text)); return
        if path == "/api/v3/learn":
            self._json(200, self.api.v3.learn_once()); return
        if path == "/api/learning/cycles":
            self._json(200, self.api.v3.learn_once()); return
        if path == "/api/learning/pause":
            self._json(200, self.api.v3.set_learning(False)); return
        if path == "/api/learning/resume":
            self._json(200, self.api.v3.set_learning(True)); return
        if path == "/api/learning/stop":
            self._json(200, self.api.v3.set_learning(False)); return
        if path == "/api/memories":
            self._json(201, self.api.v3.memory_create(body)); return
        if path.startswith("/api/memories/") and path.endswith("/correct"):
            mid = int(path[len("/api/memories/"):-len("/correct")])
            self._json(200, self.api.v3.memory_correct(mid, body)); return
        if path == "/api/v3/learning":
            self._json(200, self.api.v3.set_learning(bool(body.get("enabled")))); return
        if path.startswith("/api/v33/learning/"):
            action = path[len("/api/v33/learning/"):]
            handlers = {"run-once": self.api.v3.v33_run_once, "pause": self.api.v3.v33_pause, "resume": self.api.v3.v33_resume, "stop": self.api.v3.v33_stop}
            if action == "start-session": self._json(200, self.api.v3.set_learning(True)); return
            if action in handlers: self._json(200, handlers[action]()); return
        if path == "/api/v34/knowledge/entities": self._json(201,self.api.v3.v34_add_entity(body)); return
        if path == "/api/v34/knowledge/sources/import-local": self._json(201,self.api.v3.v34_add_evidence(body)); return
        if path == "/api/v34/knowledge/assertions": self._json(201,self.api.v3.v34_add_assertion(body)); return
        if path.startswith("/api/v34/knowledge/conflicts/") and path.endswith("/resolve"):
            cid=path[len("/api/v34/knowledge/conflicts/"):-len("/resolve")]; self._json(200,self.api.v3.graph.resolve_conflict(cid,str(body.get("resolution") or ""))); return
        if path.startswith("/api/v34/experiences/") and path.endswith("/revoke"):
            eid=path[len("/api/v34/experiences/"):-len("/revoke")]; self._json(200,self.api.v3.graph.revoke_experience(eid)); return
        if path == "/api/v35/plans/preview": self._json(201,self.api.v35_preview(str(body.get("request") or ""))); return
        if path.startswith("/api/v35/plans/") and path.endswith("/approve"):
            pid=path[len("/api/v35/plans/"):-len("/approve")]; self._json(200,self.api.v35_approve(pid)); return
        if path.startswith("/api/v35/plans/") and path.endswith("/execute"):
            pid=path[len("/api/v35/plans/"):-len("/execute")]; self._json(200,self.api.v35_execute(pid)); return
        if path == "/api/v3/tasks":
            title = str(body.get("title") or "V3任务")
            steps = body.get("steps") if isinstance(body.get("steps"), list) else []
            self._json(201, self.api.v3.tasks.create(title, steps).to_dict()); return
        if path.startswith("/api/v3/tasks/") and path.endswith("/transition"):
            tid = path[len("/api/v3/tasks/"):-len("/transition")]
            self._json(200, self.api.v3.tasks.transition(tid, str(body.get("status") or ""), str(body.get("error") or "")).to_dict()); return
        if path == "/api/chat/sessions":
            title = str(body.get("title") or "新会话")[:40]
            self._json(201, {"session_id": self.api.create_session(title)})
            return
        if path.startswith("/api/chat/sessions/") and path.endswith("/messages"):
            sid = path[len("/api/chat/sessions/"):-len("/messages")]
            text = str(body.get("text") or "").strip()
            if not text:
                self._json(400, {"error": "text 不能为空"})
                return
            out = self.api.submit_message(sid, text)
            self._json(202, out)                    # 已入队：需用 message_id 领取回复
            return
        if path == "/api/knowledge/search":
            q = str(body.get("q") or "").strip()
            self._json(200, {"query": q, "results": self.api.knowledge_search(q,
                                                                            limit=int(body.get("limit") or 5))})
            return
        if path == "/api/knowledge/import-fixture":
            self._json(200, self.api.import_fixture())
            return
        if path.startswith("/api/memories/") and path.endswith("/feedback"):
            mid = int(path[len("/api/memories/"):-len("/feedback")])
            self._json(200, self.api.memory_feedback(mid, body))
            return
        if path == "/api/planning/goals":
            request = str(body.get("request") or "").strip()
            if not request:
                self._json(400, {"error": "request 不能为空"})
                return
            self._json(202, self.api.create_planning_goal(request))
            return
        if path == "/api/planning/control":
            self._json(200, self.api.planning_control(str(body.get("action") or "")))
            return
        if path == "/api/web/sites":
            host = str(body.get("host") or "").strip().lower().rstrip("/")
            scope = str(body.get("path_scope") or "")
            self._json(201, self.api.add_site(host, scope))
            return
        if path == "/api/web/explore":
            self._json(200, self.api.set_exploring(bool(body.get("enabled"))))
            return
        if path == "/api/web/fetch":
            url = str(body.get("url") or "").strip()
            self._json(200, self.api.user_fetch(url))
            return
        if path == "/api/control":
            action = str(body.get("action") or "")
            self._json(200, self.api.control(action))
            return
        self._json(404, {"error": "未知路径"})

    def do_PATCH(self) -> None:
        try:
            path = urlparse(self.path).path; body = self._read_body()
            if path.startswith("/api/v34/knowledge/assertions/"):
                aid = path[len("/api/v34/knowledge/assertions/"):]
                if body.get("revoked") is True:
                    self._json(200, self.api.v3.graph.revoke_assertion(aid, "user_patch")); return
            self._json(404, {"error": "unknown path"})
        except Exception as exc: self._json(400, {"error": f"{type(exc).__name__}: {exc}"})

    def do_DELETE(self) -> None:
        path = urlparse(self.path).path
        try:
            if path.startswith("/api/web/sites/"):
                host = path[len("/api/web/sites/"):]
                self._json(200, {"removed": self.api.remove_site(host)})
                return
            if path.startswith("/api/chat/sessions/"):
                sid = path[len("/api/chat/sessions/"):]
                n = self.api.delete_session(sid)
                self._json(200, {"deleted": sid, "messages_removed": n})
                return
            if path.startswith("/api/memories/") and path.endswith("/correct"):
                self._json(405, {"error": "use POST /api/memories/{id}/correct"}); return
            if path.startswith("/api/memories/"):
                mid = int(path[len("/api/memories/"):])
                self._json(200, self.api.v3.memory_delete(mid)); return
            self._json(404, {"error": "未知路径"})
        except Exception as exc:                    # noqa: BLE001
            self._json(500, {"error": f"{type(exc).__name__}: {exc}"})


def _guess_ctype(name: str) -> str:
    if name.endswith(".png"):
        return "image/png"
    if name.endswith((".jpg", ".jpeg")):
        return "image/jpeg"
    if name.endswith(".html"):
        return "text/html; charset=utf-8"
    if name.endswith(".css"):
        return "text/css; charset=utf-8"
    if name.endswith(".js"):
        return "application/javascript"
    return "application/octet-stream"


class WebBotServer:
    """HTTP UI 服务 + 同一 BotRuntime 服务线程。"""

    def __init__(self, *, root_dir, client, chat_service, knowledge, explorer,
                 offline: bool, host: str = "127.0.0.1", port: int = 0) -> None:
        self.client = client
        self.chat_service = chat_service
        self.knowledge = knowledge
        self.explorer = explorer
        self.offline = offline
        self.host, self.port = host, int(port)
        self._httpd: ThreadingHTTPServer | None = None
        self._service_thread: threading.Thread | None = None
        self._url: str = ""
        self.runtime: Any = None
        from ..v3 import V3Facade
        self.v3 = V3Facade(root_dir, chat=chat_service,
                           memory=getattr(chat_service, "memory_service", None),
                           knowledge=knowledge, runtime=self.runtime)
        from ..planning import PlanningStore, PlanExecutor, RulePlanner
        from ..planning.tool_adapter import build_tools
        self.planning_store = PlanningStore(Path(root_dir) / "planning.db")
        self.planner = RulePlanner()
        self.planning_executor = PlanExecutor(self.planning_store, build_tools(knowledge=knowledge, memory=getattr(chat_service, "memory_service", None)), memory=getattr(chat_service, "memory_service", None))
        from ..planning.v35_store import V35PlanStore
        from ..planning.safe_actions import SafeActionRegistry
        self.v35_store = V35PlanStore(Path(root_dir) / "v35_plans.db")
        self.v35_actions = SafeActionRegistry(workspace=Path(root_dir), graph=getattr(self.v3, "graph", None), knowledge=knowledge, experiences=getattr(self.v3, "graph", None).experiences if getattr(self.v3, "graph", None) else None)
        self.v35_tools = self.v35_actions.tools()
        self.v35_tools.update(build_tools(knowledge=knowledge, memory=getattr(chat_service, "memory_service", None)))
        self.cloud_config = {"enabled": False, "provider": "off", "model": "", "key_configured": False}
        self._build_runtime(root_dir, offline)

    def _build_runtime(self, root_dir, offline) -> None:
        from pathlib import Path

        from .policy import BotPolicy

        # 与 GUI/CLI 相同的组装：优先外部已跑实例（只连接），否则本进程起服务
        from .runtime import BotRuntime, read_status
        lock_exists = Path(root_dir).joinpath("lock.json").exists()
        external = read_status(root_dir).get("running", False) if lock_exists else False
        if external:
            return                          # 只连接：不建第二份运行时、不碰锁
        policy = BotPolicy()
        if offline:
            policy.network_enabled = False
        self.runtime = BotRuntime(root_dir=root_dir, policy=policy)
        if hasattr(self, "v3"):
            self.v3.runtime = self.runtime

    # ---------------------------------------------------------------- 生命周期
    def start(self) -> str:
        """启动 HTTP 服务与服务线程，返回本地访问 URL。"""
        if self._httpd is not None:
            return self._url
        self._httpd = ThreadingHTTPServer((self.host, self.port), _Handler)
        self._httpd.api = self                      # type: ignore[attr-defined]
        port = self._httpd.server_address[1]
        self._url = f"http://127.0.0.1:{port}/"
        self._http_thread = threading.Thread(
            target=self._httpd.serve_forever, daemon=True, name="nmind-web-http")
        self._http_thread.start()
        if self.runtime is not None:
            self._service_thread = threading.Thread(
                target=self._service_main, daemon=True, name="nmind-web-service")
            self._service_thread.start()
        return self._url

    def _service_main(self) -> None:
        try:
            self.runtime.run_forever(chat_service=self.chat_service,
                                     explorer=self.explorer, poll_s=0.3)
        except Exception as exc:                    # noqa: BLE001
            print(f"[web] 服务线程异常：{type(exc).__name__}: {exc}")

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        if hasattr(self, "v35_store"):
            self.v35_store.close()
        if self.runtime is not None and self.runtime.state.name != "STOPPED":
            # 先发停止信号让驻留循环在安全边界退出，再关库——
            # 顺序颠倒会触发"Cannot operate on a closed database"
            from .runtime import request_control
            request_control(self.client.root, "stop")
            if self._service_thread is not None:
                self._service_thread.join(timeout=5.0)
            self.runtime.stop("Web UI 关闭")

    @property
    def serving(self) -> bool:
        return self._httpd is not None

    # ---------------------------------------------------------------- API
    def health_payload(self) -> dict[str, Any]:
        st: dict[str, Any] = {"running": self.serving, "state": "?",
                              "offline": self.offline,
                              "network_enabled": False}
        if self.runtime is not None:
            st["state"] = self.runtime.state.name
            st["network_enabled"] = bool(self.runtime.policy.network_enabled)
        else:
            from .runtime import read_status
            st["state"] = str(read_status(self.client.root).get("state", "?"))
        st["model_loaded"] = self.chat_service.model_loaded
        st["model_sha"] = self.chat_service.model_sha
        st["vocab_size"] = self.chat_service.tok.vocab_size if self.chat_service.tok else None
        st["generation_protocol_id"] = "deepseek-r1-qwen3-8b-sse"
        try:
            st.update(self.knowledge.stats())
        except Exception as exc:                    # noqa: BLE001
            print(f"[web] 知识统计刷新失败（状态页降级显示）：{type(exc).__name__}")
        st["quota"] = self.explorer.quota_snapshot()
        return st

    # ---------------------------------------------------------------- 真实神经后端状态
    def model_info(self) -> dict[str, Any]:
        """真实模型身份与加载状态；未加载时如实说明，不用旧小模型冒充。"""
        svc = self.chat_service
        if hasattr(svc, "load_info"):
            return svc.load_info()
        return {"status": "LEGACY_MODEL", "model_loaded": bool(getattr(svc, "model_loaded", False)),
                "note": "当前为工作台自带旧语言模型；8B 适配器未接入。"}

    def neural_state(self) -> dict[str, Any]:
        svc = self.chat_service
        if hasattr(svc, "neural_snapshot"):
            return svc.neural_snapshot()
        return {"status": "NOT_AVAILABLE", "note": "当前后端不支持神经活动采集。"}

    def resource_state(self) -> dict[str, Any]:
        svc = self.chat_service
        if hasattr(svc, "resource_status"):
            return svc.resource_status()
        return {"status": "NOT_AVAILABLE"}

    def learn_state(self) -> dict[str, Any]:
        """学习工作台状态：直接复用 learning_workbench 的账本/日志读取逻辑。

        不加载模型、不依赖 8766 是否启动；数据来自 product/ 下的真实文件，
        未训练时明确 NOT_TRAINED，绝不虚构成长曲线。
        """
        import sys
        from pathlib import Path
        product = Path(__file__).resolve().parents[3] / "product"
        if str(product) not in sys.path:
            sys.path.insert(0, str(product))
        try:
            from learning_workbench import state as _empty  # noqa: F401
        except Exception:
            pass
        # 直接复刻 learning_workbench.state 的读取口径，保持单一数据源
        def read_json(p: Path, default=None):
            if p.exists():
                try:
                    return json.loads(p.read_text(encoding="utf-8"))
                except Exception:
                    return default
            return default

        ledger_dir = product / "budgets"
        ledgers = sorted(ledger_dir.glob("LEDGER_QLORA_*.json")) if ledger_dir.exists() else []
        latest = read_json(ledgers[-1]) if ledgers else None
        adapters_dir = product / "checkpoints" / "adapters"
        adapters = [str(x.name) for x in sorted(adapters_dir.glob("adapter_*"))] \
            if adapters_dir.exists() else []
        check = read_json(product / "logs" / "QLORA_STATIC_ZERO_UPDATE_CHECK.json")
        cache_report = read_json(product / "logs" / "CACHE_BUILD_REPORT.json")
        budget = read_json(ledger_dir / "BUDGET_QLORA_R1.json")
        return {
            "status": "TRAINED" if latest else "NOT_TRAINED",
            "message": "已有真实学习记录" if latest else "尚未执行训练（0 次 optimizer.step）",
            "budget_status": budget.get("status") if budget else None,
            "latest_ledger": latest, "adapters": adapters,
            "zero_update_check": check, "cache_report": cache_report,
            "note": "所有数据来自真实日志/账本；未训练时不显示任何虚构曲线。",
        }

    def teachers_status(self) -> dict[str, Any]:
        """统一教师引擎状态；无凭据时降级为方法参考，不阻塞学生独立回答。"""
        import sys
        from pathlib import Path
        product = Path(__file__).resolve().parents[3] / "product"
        if str(product) not in sys.path:
            sys.path.insert(0, str(product))
        try:
            from teacher_unified import UnifiedTeacher
            ut = UnifiedTeacher.from_env()
            return {"teachers": [s.__dict__ for s in ut.status()],
                    "note": "教师关闭后学生仍由本地 8B 独立回答。"}
        except Exception as exc:  # noqa: BLE001
            return {"teachers": [], "error": f"{type(exc).__name__}: {exc}",
                    "note": "教师系统状态读取失败，如实报告。"}

    def cloud_status(self) -> dict[str, Any]:
        return {**self.cloud_config, "network_enabled": not self.offline,
                "source": "in_memory_only", "training_target": False}

    def configure_cloud(self, body: dict[str, Any]) -> dict[str, Any]:
        if self.offline:
            return {"ok": False, "error": "offline_mode_enabled"}
        from ..bridge.providers import DeepSeekProvider
        enabled = bool(body.get("enabled"))
        key = str(body.get("api_key") or "")
        model = str(body.get("model") or "deepseek-chat")
        if not enabled:
            self.chat_service.cloud_provider = None
            self.cloud_config = {"enabled": False, "provider": "off", "model": "", "key_configured": False}
            return {"ok": True, **self.cloud_config}
        if not key:
            return {"ok": False, "error": "api_key_required"}
        self.chat_service.cloud_provider = DeepSeekProvider(
            base_url="https://api.deepseek.com", api_key=key, model=model,
            timeout_s=20.0, max_retries=1, max_payload_chars=4000)
        self.cloud_config = {"enabled": True, "provider": "deepseek", "model": model,
                             "key_configured": True}
        return {"ok": True, **self.cloud_config}

    def test_cloud(self) -> dict[str, Any]:
        provider = getattr(self.chat_service, "cloud_provider", None)
        if provider is None:
            return {"ok": False, "stage": "configuration", "error": "not_configured"}
        from ..bridge.providers import CloudRequest
        started = time.perf_counter()
        try:
            result = provider.ask(CloudRequest(category="public_web_query", question="请只回复OK。", max_response_chars=32))
            return {"ok": bool(result.ok), "stage": "provider", "provider": result.provider,
                    "model": self.cloud_config.get("model", ""), "latency_ms": round((time.perf_counter()-started)*1000, 2),
                    "error": result.error, "response_chars": len(result.text)}
        except Exception as exc:
            return {"ok": False, "stage": "exception", "error": f"{type(exc).__name__}: {exc}"}

    def select_cloud_model(self, model: str) -> dict[str, Any]:
        provider = getattr(self.chat_service, "cloud_provider", None)
        if provider is None:
            return {"ok": False, "error": "not_configured"}
        if getattr(provider, "name", "") != "deepseek":
            return {"ok": False, "error": "provider_not_deepseek"}
        if model not in {"deepseek-chat", "deepseek-reasoner"}:
            return {"ok": False, "error": "unsupported_model"}
        provider.model = model
        self.cloud_config["model"] = model
        return {"ok": True, "provider": "deepseek", "model": model}

    def knowledge_stats(self) -> dict[str, Any]:
        return self.knowledge.stats()

    def knowledge_search(self, q: str, *, limit: int = 5) -> list[dict[str, Any]]:
        return self.knowledge.search(q, limit=limit)

    def _memory_service(self):
        return getattr(self.chat_service, "memory_service", None)

    def memory_search(self, q: str) -> dict[str, Any]:
        service = self._memory_service()
        if service is None:
            return {"query": q, "hits": [], "status": "DISABLED"}
        return {"query": q, "hits": service.recall(q).hits, "status": "READY"}

    def memory_feedback(self, memory_id: int, body: dict[str, Any]) -> dict[str, Any]:
        service = self._memory_service()
        if service is None:
            return {"ok": False, "error": "Memory V2 未启用"}
        service.feedback(kind=str(body.get("kind") or "episode"), memory_id=memory_id,
                         value=float(body.get("value", 1)), task_id=str(body.get("task_id") or ""),
                         note=str(body.get("note") or ""))
        return {"ok": True, "memory_id": memory_id}

    def memory_forget(self, memory_id: int) -> bool:
        service = self._memory_service()
        return bool(service and service.forget(kind="episode", memory_id=memory_id))

    def import_fixture(self) -> dict[str, Any]:
        from .knowledge_ingest import import_fixture_pages
        return import_fixture_pages(self.knowledge.store)

    def web_status(self) -> dict[str, Any]:
        return {"registry": {"sites": self.explorer.registry.list(),
                             "exploring_enabled": self.explorer.registry.exploring_enabled,
                             "policy_hash": self.explorer.registry.policy_hash()},
                "limits": self.explorer.registry.limits,
                "offline": self.offline,
                "quota": self.explorer.quota_snapshot()}

    def add_site(self, host: str, path_scope: str = "") -> dict[str, Any]:
        return self.explorer.registry.add(host, path_scope=path_scope,
                                          created_by="web-ui-user")

    def remove_site(self, host: str) -> bool:
        return self.explorer.registry.remove(host)

    def set_exploring(self, enabled: bool) -> dict[str, Any]:
        self.explorer.set_enabled(enabled)
        self.explorer.registry.set_exploring(enabled)
        return {"exploring_enabled": enabled}

    def user_fetch(self, url: str) -> dict[str, Any]:
        return self.explorer.fetch_for_user(url)

    def control(self, action: str) -> dict[str, Any]:
        from .runtime import request_control
        if action not in ("pause", "resume", "stop"):
            return {"ok": False, "error": f"未知控制动作 {action!r}"}
        return request_control(self.client.root, action)

    def activity(self) -> dict[str, Any]:
        try:
            from .store import BotStore
            store = BotStore(self.client.root / "bot_state.db")
            events = store.list_events(limit=15)
            store.close()
            return {"events": events}
        except Exception as exc:                    # noqa: BLE001
            return {"events": [], "error": str(exc)}

    # ---------------------------------------------------------------- 聊天
    def create_planning_goal(self, request: str) -> dict[str, Any]:
        from ..planning import GoalSpec
        goal = GoalSpec.create(request)
        plan = self.planner.plan(goal)
        result = self.planning_executor.run(goal, plan)
        return {"goal_id": goal.goal_id, "plan_source": plan.source,
                "steps": [s.__dict__ for s in plan.steps], "result": result}

    def v35_preview(self, request: str) -> dict[str, Any]:
        from ..planning import GoalSpec
        from ..planning.contracts import Plan, PlanStep
        import uuid, hashlib
        goal=GoalSpec.create(request, budget={"max_steps":5,"max_tool_calls":8,"max_replans":1})
        tid=goal.goal_id; steps=[]
        def add(n,kind,obj,tool,deps=(),approval=False): steps.append(PlanStep(f"{tid[:8]}-{n}",tid,kind,obj,tuple(deps),tool,verifier_id="deterministic_oracle",idempotency_key=hashlib.sha256(f"{tid}:{n}".encode()).hexdigest()[:16],max_attempts=2,requires_approval=approval))
        low=request.casefold()
        calc = any(x in request for x in ("计算", "+", "-", "*", "/"))
        if calc: add(1,"calculator","执行确定性计算","calculator")
        elif "读取" in request or "read" in low: add(1,"read_file","读取授权工作区文件","read_local_file",approval=False)
        if not calc:
            add(len(steps)+1,"knowledge_query","查询本地知识与证据","knowledge_query",tuple(s.step_id for s in steps))
            add(len(steps)+1,"experience_search","检索经过验证的历史经验","experience_search",tuple(s.step_id for s in steps))
        add(len(steps)+1,"structured_report","生成结构化报告","structured_report",tuple(s.step_id for s in steps))
        plan=Plan(tid,goal.goal_id,1,"v35_rule",tuple(steps)); payload={"goal":goal.__dict__,"plan":{"task_id":plan.task_id,"goal_id":plan.goal_id,"version":1,"source":plan.source,"steps":[s.__dict__ for s in plan.steps]},"approval_scope":"local read/query/report only","side_effects":[]}
        self.v35_store.save(tid,payload); self.planning_store.event(tid,"v35_plan_preview",payload); return {"plan_id":tid,"status":"AWAITING_APPROVAL","requires_approval":True,**payload}
    def v35_approve(self, plan_id: str): return self.v35_store.set_status(plan_id,"APPROVED",True)
    def v35_execute(self, plan_id: str):
        rec=self.v35_store.get(plan_id)
        if not rec: return {"status":"NOT_FOUND"}
        if not rec["approved"]: return {"status":"PERMISSION_BLOCKED","reason":"USER_APPROVAL_REQUIRED"}
        from ..planning import GoalSpec, Plan, PlanStep, PlanExecutor
        g=GoalSpec(**rec["payload"]["goal"]); p=rec["payload"]["plan"]; plan=Plan(p["task_id"],p["goal_id"],p["version"],p["source"],tuple(PlanStep(**s) for s in p["steps"]))
        ex=PlanExecutor(self.planning_store,self.v35_tools,max_replans=1,memory=getattr(self.chat_service,"memory_service",None)); result=ex.run(g,plan); self.v35_store.set_status(plan_id,result.get("status","UNKNOWN")); return {"plan_id":plan_id,"status":result.get("status"),"result":result}

    def planning_control(self, action: str) -> dict[str, Any]:
        if action == "pause": self.planning_executor.pause()
        elif action == "resume": self.planning_executor.resume()
        elif action == "cancel": self.planning_executor.cancel()
        else: return {"ok": False, "error": "unknown planning action"}
        return {"ok": True, "action": action}

    def create_session(self, title: str) -> str:
        sid = self.client.new_session_id()
        self.client.create_session(sid, title)
        return sid

    def sessions_list(self) -> list[dict[str, Any]]:
        return self.client.sessions()

    def session_history(self, sid: str) -> list[dict[str, Any]]:
        return self.client.history(sid, limit=200)

    def delete_session(self, sid: str) -> int:
        return self.client.delete_session(sid)

    def submit_message(self, sid: str, text: str) -> dict[str, Any]:
        return self.client.submit(text, session_id=sid)

    def direct_message(self, sid: str, text: str) -> dict[str, Any]:
        """Run one answer in this server's ChatService, including the selected cloud provider.

        The legacy queue can be serviced by a runtime created before cloud configuration.
        This synchronous, bounded path keeps model selection and the configured provider in
        the same process, then persists the paired user/assistant rows for reload recovery.
        """
        req = ChatRequest.new(sid, text)
        self.client._store.enqueue_message(req.message_id, sid, text)
        resp = self.chat_service.reply(req, prefer_cloud=True)
        self.client._store.finish_message(
            req.message_id, reply_text=resp.text, response_source=resp.response_source,
            citations=[c.to_dict() for c in resp.citations], model_sha=resp.model_sha,
            protocol_id=resp.protocol_id, latency_ms=resp.latency_ms, note=resp.note,
            extra={"unk_rate": resp.unk_rate, "finish_reason": resp.finish_reason,
                   "memory_hits": resp.memory_hits,
                   "learning_target": "external_teacher_candidate" if resp.response_source == "cloud_teacher" else "none",
                   "training_applied": False},
        )
        return {"message_id": req.message_id, "status": "DONE", "text": resp.text,
                "response_source": resp.response_source,
                "citations": [c.to_dict() for c in resp.citations],
                "latency_ms": resp.latency_ms, "note": resp.note}

    def poll_reply(self, message_id: str) -> dict[str, Any] | None:
        rec = self.client.message(message_id)
        if rec is None:
            return None
        rec["status"] = rec.get("status") or "DONE"
        return rec


def launch_web_ui(args) -> int:
    """`bot ui` 默认模式：启动本地 Web 工作台并打开浏览器。"""
    import webbrowser

    from .cli import _policy, _root
    from .client import BotClient
    from .knowledge_ingest import ensure_fixture_pages
    from .knowledge_search import KnowledgeSearch
    from .knowledge_store import KnowledgeStore
    from .source_registry import SourceRegistry
    from .ui.web_desktop import build_web_stack

    root_dir = _root(args)
    policy = _policy(args)
    os = __import__("os")
    cloud_explicit = os.environ.get("NMIND_ENABLE_CLOUD") == "1"
    # 网页探索需要独立于云端模型的联网开关；不能因为允许抓取官方公开资料
    # 就同时打开外部教师模型。默认仍关闭，只有用户显式设置才启用。
    network_explicit = os.environ.get("NMIND_ENABLE_NETWORK") == "1"
    if cloud_explicit or network_explicit:
        policy.network_enabled = True
    if cloud_explicit:
        policy.allow_remote_model_api = True
    offline = bool(getattr(args, "offline", False)) or (not policy.network_enabled and not cloud_explicit)
    if offline:
        policy.network_enabled = False
    if getattr(args, "no_idle", False):
        policy.idle_learning_enabled = False

    client = BotClient(root_dir)
    knowledge = KnowledgeSearch(KnowledgeStore(root_dir / "knowledge.db"))
    registry = SourceRegistry(root_dir / "sources.json")
    # 保留原有探索器/知识库栈；把聊天神经后端换成真实 DeepSeek 8B。
    # 云教师 provider 若显式配置仍可经 ChatService 路由（标注来源，不计神经能力）。
    from .deepseek8b_service import DeepSeek8BChatService
    from .idle_explorer import IdleWebExplorer
    from ..memory import MemoryService, MemoryStore
    from ..bridge.providers import build_provider
    from ..config import load_config
    cfg = load_config(getattr(args, "config", None))
    if offline:
        cfg = cfg.model_copy(update={"offline_forced": True})
    cloud_provider = None if cfg.effective_cloud_mode() == "off" else build_provider(
        cfg.effective_cloud_mode(), cfg)
    memory_service = MemoryService(MemoryStore(root_dir / "memory.db"))
    chat_service = DeepSeek8BChatService(
        knowledge=knowledge, memory_service=memory_service,
        cloud_provider=cloud_provider,
        history_provider=lambda sid: client.history(sid, limit=32),
        lazy_load=True,
    )
    explorer = IdleWebExplorer(registry=registry, knowledge_store=knowledge.store,
                               policy=policy, offline=offline)
    ensure_fixture_pages(knowledge.store)

    server = WebBotServer(root_dir=root_dir, client=client, chat_service=chat_service,
                          knowledge=knowledge, explorer=explorer, offline=offline,
                          port=int(getattr(args, "port", 8765)))
    url = server.start()
    print(f"N-MIND Bot V1 已启动：{url}（仅绑定本机回环地址，零公网暴露）")
    print("按 Ctrl+C 或窗口标题栏关闭即可停止；真实联网默认关闭，需配置+逐站点授权。")
    try:
        import webbrowser
        webbrowser.open(url)
    except Exception as exc:                      # noqa: BLE001
        print(f"浏览器自动打开失败（可手动访问 {url}）：{exc}")
    # 主线程驻留：等键盘 Ctrl+C 后干净退出
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n收到停止请求，正在释放服务…")
    finally:
        server.stop()
        client.close()
    return 0
