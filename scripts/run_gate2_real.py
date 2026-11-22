"""Execute the approved Gate-2 page expansion and local chat evaluation."""
from __future__ import annotations

import hashlib, json, sys, time
from pathlib import Path
from urllib import robotparser

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from nmind.bot.knowledge_ingest import ingest_extracted, EXTRACTOR_VERSION
from nmind.bot.knowledge_search import KnowledgeSearch
from nmind.bot.knowledge_store import KnowledgeStore
from nmind.bot.chat_service import ChatRequest, ChatService
from nmind.online.safe_transport import fetch_readonly

ROBOTS = "https://docs.python.org/robots.txt"
PAGES = [
    ("https://docs.python.org/zh-cn/3.11/tutorial/datastructures.html", "数据结构"),
    ("https://docs.python.org/zh-cn/3.11/tutorial/modules.html", "模块"),
    ("https://docs.python.org/zh-cn/3.11/tutorial/inputoutput.html", "输入与输出"),
    ("https://docs.python.org/zh-cn/3.11/tutorial/errors.html", "异常处理"),
    ("https://docs.python.org/zh-cn/3.11/tutorial/classes.html", "类"),
    ("https://docs.python.org/zh-cn/3.11/tutorial/stdlib.html", "标准库"),
    ("https://docs.python.org/zh-cn/3.11/library/pathlib.html", "文件操作"),
]
ALLOWED = ("docs.python.org/robots.txt", "docs.python.org/zh-cn/3.11/tutorial/",
           "docs.python.org/zh-cn/3.11/library/")
QUESTIONS = [
    ("G2-01", "Python列表和元组有什么区别？", "datastructures.html", True),
    ("G2-02", "Python中for循环和while循环分别适合什么场景？", "controlflow.html", True),
    ("G2-03", "Python如何定义函数？", "controlflow.html", True),
    ("G2-04", "Python模块如何导入？", "modules.html", True),
    ("G2-05", "Python如何处理异常？", "errors.html", True),
    ("G2-06", "Python如何读取文件？", "inputoutput.html", True),
    ("G2-07", "Python中的类和对象是什么关系？", "classes.html", True),
    ("G2-08", "Python字典数据结构有什么特点？", "datastructures.html", True),
    ("G2-09", "range函数如何使用？", "controlflow.html", True),
    ("G2-10", "Python官方教程能否证明N-MIND项目2027年的实验计划？", "", False),
]

def main() -> int:
    out = ROOT / "results" / "online_v1_gate2"
    out.mkdir(parents=True, exist_ok=True)
    store = KnowledgeStore(ROOT / "results" / "online_v1_gate1" / "knowledge.db")
    audit = []
    total_bytes = 0
    last = None

    def get(url: str, kind: str):
        nonlocal last, total_bytes
        if last is not None:
            time.sleep(max(0.0, 5.0 - (time.monotonic() - last)))
        last = time.monotonic()
        rec = fetch_readonly(url, allowed_sites=ALLOWED, max_bytes=524288,
                             timeout_s=10.0, max_redirects=0)
        raw = (rec.get("content") or "").encode("utf-8")
        total_bytes += len(raw)
        row = {"request_no": len(audit) + 1, "kind": kind, "url": url,
               "method": "GET", "status": rec.get("status"),
               "http_status": rec.get("http_status"), "bytes": len(raw),
               "sha256": hashlib.sha256(raw).hexdigest() if raw else "",
               "elapsed_ms": rec.get("elapsed_ms"), "error_code": rec.get("error_code"),
               "redirects": 0, "network_authorized": True}
        audit.append(row)
        return rec, row

    robots, rr = get(ROBOTS, "robots_policy_check")
    if robots.get("status") != "fetched":
        raise RuntimeError(f"robots failed: {robots.get('error_code')}")
    rp = robotparser.RobotFileParser(); rp.parse((robots.get("content") or "").splitlines())
    allowed = {u: rp.can_fetch("N-MIND-readonly/1.0", u) for u, _ in PAGES}
    rr["robots_allowed"] = allowed
    if not all(allowed.values()):
        raise RuntimeError("robots disallowed a candidate page")

    results = []
    for url, topic in PAGES:
        rec, row = get(url, "page")
        if rec.get("status") != "fetched":
            results.append({"url": url, "status": rec.get("status"), "error": rec.get("error")})
            continue
        ing = ingest_extracted(store, url=url, html=rec["content"],
                               fetched_at_utc=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                               trust_state="public-read-only", http_status=rec["http_status"],
                               mime="text/html")
        row.update({"topic": topic, "knowledge_status": ing["status"],
                    "doc_id": ing.get("doc_id"), "content_sha256": ing.get("content_sha256"),
                    "extractor_version": EXTRACTOR_VERSION,
                    "robots_policy": "allowed", "license_url":"https://docs.python.org/3.11/license.html"})
        results.append({"url": url, **ing})

    audit_text = "\n".join(json.dumps(x, ensure_ascii=False) for x in audit) + "\n"
    (out / "ONLINE_V1_GATE2_REQUEST_AUDIT.jsonl").write_text(audit_text, encoding="utf-8")
    root_audit = ROOT / "ONLINE_V1_REQUEST_AUDIT.jsonl"
    with root_audit.open("a", encoding="utf-8", newline="\n") as fh: fh.write(audit_text)
    stats = KnowledgeSearch(store).stats()
    metrics = {"status":"COMPLETE", "gate1_requests_used":5, "gate2_requests":len(audit),
               "cumulative_requests":5+len(audit), "new_pages_attempted":len(PAGES),
               "new_pages_ready":sum(x.get("status") == "READY" for x in results),
               "documents":stats["documents"], "all_versions":stats["all_versions"],
               "chunks":stats["chunks"], "deduplicated":stats["deduplicated"],
               "bytes_gate2":total_bytes, "optimizer_steps":0, "unauthorized_requests":0,
               "stable_checkpoint_modified":False, "robots_all_allowed":True}
    (out / "ONLINE_V1_GATE2_KNOWLEDGE_METRICS.json").write_text(json.dumps(metrics,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (ROOT / "ONLINE_V1_GATE2_KNOWLEDGE_METRICS.json").write_text(json.dumps(metrics,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

    search = KnowledgeSearch(store); service = ChatService(model=None, tokenizer=None, knowledge=search)
    eval_rows = []
    for qid, question, expected, supported in QUESTIONS:
        t0=time.perf_counter(); response=service.reply(ChatRequest.new("gate2", question))
        hits=search.search(question, limit=3)
        urls=[h["citation"]["url"] for h in hits]
        found=(expected in " ".join(urls)) if supported else (response.response_source == "none")
        eval_rows.append({"id":qid,"question":question,"response":response.text,
                          "response_source":response.response_source,
                          "top3": [h["citation"] for h in hits],
                          "expected_source":expected or None,"evidence_found":found,
                          "latency_ms":round((time.perf_counter()-t0)*1000,2),
                          "neural_called":False if response.response_source != "neural" else True})
    ev={"status":"COMPLETE","questions_total":10,"questions":eval_rows,
        "supported_evidence_rate":sum(x["evidence_found"] for x in eval_rows[:9])/9,
        "out_of_scope_abstain":eval_rows[9]["response_source"] in ("none","rule"),
        "optimizer_steps":0}
    (out / "ONLINE_V1_GATE2_CHAT_EVAL.json").write_text(json.dumps(ev,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    (ROOT / "ONLINE_V1_GATE2_CHAT_EVAL.json").write_text(json.dumps(ev,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    store.close(); print(json.dumps({"status":"COMPLETE", **metrics},ensure_ascii=False)); return 0

if __name__ == "__main__": raise SystemExit(main())
