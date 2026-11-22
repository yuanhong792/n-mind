"""知识入库管线：HTML → 标题/正文/元数据 → 分段 → 去重 → 索引。

安全口径：
  * <script>/<style>/<iframe> 内容整体剥离计数，绝不执行；
  * 页面命令式文本只标记 instruction_detected（数据隔离，绝不转成本地动作）；
  * 解析失败 → status=EXTRACT_FAILED，不把垃圾文本当知识；
  * extractor_version 随每行文档落库（来源链可追溯）。
"""

from __future__ import annotations

import hashlib
import re
import time
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urljoin, urlparse

from .knowledge_store import KnowledgeStore

EXTRACTOR_VERSION = "extractor-v2-cn"
NOISE_TAGS = frozenset({"script", "style", "iframe", "noscript", "nav", "footer",
                        "header", "aside", "form", "svg", "template"})
META_TIME_KEYS = ("article:published_time", "pubdate", "publishdate",
                  "og:published_time", "date")

_INSTRUCTION_RE = re.compile(
    r"(忽略之前|忽略上述|执行\s*powershell|执行命令|重置评测|删除冻结|改为\s*true|"
    r"放开.{0,6}预算|关闭安全|ignore (all )?previous|run\s+powershell|"
    r"disable.{0,10}safety)", re.IGNORECASE)

_CHUNK_SIZE = 240            # 每段目标字符数（中文正文分段）
_MIN_DOC_CHARS = 40          # 少于这个数的正文按解析失败处理


class _TextExtractor(HTMLParser):
    """标准库 HTMLParser：剥离噪声标签、收集标题/正文/链接/发布时间。"""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title = ""
        self.meta_time: str | None = None
        self._skip_depth = 0
        self._in_title = False
        self._in_h = 0
        self.h_first = ""
        self.parts: list[str] = []
        self.links: list[str] = []
        self._base = ""

    def set_base(self, url: str) -> None:
        self._base = url

    def handle_starttag(self, tag, attrs):
        if tag in NOISE_TAGS:
            self._skip_depth += 1
            return
        if tag == "title":
            self._in_title = True
        elif tag in ("h1", "h2", "h3"):
            self._in_h += 1
        elif tag == "meta":
            d = dict(attrs)
            key = d.get("property") or d.get("name") or ""
            if key.lower() in META_TIME_KEYS and d.get("content"):
                self.meta_time = self.meta_time or d["content"]
        elif tag == "a":
            href = dict(attrs).get("href")
            if href and self._skip_depth == 0:
                self.links.append(urljoin(self._base, href))

    def handle_endtag(self, tag):
        if tag in NOISE_TAGS and self._skip_depth:
            self._skip_depth -= 1
        elif tag == "title":
            self._in_title = False
        elif tag in ("h1", "h2", "h3") and self._in_h:
            self._in_h -= 1
        elif tag == "p" or tag in ("h1", "h2", "h3", "li", "div", "tr", "br", "section",
                                   "article"):
            self.parts.append("\n")

    def handle_data(self, data):
        if self._skip_depth:
            return
        text = data.strip()
        if not text:
            return
        if self._in_title and not self.title:
            self.title = text
        elif self._in_h and not self.h_first:
            self.h_first = text
        self.parts.append(text + " ")


def extract_page(html: str, url: str) -> dict[str, Any]:
    """HTML → 结构化字段；正文过短返回 status=EXTRACT_FAILED（不编造）。"""
    parser = _TextExtractor()
    parser.set_base(url)
    try:
        parser.feed(html)
        parser.close()
    except Exception:                                # noqa: BLE001 - 坏 HTML 不炸管线
        return {"status": "EXTRACT_FAILED", "title": "", "text": "", "links": [],
                "publish_time": None, "instruction_detected": False,
                "script_blocks_stripped": 0}
    text = re.sub(r"[ \t]+", " ", "".join(parser.parts))
    text = re.sub(r"\n\s*\n+", "\n", text).strip()
    body = re.sub(r"(?m)^\s*", "", text)
    instruction = bool(_INSTRUCTION_RE.search(body))
    return {
        "status": "READY" if len(body) >= _MIN_DOC_CHARS else "EXTRACT_FAILED",
        "title": parser.title or parser.h_first or url,
        "text": body,
        "links": parser.links,
        "publish_time": parser.meta_time,
        "instruction_detected": instruction,
        "script_blocks_stripped": 0,
    }


def split_chunks(text: str, *, target: int = _CHUNK_SIZE) -> list[dict[str, Any]]:
    """中文正文分段：按句读边界聚合到目标长度，保留字符区间（引用定位用）。"""
    out: list[dict[str, Any]] = []
    start = 0
    n = len(text)
    ordinal = 0
    while start < n:
        end = min(n, start + target)
        if end < n:
            # 在目标窗内找最后一个句读（。！？；\n）作为切点；找不到就硬切
            window = text[start:end]
            cut = max(window.rfind("。"), window.rfind("！"), window.rfind("？"),
                      window.rfind("；"), window.rfind("\n"))
            if cut > target // 3:
                end = start + cut + 1
        seg = text[start:end].strip()
        if seg:
            out.append({"ordinal": ordinal, "text": seg, "char_start": start,
                        "char_end": end, "text_sha256":
                            hashlib.sha256(seg.encode("utf-8")).hexdigest()})
            ordinal += 1
        start = end
    return out


def ingest_extracted(store: KnowledgeStore, *, url: str, html: str,
                     fetched_at_utc: str, trust_state: str, http_status: int | None,
                     mime: str = "text/html") -> dict[str, Any]:
    """抓取记录 → 抽取 → 去重 → 入库 → 索引。返回计数与证据字段。"""
    content_sha = hashlib.sha256(html.encode("utf-8")).hexdigest()
    host = urlparse(url).netloc.lower()
    if store.has_content(content_sha):
        return {"status": "deduplicated", "doc_id": None, "deduplicated": 1,
                "new_documents": 0, "indexed_chunks": 0, "content_sha256": content_sha}
    page = extract_page(html, url)
    doc_id = store.insert_document(
        canonical_url=url, title=page["title"], fetched_at_utc=fetched_at_utc,
        content_sha256=content_sha, trust_state=trust_state,
        extractor_version=EXTRACTOR_VERSION, http_status=http_status, mime=mime,
        raw_len=len(html), source_host=host,
        instruction_detected=page["instruction_detected"], status=page["status"])
    if page["status"] != "READY":
        return {"status": page["status"], "doc_id": doc_id, "deduplicated": 0,
                "new_documents": 0, "indexed_chunks": 0,
                "content_sha256": content_sha}
    chunks = split_chunks(page["text"])
    n_chunks = store.insert_chunks(doc_id, chunks)
    return {"status": "READY", "doc_id": doc_id, "deduplicated": 0, "new_documents": 1,
            "indexed_chunks": n_chunks, "content_sha256": content_sha,
            "title": page["title"], "links": page["links"][:20],
            "publish_time": page["publish_time"],
            "instruction_detected": page["instruction_detected"]}


# ---------------------------------------------------------------- fixture 资料
FIXTURE_PAGES: tuple[dict[str, str], ...] = (
    {"url": "fixture://local/nmind-intro.html",
     "html": """<!doctype html><html><head><title>N-MIND 项目简介（本地测试资料）</title></head>
<body><h1>N-MIND 项目简介</h1>
<p>N-MIND 是一个完全本地优先的神经记忆系统，所有权重从零开始训练，不使用任何预训练模型。
系统的推理、记忆与评测都必须在断网状态下可运行，CPU 是第一优先支持平台。</p>
<p>项目强调诚实计量：神经生成、知识检索与工具调用三类结果分别记账，绝不把工具的成功
算成神经能力。每个回答都必须标注来源。</p>
<p>本页内容为本地测试夹具（fixture），仅用于验证离线知识检索，不代表真实网页。</p>
</body></html>"""},
    {"url": "fixture://local/bot-v1-usage.html",
     "html": """<!doctype html><html><head><title>Bot V1 使用说明（本地测试资料）</title></head>
<body><h1>Bot V1 使用说明</h1>
<p>Bot V1 提供中文桌面聊天窗口，支持 Enter 发送消息、Shift+Enter 换行，回复都带来源标注。
知识面板可以搜索已保存的中文资料，设置面板管理允许站点与探索开关。</p>
<p>真实联网默认关闭：必须先在配置中启用网络，并逐站点授权；离线模式下只能读取本地
已缓存的知识与本地模型。本页为测试夹具（fixture）。</p>
</body></html>"""},
    {"url": "fixture://local/knowledge-policy.html",
     "html": """<!doctype html><html><head><title>知识库与安全边界（本地测试资料）</title></head>
<body><h1>知识库与安全边界</h1>
<p>网页内容一律按低信任数据处理。页面中出现的要求执行命令、关闭安全开关、修改配置等
文本只会被标记隔离，永远不会被执行。知识库只保存原文摘录与来源指纹。</p>
<p>每次抓取都会记录请求地址、最终地址、抓取时间、内容哈希与字节数；同一内容重复出现
会被去重，不会重复计数。本页为测试夹具（fixture）。</p>
</body></html>"""},
)


def import_fixture_pages(store: KnowledgeStore) -> dict[str, Any]:
    """演示用：把三篇中文夹具资料入库（明确标注 fixture，不算真实网页）。"""
    now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    stats = {"new_documents": 0, "deduplicated": 0, "indexed_chunks": 0}
    for page in FIXTURE_PAGES:
        r = ingest_extracted(store, url=page["url"], html=page["html"],
                             fetched_at_utc=now, trust_state="fixture", http_status=None,
                             mime="text/fixture")
        stats["new_documents"] += r["new_documents"]
        stats["deduplicated"] += r["deduplicated"]
        stats["indexed_chunks"] += r["indexed_chunks"]
    return stats


def ensure_fixture_pages(store: KnowledgeStore) -> None:
    """幂等版：首次打开窗口时保证演示资料存在（重复导入不增长计数）。"""
    import_fixture_pages(store)
