"""HTML/结构化提取器：网页 → 纯数据与结构化记录。

安全口径（M4 任务书第 3 节）：
  * <script>/<style> 内容整体剥离并计数，绝不执行；
  * 页面中的命令式文本（"执行 powershell"、"重置评测门槛"等）只被**标记**为
    instruction_attempt，进入 rejected 经验流，绝不派生任何本地动作；
  * 本模块是工程脚手架（规则表解析器），如实标注——不能把工具解析结果冒充
    模型自己学会了阅读网页。
"""

from __future__ import annotations

import re
from typing import Any

SCRIPT_STYLE_RE = re.compile(r"<(script|style)\b[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
ROW_RE = re.compile(r"<tr>\s*<th>(.*?)</th>\s*<td>(.*?)</td>\s*</tr>", re.IGNORECASE | re.DOTALL)
TAG_RE = re.compile(r"<[^>]+>")
# 命令式指令特征：只用于"标记数据不可信"，不用于任何执行决策
INSTRUCTION_RE = re.compile(
    r"(忽略之前|执行\s*powershell|执行命令|重置评测|删除冻结|改为\s*true|放开.*预算|"
    r"ignore (all )?previous|run\s+powershell|reset .*threshold)", re.IGNORECASE)

EXTRACTOR_VERSION = "extractor-v1"


def strip_scripts(html: str) -> tuple[str, int]:
    """剥离 <script>/<style>，返回 (清理后文本, 剥离块数）。内容只丢弃，不执行。"""
    blocks = len(SCRIPT_STYLE_RE.findall(html))
    return SCRIPT_STYLE_RE.sub("", html), blocks


def extract_rule_table(html: str) -> dict[str, Any]:
    """解析 <tr><th>key</th><td>value</td></tr> 规则表为 dict；非表格文本忽略。"""
    cleaned, stripped = strip_scripts(html)
    pairs = {TAG_RE.sub("", k).strip(): TAG_RE.sub("", v).strip()
             for k, v in ROW_RE.findall(cleaned)}
    return {"fields": pairs, "script_blocks_stripped": stripped}


def detect_instructions(html: str) -> dict[str, Any]:
    """标记页面中的命令式文本（数据隔离用途）。返回命中片段，不做任何执行。"""
    cleaned, stripped = strip_scripts(html)
    hits = [m.group(0) for m in INSTRUCTION_RE.finditer(cleaned)]
    return {"instruction_detected": bool(hits), "snippets": hits[:5],
            "script_blocks_stripped": stripped}


def extract_text(html: str, limit: int = 500) -> str:
    cleaned, _ = strip_scripts(html)
    return TAG_RE.sub(" ", re.sub(r"\s+", " ", cleaned)).strip()[:limit]
