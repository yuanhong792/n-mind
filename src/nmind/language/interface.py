"""语言接口：把结构化事件与决策说成中文，并把中文规则解析回结构化字段。

**这一层不做通用自然语言理解**。它做的是有边界的模板映射：
  * 事件 → 中文句子（受控词表，输入维度已知）；
  * 中文规则句 → 结构化字段（只识别固定句式，识别不了就返回 unknown 并说明）；
  * 决策 → 中文说明（附带置信度"未经校准"的显式提醒）。

需要真正的通用语言能力，请接 M4 的本地预训练模型（language/local_adapter.py），
并在报告里写明"通用语言能力来自外部预训练模型，不是 N-Core 学到的"。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

DIR_WORDS = {"上": "up", "右": "right", "下": "down", "左": "left",
             "up": "up", "right": "right", "down": "down", "left": "left",
             "北": "up", "东": "right", "南": "down", "西": "left"}
ACTION_NAMES = ("up", "right", "down", "left")
ACTION_CN = ("上", "右", "下", "左")


@dataclass
class ParsedRule:
    """中文规则句的解析结果。matched=False 时绝不猜意思。"""

    text: str
    matched: bool = False
    kind: str = ""
    direction: str | None = None
    target: tuple[int, int] | None = None
    subject: str | None = None
    unknown_spans: list[str] = field(default_factory=list)

    def as_claims(self) -> list[list[str]]:
        """转成记忆层能存的主张三元组（巩固阶段要有≥2 条证据才会升格为事实）。"""
        if not self.matched:
            return []
        if self.kind == "rule" and self.subject and self.direction:
            return [[self.subject, "moves", self.direction]]
        if self.kind == "goal" and self.subject and self.target:
            return [[self.subject, "goal_at", f"{self.target[0]}:{self.target[1]}"]]
        return []


def describe_event(numeric: Any, *, size: int = 6) -> str:
    """把网格观测向量翻成一句中文（顺序遵循 datasets._encode 的 26 维布局）。"""
    v = [float(x) for x in (numeric if hasattr(numeric, "__iter__") else [numeric])]
    if len(v) < 26:
        return f"观测维度不足（{len(v)}<26），无法可靠描述"

    def argmax(a: int, b: int) -> int:
        seg = v[a:b]
        m = max(seg)
        return -1 if m < 0.5 else int(seg.index(m))

    x, y, gx, gy = argmax(0, 6), argmax(6, 12), argmax(12, 18), argmax(18, 24)
    occluded, collided = v[24] > 0.5, v[25] > 0.5
    parts = [f"物体在 ({x},{y})" if x >= 0 and y >= 0 else "物体位置未知"]
    parts.append("目标被遮挡" if occluded or gx < 0 else f"目标在 ({gx},{gy})")
    if collided:
        parts.append("本步发生碰撞（未移动）")
    return "，".join(parts) + "。"


def parse_rule(text: str) -> ParsedRule:
    """受控句式解析。支持：
        "位于 x y 的物体向<方向>移动"
        "目标在 (x, y)" / "目标在 x y"
        "物体朝左走"
    其余返回 matched=False，并把没认出来的片段列出来（诚实比假装懂重要）。
    """
    raw = (text or "").strip()
    if not raw:
        return ParsedRule(text=raw, unknown_spans=[])

    m = re.search(r"(?:位于|在)\s*\(?\s*(\d+)\s*[,\s]\s*(\d+)\s*\)?[^。;；]*?"
                  r"(?:向|朝|往|->|→)\s*([上下左右updownrightleft北南东西]{1,6})", raw, re.I)
    if m:
        x, y = int(m.group(1)), int(m.group(2))
        d = DIR_WORDS.get(m.group(3).lower()) or DIR_WORDS.get(m.group(3))
        if d:
            return ParsedRule(text=raw, matched=True, kind="rule", subject=f"cell:{x}:{y}",
                              direction=d, unknown_spans=_leftovers(raw, m.group(0)))

    m = re.search(r"(?:物体|它)?[^。;；]{0,8}(?:向|朝|往)\s*([上下左右北南东西]{1,2})\s*(?:移动|走|前进)", raw)
    if m:
        d = DIR_WORDS.get(m.group(1))
        if d:
            return ParsedRule(text=raw, matched=True, kind="rule", subject="object", direction=d,
                              unknown_spans=_leftovers(raw, m.group(0)))

    m = re.search(r"目标(?:位置)?[^0-9]{0,6}\(?\s*(\d+)\s*[,\s]\s*(\d+)\s*\)?", raw)
    if m:
        return ParsedRule(text=raw, matched=True, kind="goal", subject="object",
                          target=(int(m.group(1)), int(m.group(2))), unknown_spans=_leftovers(raw, m.group(0)))

    nums = re.findall(r"\d+", raw)
    return ParsedRule(text=raw, matched=False, unknown_spans=[raw] if not nums else [s for s in re.split(r"\d+", raw) if s.strip()])


def _leftovers(text: str, matched_span: str) -> list[str]:
    rest = text.replace(matched_span, "").strip(" 。;；，,")
    return [rest] if rest else []


def verbalize_decision(decision: dict[str, Any], *, event_text: str = "",
                       retrieved: int = 0, cloud_used: bool = False,
                       degraded: bool = False) -> str:
    """把决策翻成人话。置信度是未校准的，所以措辞用"倾向于"而不是"有 X% 把握"。"""
    idx = int(decision.get("action_index", decision.get("action", 0)))
    name = ACTION_CN[idx] if 0 <= idx < 4 else str(decision.get("action", "?"))
    conf = float(decision.get("confidence", 0.0))
    margin = float(decision.get("margin", 0.0))
    bits = [f"倾向于向{name}移动"]
    if margin < 0.10:
        bits.append("（各方向得分接近，本地不确定度高）")
    elif conf > 0.85:
        bits.append("（本地判断较一致）")
    if retrieved:
        bits.append(f"参考了 {retrieved} 条本地记忆")
    if cloud_used:
        bits.append("含云端补充信息（未当作真值）")
    if degraded:
        bits.append("已按预算降级计算")
    if event_text:
        return f"{event_text} → " + "，".join(bits) + "。"
    return "，".join(bits) + "。"


def capability_note() -> str:
    return ("当前语言模块是受控模板映射，只处理本任务词表；"
            "不具备通用自然语言能力。接入本地预训练模型后，通用语言能力来自该外部模型。")
