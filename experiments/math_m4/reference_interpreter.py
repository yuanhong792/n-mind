"""M41F 独立参考解释器：仅用于数据生成、数据审计与评测核验。

禁止在模型 forward / 训练推理路径中导入或调用本模块。
语义：
- 普通赋值 `v=5` 写入当前值；
- 变量引用赋值 `b=a` 是值复制，复制 a 在该时刻的值；
- `GET v` 返回 v 的当前值；
- `GET v~OLD` 返回 v 的历史首个值；`GET v~NEW` 返回当前值。
"""

from __future__ import annotations

import re

_ASSIGN = re.compile(r"^([a-z])=([0-9]|[a-z])$")


def parse_prompt(prompt: str):
    assigns, query = [], None
    for raw in prompt.strip().splitlines():
        s = raw.strip()
        m = _ASSIGN.fullmatch(s)
        if m:
            assigns.append((m.group(1), m.group(2)))
        elif s.startswith("GET "):
            query = s[4:].strip()
        # 非赋值/非查询行（如 OUT/COPY/REVERSE）由任务级参考逻辑处理，
        # canonical 只保留 TASK|prompt 标记。
    return assigns, query


def canonical(prompt: str) -> str:
    assigns, query = parse_prompt(prompt)
    if query is None:
        return "TASK|" + prompt
    parts = [f"{k}={v if v.isdigit() else '@' + v}" for k, v in assigns]
    return ";".join(parts) + f"|GET:{query}"


def reference_answer(task: str, prompt: str) -> str:
    """独立计算参考答案；任何标签 mismatch 都应视为数据生成 bug。"""
    s = prompt.strip()
    if task == "A1_copy1" and s.startswith("OUT "):
        return s[4:]
    if task in ("A2_copy2", "A3_copyN") and s.startswith("COPY "):
        return s[5:]
    if task == "A4_reverse" and s.startswith("REVERSE "):
        return s[8:][::-1]
    if task == "A5_repeat":
        m = re.fullmatch(r"REPEAT (\S+) ([0-9]+)", s)
        if m:
            return m.group(1) * int(m.group(2))
    if task == "A6_index":
        m = re.fullmatch(r"INDEX (\S+) ([0-9]+)", s)
        if m:
            text, i = m.group(1), int(m.group(2))
            if 1 <= i <= len(text):
                return text[i - 1]
    if task == "A3_copyN" and s.startswith("ECHO "):
        return s[5:]

    assigns, query = parse_prompt(prompt)
    if query is None:
        raise ValueError(f"{task} 缺少 GET 查询：{prompt!r}")
    current, history = {}, {}
    for var, rhs in assigns:
        if rhs.isdigit():
            value = rhs
        elif rhs in current:
            value = current[rhs]  # 值复制，不建立后续引用依赖
        else:
            raise ValueError(f"变量引用尚未赋值：{var}={rhs}，prompt={prompt!r}")
        current[var] = value
        history.setdefault(var, []).append(value)

    if "~" in query:
        var, mode = query.split("~", 1)
    else:
        var, mode = query, "NEW"
    if mode == "NEW":
        return current[var]
    if mode == "OLD":
        return history[var][0]
    raise ValueError(f"未知查询模式：{mode}")
