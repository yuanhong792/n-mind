"""白名单确定性工具（M5-Language 任务书 P3）：唯一可被技能/资源管理器调用的执行实现。

* 网页提供的任何代码一律不注册为可执行程序——本文件实现是仓库内置、可审计的；
* calculator 用 ast 白名单（数字/四则/括号/一元正负/幂/模/整除），拒绝其它一切节点；
* 所有函数确定性、无网络、无副作用。
"""

from __future__ import annotations

import ast
import json

_ALLOWED_NODES = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant, ast.Load,
                   ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod,
                   ast.Pow, ast.USub, ast.UAdd)


def calculator(expression: str) -> float:
    """安全算术求值：ast 白名单，只允许数字与四则运算。"""
    node = ast.parse(expression.strip(), mode="eval")
    for n in ast.walk(node):
        if not isinstance(n, _ALLOWED_NODES):
            raise ValueError(f"表达式包含不允许的节点：{type(n).__name__}")
        if isinstance(n, ast.Constant) and not isinstance(n.value, (int, float)):
            raise ValueError("只允许数字常量")
    return float(eval(compile(node, "<nmind-calculator>", "eval")))  # noqa: S307 白名单已限制


def json_extractor(payload: str, field_path: str) -> object:
    """从 JSON 字符串按点路径取字段（列表用数字下标）。"""
    data = json.loads(payload)
    cur: object = data
    for part in field_path.split("."):
        if isinstance(cur, list):
            cur = cur[int(part)]
        elif isinstance(cur, dict):
            cur = cur[part]
        else:
            raise ValueError(f"路径 {field_path!r} 在标量上继续取字段")
    return cur


_UNIT_TABLE: dict[tuple[str, str], float] = {
    ("km", "m"): 1000.0, ("m", "km"): 0.001, ("m", "cm"): 100.0, ("cm", "m"): 0.01,
    ("kg", "g"): 1000.0, ("g", "kg"): 0.001, ("l", "ml"): 1000.0, ("ml", "l"): 0.001,
}


def unit_converter(value: float, unit_from: str, unit_to: str) -> float:
    key = (unit_from.lower(), unit_to.lower())
    if key == ("c", "f"):
        return value * 9 / 5 + 32
    if key == ("f", "c"):
        return (value - 32) * 5 / 9
    if key not in _UNIT_TABLE:
        raise ValueError(f"不支持的单位对 {key}")
    return value * _UNIT_TABLE[key]


def api_param_builder(schema: dict[str, str], values: dict[str, object]) -> dict[str, object]:
    """按 schema 校验/过滤参数；多余键拒绝（授权范围约束）。"""
    extra = sorted(set(values) - set(schema))
    if extra:
        raise ValueError(f"未授权参数：{extra}")
    out: dict[str, object] = {}
    for k, typ in schema.items():
        if k not in values:
            raise ValueError(f"缺少参数 {k}")
        if typ == "number":
            out[k] = float(values[k])            # type: ignore[arg-type]
        elif typ == "string":
            out[k] = str(values[k])
        else:
            raise ValueError(f"未知类型 {typ!r}")
    return out


BUILTIN_TOOLS = {
    "calculator": calculator,
    "json_extractor": json_extractor,
    "unit_converter": unit_converter,
    "api_param_builder": api_param_builder,
}
