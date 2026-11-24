"""N-MIND MiniLang：白名单、受控、可验证的微型编程语言解释器。

严格安全：
- 只解析预定义指令，绝不 eval/exec/系统命令/文件/网络/环境变量。
- 限制指令数、变量数、整数范围、循环次数、执行时间。
- 无无限循环；无无限制内存分配。
与神经推理严格隔离：解释器只在数据生成与验证阶段提供标准答案/反馈。
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

MAX_STEPS = 256
MAX_VARS = 32
INT_MIN, INT_MAX = -10**6, 10**6
MAX_LOOP = 16


class MiniLangError(ValueError):
    pass


@dataclass
class ExecResult:
    output: str = ""
    variables: dict = field(default_factory=dict)
    steps: int = 0
    error: str | None = None
    trace: list = field(default_factory=list)


_TOKEN = re.compile(r"[A-Za-z]+|-?\d+|\S")


class MiniLang:
    """仅支持 SET/GET/ADD/SUB/MUL/DIV/PRINT/CMP/IF 的确定性解释器。"""

    def __init__(self, max_steps=MAX_STEPS, max_vars=MAX_VARS):
        self.max_steps = max_steps
        self.max_vars = max_vars

    def validate(self, program: str) -> bool:
        """语法预检：任何未白名单指令/非法结构直接拒绝，不进入执行。"""
        try:
            self._parse(program)
            return True
        except MiniLangError:
            return False

    def _parse(self, program: str):
        lines = []
        for raw in program.splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            toks = _TOKEN.findall(line)
            if not toks:
                continue
            op = toks[0].upper()
            if op not in {"SET", "GET", "ADD", "SUB", "MUL", "DIV", "PRINT", "CMP", "IF"}:
                raise MiniLangError(f"未知指令: {op}")
            lines.append((op, toks[1:]))
        return lines

    def _value(self, t: str, env: dict) -> int:
        if re.fullmatch(r"-?\d+", t):
            v = int(t)
        elif t in env:
            v = env[t]
        else:
            raise MiniLangError(f"未定义变量: {t}")
        if not (INT_MIN <= v <= INT_MAX):
            raise MiniLangError("整数越界")
        return v

    def run(self, program: str) -> ExecResult:
        t0 = time.perf_counter()
        res = ExecResult()
        env: dict[str, int] = {}
        try:
            lines = self._parse(program)
        except MiniLangError as e:
            res.error = str(e)
            return res
        executed = 0
        pc = 0
        while pc < len(lines):
            if executed >= self.max_steps or time.perf_counter() - t0 > 1.0:
                res.error = "执行预算超限"
                break
            op, args = lines[pc]
            try:
                if op == "SET":
                    if len(args) != 2 or not re.fullmatch(r"[a-zA-Z]\w*", args[0]):
                        raise MiniLangError("SET 需要 <变量> <值>")
                    if args[0] not in env and len(env) >= self.max_vars:
                        raise MiniLangError("变量数超限")
                    env[args[0]] = self._value(args[1], env)
                elif op == "GET":
                    if len(args) != 1:
                        raise MiniLangError("GET 需要 1 个变量")
                    res.output = str(self._value(args[0], env))
                elif op in {"ADD", "SUB", "MUL", "DIV"}:
                    if len(args) not in (2, 3):
                        raise MiniLangError(f"{op} 需要 2 或 3 个参数")
                    a, b = self._value(args[0], env), self._value(args[1], env)
                    if op == "ADD": v = a + b
                    elif op == "SUB": v = a - b
                    elif op == "MUL": v = a * b
                    else:
                        if b == 0:
                            raise MiniLangError("除零")
                        v = a // b
                    if not (INT_MIN <= v <= INT_MAX):
                        raise MiniLangError("整数越界")
                    if len(args) == 3:
                        env[args[2]] = v
                    else:
                        res.output = str(v)
                elif op == "PRINT":
                    if len(args) != 1:
                        raise MiniLangError("PRINT 需要 1 个参数")
                    res.output = str(self._value(args[0], env))
                elif op == "CMP":
                    if len(args) != 3 or args[1] not in {">", "<", "=", ">=", "<="}:
                        raise MiniLangError("CMP <a> <op> <b>")
                    a, b = self._value(args[0], env), self._value(args[2], env)
                    flag = {">": a > b, "<": a < b, "=": a == b,
                            ">=": a >= b, "<=": a <= b}[args[1]]
                    env[args[0] + "_cmp"] = int(flag)
                elif op == "IF":
                    # IF <var>_cmp <value> : 简化条件跳转由道具位控制
                    if len(args) != 3:
                        raise MiniLangError("IF <flag> <then_value> <else_value>")
                    flag = self._value(args[0], env)
                    res.output = str(self._value(args[1], env)) if flag else str(self._value(args[2], env))
            except MiniLangError as e:
                res.error = str(e)
                break
            res.trace.append((op, args, dict(env)))
            executed += 1
            pc += 1
        res.steps = executed
        res.variables = env
        return res
