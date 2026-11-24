"""M3-P6：受限数学表达式语法（禁止任意 Python 代码执行）。

允许的原子算子（对状态张量 h 与输入 x 的变换）：
- identity(h), diag(h; a), lowrank(h; U,V,r), nonlinear(tanh/sigmoid/relu/gelu)
- add(a,b), mul(a,b)（逐元素/广播，张量形状由 verifier 检查）
- gate(h, cand; g)（h + g*(cand-h)）
- matmul(A, h)（固定维度矩阵）
- residual(h, expr)（h + expr）
- input_proj(x; W)（x 投影到状态维度）
- norm(h)（L2 归一化）

禁止：eval/exec/任意函数/字符串构造答案/索引到词表常量/读取数据集标签。
表达式以树结构表示，节点为 dict：{"op":..., "args":[...], "params":{...}}。
"""
from __future__ import annotations

ALLOWED_OPS = {
    "identity", "diag", "lowrank", "tanh", "sigmoid", "relu", "gelu",
    "add", "mul", "gate", "matmul", "residual", "input_proj", "norm",
}
FORBIDDEN_OPS = {"eval", "exec", "open", "getattr", "setattr", "import",
                 "answer", "label", "constant_lookup", "print", "compile"}


def validate_tree(node: dict, allowed_ops=ALLOWED_OPS) -> tuple[bool, str]:
    """递归校验表达式树只含允许算子，参数仅为数值/字符串形状标记。"""
    if not isinstance(node, dict) or "op" not in node:
        return False, "node must be dict with 'op'"
    op = node["op"]
    if op not in allowed_ops:
        return False, f"forbidden op: {op}"
    for arg in node.get("args", []):
        if isinstance(arg, dict):
            ok, err = validate_tree(arg, allowed_ops)
            if not ok:
                return False, err
        elif not (isinstance(arg, (int, float, str)) or arg is None):
            return False, f"invalid arg type: {type(arg)}"
    for k, v in node.get("params", {}).items():
        if not isinstance(v, (int, float, str, list)):
            return False, f"invalid param {k}: {type(v)}"
    return True, "ok"


def tree_to_str(node: dict) -> str:
    op = node["op"]
    args = node.get("args", [])
    parts = []
    for a in args:
        parts.append(tree_to_str(a) if isinstance(a, dict) else str(a))
    params = node.get("params", {})
    ps = ",".join(f"{k}={v}" for k, v in sorted(params.items()))
    return f"{op}({','.join(parts)}{';' + ps if ps else ''})"
