"""生成 continuity_bug_regression.json：P0 参数继承回归 + 联网安全回归的实际运行证据。"""

from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def main() -> int:
    os.chdir(ROOT)
    xml = HERE / "_continuity_junit.xml"
    code = pytest.main([
        "tests/test_true_continual.py", "tests/test_online_learning.py",
        "-v", "--tb=short", "-p", "no:warnings", f"--junitxml={xml}",
    ])
    tree = ET.parse(xml)
    cases = []
    for tc in tree.iter("testcase"):
        failed = tc.find("failure") is not None or tc.find("error") is not None
        cases.append({"test": f"{tc.get('classname')}::{tc.get('name')}",
                      "outcome": "failed" if failed else "passed"})
    p0 = [c for c in cases if "test_true_continual" in c["test"]]
    online = [c for c in cases if "test_online_learning" in c["test"]]
    payload = {
        "generated_at": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
        "command": "pytest tests/test_true_continual.py tests/test_online_learning.py -v",
        "exit_code": int(code),
        "total": len(cases),
        "passed": sum(1 for c in cases if c["outcome"] == "passed"),
        "failed": sum(1 for c in cases if c["outcome"] == "failed"),
        "p0_continuity_regression": {
            "conclusion": "PIPELINE_FIXED" if code == 0 else "REGRESSION",
            "assertions": ["B 阶段起点 = A 结束权重逐位一致（且 ≠ 随机初始化）",
                            "显式 model= 优先，语义冲突报错不静默重建",
                            "增量更新只动允许参数，冻结参数逐位不变",
                            "lr=0 / 全冻结零学习收益（逐位不变）",
                            "各臂同一父快照（哈希一致）",
                            "reversed_control 存在有效冲突样本",
                            "子 checkpoint 保存→重载分数一致 + 预算中止不触 stable",
                            "fresh_build 路径行为不变"],
            "cases": p0},
        "online_safety_regression": {"cases": online},
        "cases": cases,
    }
    (HERE / "continuity_bug_regression.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    xml.unlink(missing_ok=True)
    print(json.dumps({"exit_code": code, "total": payload["total"],
                      "passed": payload["passed"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
