"""本地确定性有限候选搜索器：不依赖外部LLM，不做无限搜索。"""
from __future__ import annotations

import json
from pathlib import Path

from candidate_registry import CANDIDATES, build_model

HERE = Path(__file__).resolve().parent


def math_verifier_ok() -> bool:
    p = HERE / "M1_MATH_TESTS.json"
    if not p.exists():
        return False
    report = json.loads(p.read_text(encoding="utf-8"))
    return report.get("status") == "M1_MATH_TESTS_PASS"


def enumerate_candidates():
    """预登记四项；顺序固定，不使用随机或外部生成。"""
    out = []
    for name, spec in CANDIDATES.items():
        item = {"name": name, **spec, "math_verifier_pass": math_verifier_ok()}
        out.append(item)
    return out


def filter_by_verifier(candidates):
    return [c for c in candidates if c["math_verifier_pass"]]


def manifest():
    candidates = enumerate_candidates()
    return {"status": "ARCHITECTURE_SEARCH_MANIFEST",
            "method": "deterministic_finite_enumeration",
            "external_llm": False,
            "math_verifier_pass": math_verifier_ok(),
            "candidates": candidates,
            "eliminated": [c["name"] for c in candidates if not c["math_verifier_pass"]]}


if __name__ == "__main__":
    print(json.dumps(manifest(), ensure_ascii=False, indent=2))
