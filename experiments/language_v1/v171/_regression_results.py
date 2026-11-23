"""V171 P6: collect real pytest results into JSON (never fabricate)."""
from __future__ import annotations
import json, re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
V171 = ROOT / "experiments/language_v1/v171"

def parse(logname):
    p = V171 / logname
    if not p.exists():
        return {"log": logname, "status": "NOT_RUN"}
    text = p.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"(\d+) passed(?:, (\d+) failed)?(?:, (\d+) errors?)?", text)
    fails = re.findall(r"FAILED (\S+)", text)
    errs = re.findall(r"ERROR (\S+)", text)
    if m:
        passed = int(m.group(1)); failed = int(m.group(2) or 0); errors = int(m.group(3) or 0)
    elif "EXIT=0" in text or "[100%]" in text:
        # pytest -q dot format: count dots/skips before warnings summary; no FAILED/ERROR = all pass
        head = text.split("warnings summary")[0].split("-- Docs:")[0]
        passed = sum(1 for c in head if c == ".")
        failed = len(fails); errors = len(errs)
    else:
        passed = failed = errors = None
    return {"log": logname, "passed": passed, "failed": failed, "errors": errors,
            "failed_tests": fails, "error_tests": errs,
            "exit_marker": "EXIT=0" in text or "passed" in text,
            "tail": "\n".join(text.strip().splitlines()[-5:])}


def main():
    report = {
        "purpose": "V171 P6 real pytest results; no test marked passed unless actually executed",
        "targeted_suite": parse("_pytest_run1.log"),
        "full_suite": parse("_pytest_full.log"),
        "coverage": {
            "checkpoint_save_load": "tests/test_v171_checkpoint.py",
            "best_final_distinction": "tests/test_v171_checkpoint.py",
            "router_gradient": "tests/test_router_language_grad.py, tests/test_router_ste.py, tests/test_router.py",
            "bpe_vocab_compat": "tests/test_bpe_utf8_decoder.py, tests/test_language_v13_vocab.py",
            "answer_mask": "tests/test_v171_protocol.py::test_answer_mask_includes_eos",
            "full_answer_scoring": "tests/test_v171_protocol.py",
            "free_generation_eos": "tests/test_v171_protocol.py",
            "utf8": "tests/test_bpe_utf8_decoder.py",
            "fact_level_dedup": "tests/test_v171_data.py",
            "counterfactual_pairing": "tests/test_v171_data.py",
        },
    }
    (V171 / "V171_REGRESSION_RESULTS.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
