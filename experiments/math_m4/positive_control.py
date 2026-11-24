"""M41：可学习性正控制数据/门禁定义（无训练入口）。

M4.1 修复：
- 本文件不再创建 optimizer，不再调用 optimizer.step；
- 唯一正式训练入口是 train_m4.py，所有 step 必须先经 BudgetManager reserve；
- 正控制固定使用 64 条样本，并在同一 64 条样本上评测拟合能力；
- 正控制权重不进入主实验；成绩不等于未见数据泛化。

未授权时只检查数据/监督对齐，标 NOT_RUN_TRAINING。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(HERE))
from evaluate_m4 import make_batch  # noqa: E402
from tokenizer_m4 import M4Tokenizer, SEP  # noqa: E402

N_CONTROL = 64
TF_MIN = 0.95
NATURAL_EOS_FULL_MIN = 0.90


def control_rows(tok, n=N_CONTROL):
    """覆盖双字符复制、多字符复制、双变量绑定的小型固定正控制集。"""
    rows = []
    n_copy2 = max(1, n // 2)
    n_copyn = max(1, n // 4)
    n_bind = max(1, n - n_copy2 - n_copyn)

    for i in range(n_copy2):
        a = chr(ord("A") + (i % 20))
        b = chr(ord("0") + (i % 10))
        s = a + b
        rows.append({"task": "A2_copy2", "prompt": f"COPY {s}", "answer": s})

    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    for i in range(n_copyn):
        s = "".join(alphabet[(i * 3 + j) % len(alphabet)] for j in range(3 + (i % 2)))
        rows.append({"task": "A3_copyN", "prompt": f"COPY {s}", "answer": s})

    for i in range(n_bind):
        target = i % 2
        v1, v2 = "a", "b"
        x, y = str((i % 9) + 1), str(((i // 9) % 9) + 1)
        if target == 0:
            prompt = f"{v1}={x}\n{v2}={y}\nGET {v1}"
            answer = x
        else:
            prompt = f"{v1}={x}\n{v2}={y}\nGET {v2}"
            answer = y
        rows.append({"task": "B2_bind", "prompt": prompt, "answer": answer})
    return rows[:n]


def alignment_report(tok, rows):
    """检查 answer-only 监督是否从 SEP 之后开始，并核对每个样本对齐。"""
    x, valid, w, tgt = make_batch(tok, rows, "A")
    failures = []
    for i, r in enumerate(rows):
        seq, _ = tok.encode_pair(r["prompt"], r["answer"])
        sep = seq.index(SEP)
        if w[i, :sep + 1].sum() != 0 or w[i, sep + 1:].sum() <= 0:
            failures.append({"i": i, "prompt": r["prompt"], "sep": sep})
    return {"n_samples": len(rows), "answer_only_alignment": not failures,
            "failures": failures, "batch": list(x.shape),
            "supervised_tokens": int(w.sum())}


def gate_spec():
    return {"tf_accuracy_min": TF_MIN,
            "natural_eos_full_accuracy_min": NATURAL_EOS_FULL_MIN,
            "gradient_finite_and_nonzero_required": True,
            "parameter_update_required": True,
            "future_leakage_forbidden": True,
            "evaluate_on_same_64_rows": True,
            "weights_enter_main_experiment": False,
            "abort_main_on_reference_failure": True,
            "sole_training_entry": "train_m4.py"}


def evaluate_gate(tf_accuracy, natural_eos_full_accuracy, gradient_ok,
                  parameter_updated, causality_pass):
    spec = gate_spec()
    checks = {
        "tf_accuracy": (float(tf_accuracy) >= spec["tf_accuracy_min"],
                        float(tf_accuracy), spec["tf_accuracy_min"]),
        "natural_eos_full_accuracy": (
            float(natural_eos_full_accuracy) >= spec["natural_eos_full_accuracy_min"],
            float(natural_eos_full_accuracy), spec["natural_eos_full_accuracy_min"]),
        "gradient_finite_and_nonzero": (bool(gradient_ok), gradient_ok, True),
        "parameter_updated": (bool(parameter_updated), parameter_updated, True),
        "no_future_leakage": (bool(causality_pass), causality_pass, True),
    }
    return {"pass": all(v[0] for v in checks.values()),
            "checks": {k: {"pass": v[0], "observed": v[1], "required": v[2]}
                       for k, v in checks.items()}}


def main():
    torch.set_num_threads(20)
    tok = M4Tokenizer()
    rows = control_rows(tok)
    out = {"status": "NOT_RUN_TRAINING", "reason": "唯一正式训练入口是 train_m4.py，需用户明确授权",
           "n_samples": len(rows), "alignment": alignment_report(tok, rows),
           "gate": gate_spec(), "note": "本文件不训练、不创建 optimizer、不调用 step。"}
    p = HERE / "M4_POSITIVE_CONTROL.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[positive] status={out['status']} aligned={out['alignment']['answer_only_alignment']} "
          f"写出 {p}")


if __name__ == "__main__":
    main()
