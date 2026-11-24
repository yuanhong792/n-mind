"""P11：可学习性正控制实验（独立入口）。

32/64 条无歧义双字符复制 + 双变量绑定，验证训练器/监督/优化器是否工作。
检查：能否拟合小训练集、TF 准确率、首/次 Token、EOS、自由生成、梯度与参数更新。
正控制所有 optimizer.step 计入 M4 预算；正控制权重不得用于主实验（主实验重新随机初始化）。
正控制成绩不能当作未见数据泛化成绩。

未授权时只做零步检查（数据/监督对齐），标 NOT_RUN_TRAINING。
运行：python positive_control.py [--authorize]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(HERE))
from evaluate_m4 import make_batch, grouped_loss, eval_rows, tf_position_accuracy  # noqa: E402
from tokenizer_m4 import M4Tokenizer, SEP  # noqa: E402
from models.gru_reference import GRUReferenceM3  # noqa: E402
from models.transformer_reference import TinyTransformerM3  # noqa: E402

MODELS = {"R1": GRUReferenceM3, "R2": TinyTransformerM3}
CFG = {"R1": {}, "R2": {"d_model": 64, "nhead": 4, "layers": 2}}


def control_rows(tok, n=64):
    """覆盖双字符复制、多字符复制、双变量绑定的小型正控制集。"""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--authorize", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(20)
    tok = M4Tokenizer()
    rows = control_rows(tok)
    x, valid, w, tgt = make_batch(tok, rows, "A")
    alignment = True
    for i, r in enumerate(rows):
        seq, _ = tok.encode_pair(r["prompt"], r["answer"])
        sep = seq.index(SEP)
        if w[i, :sep + 1].sum() != 0 or w[i, sep + 1:].sum() <= 0:
            alignment = False
            break
    out = {"status": "NOT_RUN_TRAINING" if not args.authorize else "COMPLETE",
           "reason": "需明确授权" if not args.authorize else None,
           "n_samples": len(rows), "answer_only_alignment": alignment,
           "note": "正控制权重不进入主实验；成绩不等于泛化", "models": {}}
    if args.authorize:
        for name, cls in MODELS.items():
            torch.manual_seed(1)
            m = cls(tok.vocab_size, **CFG[name])
            opt = torch.optim.AdamW(m.parameters(), lr=0.003)
            losses = []
            for _ in range(128):
                m.train()
                logits, _ = m(x[:, :-1], valid_mask=valid[:, :-1])
                loss = grouped_loss(logits, tgt[:, 1:], w[:, 1:])
                opt.zero_grad(); loss.backward(); opt.step()
                losses.append(float(loss.item()))
            tf = tf_position_accuracy(m, tok, rows)
            ev = eval_rows(m, tok, rows)
            out["models"][name] = {"train_loss_first": round(losses[0], 4),
                                   "train_loss_last": round(losses[-1], 4),
                                   "teacher_forced": tf, "free_generation": {
                                       "content_accuracy": ev["content_accuracy"],
                                       "full_answer_natural_eos": ev["full_answer_natural_eos"]},
                                   "gradient_steps": 128}
    p = HERE / "M4_POSITIVE_CONTROL.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[positive] status={out['status']} alignment={alignment} 写出 {p}")


if __name__ == "__main__":
    main()
