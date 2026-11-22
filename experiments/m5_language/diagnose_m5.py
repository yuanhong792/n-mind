"""M5.1 P0 诊断（任务书第一阶段，无需重训）：定位 free-gen EM≈0 的具体原因。

检查项：
  A. 训练截断率：max_len=160 下多少训练对话被切尾（答案/问题丢失）
  B. 答案 span 的 teacher-forced EM / 字符准确率：train（记忆）vs sealed（泛化）
     ——若 train TF-EM 高而 sealed 低 ⇒ 记忆有余、泛化不足；若两者都低 ⇒ 容量/预算不足
  C. 自由生成 vs TF 的差距 = exposure bias 定量
  D. 停止原因分布、回声率（生成文本是否逐字出现在上下文）、生成长度 vs oracle 长度
  E. 路由利用率：训练累计 block_load_count（死块检测）
  F. 答案首 token 准确率（答案"开头"是否学到）
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from nmind.language.neural_dialogue import load_lm  # noqa: E402
from nmind.language.tokenizer import EOS, CharTokenizer  # noqa: E402

HERE = Path(__file__).resolve().parent
CKPT = HERE / "language_checkpoint.pt"
TOK = HERE / "tokenizer.json"
MAX_LEN = 160


def dialogue_ids(d: dict, tok: CharTokenizer) -> tuple[list[int], list[tuple[int, int, str]]]:
    """编码整条对话并返回 (ids, [(start,end,intent) 答案 span（含 EOS）])。"""
    ids: list[int] = []
    spans: list[tuple[int, int, str]] = []
    for t in d["turns"]:
        if t["role"] == "user":
            ids += tok.encode(f"用户：{t['text']}\n")
        else:
            start = len(ids)
            ids += tok.encode(f"助手：{t['text']}", add_eos=True)
            # 答案 span 不含"助手："前缀（3 个 token）
            spans.append((start + 3, len(ids), t["intent"]))
    return ids, spans


def main() -> int:
    t0 = time.perf_counter()
    tok = CharTokenizer.load(TOK)
    model, meta = load_lm(CKPT, tok)
    model.eval()
    data = HERE / "data"
    out: dict = {"experiment": "m5.1-p0-diagnose", "captured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                  "checkpoint_meta": meta}

    # ---- A. 截断率 ----
    train_payload = json.loads((data / "splits" / "train.json").read_text(encoding="utf-8"))
    truncated = 0
    for d in train_payload:
        ids, _sp = dialogue_ids(d, tok)
        if len(ids) >= MAX_LEN:
            truncated += 1
    out["A_train_truncation"] = {"n": len(train_payload), "truncated": truncated,
                                  "rate": round(truncated / len(train_payload), 4),
                                  "note": f"train_language.max_len={MAX_LEN}（超出即切尾）"}

    # ---- B/C/D. TF 答案 span / 自由生成对照（train vs sealed）----
    results: dict[str, Any] = {}
    for split in ("train", "sealed_test"):
        payload = json.loads((data / "splits" / f"{split}.json").read_text(encoding="utf-8"))
        tf_span_em = tf_span_hit = tf_char_hit = tf_char_n = 0
        n_spans = 0
        first_tok_hit = 0
        free_em = free_n = 0
        stops: Counter = Counter()
        echo = 0
        len_ratio: list[float] = []
        for d in payload:
            ids, spans = dialogue_ids(d, tok)
            ids = ids[:MAX_LEN]
            ctx_text = "\n".join(t["text"] for t in d["turns"] if t["role"] == "user")
            # TF：整序列喂入，比较答案 span 内 argmax vs 目标
            if len(ids) >= 2:
                x = torch.tensor([ids[:-1]])
                y = torch.tensor(ids[1:])
                with torch.no_grad():
                    logits, _st = model(x)
                pred = logits[0].argmax(-1)
                for (s, e, intent) in spans:
                    s_adj, e_adj = max(0, s - 1), min(len(ids) - 1, e)
                    if s_adj >= e_adj:
                        continue
                    n_spans += 1
                    tgt = y[s_adj:e_adj]
                    prd = pred[s_adj:e_adj]
                    hit = bool(torch.equal(prd, tgt))
                    tf_span_hit += int(hit)
                    tf_char_hit += int((prd == tgt).sum())
                    tf_char_n += int(tgt.numel())
                    if prd.numel() > 0:
                        first_tok_hit += int(prd[0] == tgt[0])
            # 自由生成：逐轮
            state = None
            for t in d["turns"]:
                if t["role"] == "user":
                    state = absorb(model, tok, state, f"用户：{t['text']}\n")
                else:
                    if t["intent"] in {"stmt_pos", "stmt_color", "move", "paint"}:
                        state = absorb(model, tok, state, f"助手：{t['text']}")
                        continue
                    rec, _st = generate_free(model, tok, state)
                    stops[rec.finish_reason] += 1
                    free_n += 1
                    pred = rec.text.strip()
                    free_em += int(pred == t["text"].strip())
                    if pred and pred in ctx_text:
                        echo += 1
                    ol = len(t["text"].strip())
                    if ol:
                        len_ratio.append(len(pred) / ol)
                    state = absorb(model, tok, state, f"助手：{pred}")
        results[split] = {
            "tf_answer_span_em": round(tf_span_hit / max(1, n_spans), 4),
            "tf_answer_char_acc": round(tf_char_hit / max(1, tf_char_n), 4),
            "tf_first_token_acc": round(first_tok_hit / max(1, n_spans), 4),
            "n_answer_spans": n_spans,
            "free_gen_em": round(free_em / max(1, free_n), 4),
            "free_gen_n": free_n,
            "stop_reasons": dict(stops),
            "echo_rate": round(echo / max(1, free_n), 4),
            "len_ratio_mean": round(sum(len_ratio) / max(1, len(len_ratio)), 3),
        }
    out["B_C_D_tf_vs_free"] = results

    # ---- E. 路由利用率（训练累计）----
    load = model.block_load_count.tolist()
    total = sum(load) or 1.0
    out["E_block_load_train_cum"] = {
        "load_per_block": [round(v / total, 4) for v in load],
        "dead_blocks": int(sum(1 for v in load if v < 1)),
    }

    # ---- 结论草稿（数据说话）----
    tr, se = results["train"], results["sealed_test"]
    out["findings"] = {
        "memorization_vs_generalization":
            f"train TF-span-EM={tr['tf_answer_span_em']} vs sealed={se['tf_answer_span_em']}",
        "exposure_bias_gap":
            f"sealed: TF-span-EM={se['tf_answer_span_em']} vs free-gen-EM={se['free_gen_em']}"
            f"（差 {round(se['tf_answer_span_em'] - se['free_gen_em'], 4)}）",
        "echo_dominance": f"sealed 回声率={se['echo_rate']}",
        "stop_mix": f"sealed 停止分布={dict(se['stop_reasons'])}",
    }
    (HERE / "diagnose_m5.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                            encoding="utf-8")
    print(json.dumps(out["findings"], ensure_ascii=False, indent=2), flush=True)
    print(json.dumps({"A": out["A_train_truncation"], "E": out["E_block_load_train_cum"]},
                     ensure_ascii=False), flush=True)
    print(f"→ {HERE / 'diagnose_m5.json'}（{time.perf_counter() - t0:.0f}s）", flush=True)
    return 0


def absorb(model, tok: CharTokenizer, state, text: str):
    ids = tok.encode(text, add_eos=True)
    if not ids:
        return state
    with torch.no_grad():
        _, st = model(torch.tensor([ids], dtype=torch.long), state)
    return st


def generate_free(model, tok: CharTokenizer, state):
    from nmind.language.decoder import generate
    return generate(model, tok, tok.encode("助手："), state=state,
                     max_new_tokens=32, greedy=True)


if __name__ == "__main__":
    raise SystemExit(main())
