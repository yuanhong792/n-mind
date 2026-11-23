"""V1.6 冻结参数诊断：P1 逐token loss / P2 TF vs 自由生成 / P3 实体绑定状态。不执行 optimizer.step。"""
from __future__ import annotations
import sys, json, hashlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
import torch
from nmind.language.tokenizer_v2 import ByteBPETokenizer, EOS
from nmind.language.bpe_decoder import effective_token_ids, generate_bpe
from nmind.language.neural_language_v2 import load_language_v2

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "experiments" / "language_v1"
V15 = SRC / "v15"
OUT = SRC / "v16"

LEGACY = [
    ("用户：你好\n助手：", "你好！"), ("用户：你叫什么？\n助手：", "我叫N-MIND。"),
    ("用户：谢谢\n助手：", "不客气！"), ("用户：小明有红色苹果\n助手：", "小明的苹果是红色的。"),
    ("用户：小红有绿色苹果\n助手：", "小红的苹果是绿色的。"), ("用户：小刚有蓝色苹果\n助手：", "小刚的苹果是蓝色的。"),
    ("用户：小丽有黄色苹果\n助手：", "小丽的苹果是黄色的。"), ("用户：我叫小明\n助手：", "你好，小明！"),
    ("用户：我叫什么？\n助手：", "你叫小明。"),
]
# train_v15.py 里 TRAIN（用于训练）和 SEALED/COUNTER（用于评测）
TRAIN = [
    ("用户：小明有红色苹果，小红有绿色苹果。\n用户：小明的苹果是什么颜色？\n助手：", "红色"),
    ("用户：小明有红色苹果，小红有绿色苹果。\n用户：小红的苹果是什么颜色？\n助手：", "绿色"),
    ("用户：小明有绿色苹果，小红有红色苹果。\n用户：小明的苹果是什么颜色？\n助手：", "绿色"),
    ("用户：小明有绿色苹果，小红有红色苹果。\n用户：小红的苹果是什么颜色？\n助手：", "红色"),
]
SEALED = [
    ("用户：小刚有蓝色书包，小丽有黄色书包。\n用户：小丽的书包是什么颜色？\n助手：", "黄色"),
    ("用户：小刚有黄色杯子，小丽有蓝色杯子。\n用户：小刚的杯子是什么颜色？\n助手：", "黄色"),
    ("用户：小丽有绿色苹果，小明有蓝色苹果。\n用户：小明的苹果是什么颜色？\n助手：", "蓝色"),
    ("用户：小红有黄色书包，小刚有绿色书包。\n用户：小红的书包是什么颜色？\n助手：", "黄色"),
]

def tid_text(tok, t):
    if t == EOS: return "<EOS>"
    if t == 0: return "<PAD>"
    reverse = {4 + i: i for i in range(256)}
    for i, pair in enumerate(tok.merges): reverse[260 + i] = pair
    if t not in reverse: return f"<id:{t}>"
    out = bytearray()
    def exp(x):
        if x >= 260:
            if x not in reverse: raise KeyError(x)
            for c in reverse[x]: exp(c)
        else:
            out.append(x - 4)
    try:
        exp(t)
        return bytes(out).decode("utf-8")
    except Exception:
        return f"<id:{t}>"

def fp(model):
    h = hashlib.sha256()
    for n, v in sorted(model.state_dict().items()):
        h.update(n.encode()); h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def make_ctx(tok, model):
    valid = torch.tensor(effective_token_ids(tok))
    lookup = torch.full((tok.vocab_size,), -1, dtype=torch.long)
    lookup[valid] = torch.arange(valid.numel())
    return valid, lookup

def per_token_loss(tok, model, valid, lookup, prompt, answer):
    """复现 train_v15 的掩码与 CE，返回每个监督位置的逐 token loss。"""
    seq = tok.encode(prompt + answer, add_eos=True)
    start = len(tok.encode(prompt)) - 1
    n = len(seq)
    x = torch.tensor([seq])
    with torch.inference_mode():
        logits, _ = model(x[:, :-1])
    z = logits[0].index_select(-1, valid)          # [n-1, Veff]
    target = seq[1:]                               # [n-1]
    mask_pos = list(range(start, n - 1))           # train_v15: mask[i,start:n-1]
    items = []
    for pos in range(n - 1):
        tid = target[pos]
        lp = torch.log_softmax(z[pos], dim=-1)
        ce = -float(lp[lookup[tid]]) if lookup[tid] >= 0 else None
        items.append({"pos": pos, "target_id": tid, "target_text": tid_text(tok, tid),
                      "ce": round(ce, 6) if ce is not None else None,
                      "supervised": pos in mask_pos})
    sup = [it for it in items if it["supervised"]]
    return {"prompt": prompt, "answer": answer, "seq_len": n, "start": start,
            "supervised_positions": mask_pos,
            "all_tokens": items,
            "supervised_tokens": sup,
            "mean_loss": round(sum(i["ce"] for i in sup) / len(sup), 6) if sup else None,
            "supervised_texts": [i["target_text"] for i in sup]}

def candidate_score(tok, model, valid, prompt, answer):
    pids = tok.encode(prompt); aids = tok.encode(answer)
    with torch.inference_mode():
        logits, state = model(torch.tensor([pids]))
        tls = []
        for tid in aids:
            lp = torch.log_softmax(logits[0, -1].index_select(0, valid), dim=-1)
            pos = int((valid == tid).nonzero(as_tuple=True)[0][0])
            tls.append(round(float(lp[pos]), 6))
            logits, state = model(torch.tensor([[tid]]), state)
    return {"token_ids": aids, "token_logprobs": tls, "mean_logprob": round(sum(tls)/len(tls), 6)}

def state_trace(tok, model, prompt):
    ids = tok.encode(prompt); rows = []; state = None
    with torch.inference_mode():
        for i, tid in enumerate(ids):
            logits, state = model(torch.tensor([[tid]]), state)
            rows.append({"step": i, "tok": tid_text(tok, tid),
                         "h_mean_abs": round(float(state.h.abs().mean()), 5),
                         "ws_content_mean_abs": round(float(state.ws["content"].abs().mean()), 5),
                         "lm_input_norm": round(float(state.h.mean(dim=1).norm()), 5),
                         "top3": [(int(x), tid_text(tok, int(x))) for x in torch.topk(logits[0, -1], 3).indices.tolist()]})
    return rows

def main():
    torch.set_num_threads(1)
    tok = ByteBPETokenizer.train([p + a for p, a in LEGACY], vocab_size=4096)
    model, meta = load_language_v2(V15 / "candidate_v15_baseline.pt", tok)
    model.eval(); model.use_router_load_update = False
    valid, lookup = make_ctx(tok, model)
    result = {"model": "candidate_v15_baseline.pt", "model_sha256": fp(model),
              "tokenizer_hash": tok.vocab_hash(), "checkpoint_meta": meta,
              "train_loss_breakdown": [], "sealed_loss_breakdown": [],
              "p2": [], "p3": []}

    # P1: 逐 token loss（训练集 + 封存集）
    for p, a in TRAIN:
        result["train_loss_breakdown"].append(per_token_loss(tok, model, valid, lookup, p, a))
    for p, a in SEALED:
        result["sealed_loss_breakdown"].append(per_token_loss(tok, model, valid, lookup, p, a))

    # P2: 三模式对比（用训练里出现的同一组事实，做实体/颜色/顺序交换）
    cases = [
        {"id": "a_ming_ask_hong", "prompt": "用户：小明有红色苹果，小红有绿色苹果。\n用户：小红的苹果是什么颜色？\n助手：", "expected": "绿色"},
        {"id": "a_ming_ask_ming", "prompt": "用户：小明有红色苹果，小红有绿色苹果。\n用户：小明的苹果是什么颜色？\n助手：", "expected": "红色"},
        {"id": "b_swap_color_ask_ming", "prompt": "用户：小明有绿色苹果，小红有红色苹果。\n用户：小明的苹果是什么颜色？\n助手：", "expected": "绿色"},
        {"id": "b_swap_color_ask_hong", "prompt": "用户：小明有绿色苹果，小红有红色苹果。\n用户：小红的苹果是什么颜色？\n助手：", "expected": "红色"},
        {"id": "c_swap_order_ask_hong", "prompt": "用户：小红有绿色苹果，小明有红色苹果。\n用户：小红的苹果是什么颜色？\n助手：", "expected": "绿色"},
    ]
    for c in cases:
        p = c["prompt"]
        scores = {a: candidate_score(tok, model, valid, p, a) for a in ("红色", "绿色")}
        pred = max(scores, key=lambda a: scores[a]["mean_logprob"])
        gen = generate_bpe(tok and model, tok, p, max_new_tokens=16)
        # teacher-forced 完整答案（gold）条件概率
        tf = candidate_score(tok, model, valid, p, c["expected"])
        result["p2"].append({"id": c["id"], "prompt": p, "expected": c["expected"],
                             "candidate_scores": scores, "candidate_pred": pred,
                             "candidate_correct": pred == c["expected"],
                             "free_generation": {"text": gen.text, "token_ids": gen.token_ids,
                                                 "finish_reason": gen.finish_reason},
                             "tf_gold_mean_logprob": tf["mean_logprob"],
                             "tf_gold_token_logprobs": tf["token_logprobs"]})

    # P3: 逐 token 状态轨迹（同一描述，问题只改实体）
    for q in ("小明", "小红"):
        p = f"用户：小明有红色苹果，小红有绿色苹果。\n用户：{q}的苹果是什么颜色？\n助手："
        result["p3"].append({"query": q, "prompt": p,
                             "trace_last8": state_trace(tok, model, p)[-8:]})

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "_diag_raw.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    # 打印摘要
    print("model_sha256:", result["model_sha256"])
    print("\n=== P1 训练集逐token CE ===")
    for r in result["train_loss_breakdown"]:
        print(f"答案{r['answer']} mean={r['mean_loss']} 监督={[(i['target_text'],i['ce']) for i in r['supervised_tokens']]}")
    print("\n=== P1 封存集逐token CE ===")
    for r in result["sealed_loss_breakdown"]:
        print(f"答案{r['answer']} mean={r['mean_loss']} 监督={[(i['target_text'],i['ce']) for i in r['supervised_tokens']]}")
    print("\n=== P2 三模式 ===")
    for r in result["p2"]:
        print(f"{r['id']} exp={r['expected']} cand={r['candidate_pred']}({r['candidate_correct']}) free={r['free_generation']['text']!r} tf_gold_lp={r['tf_gold_mean_logprob']}")
        print(f"   red={r['candidate_scores']['红色']['mean_logprob']} green={r['candidate_scores']['绿色']['mean_logprob']}")
    print("\n=== P3 最后一步 top3 ===")
    for r in result["p3"]:
        print(f"问{r['query']}: {r['trace_last8'][-1]['top3']}")

if __name__ == "__main__":
    main()
