from __future__ import annotations
import hashlib, json, math, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
import torch
from nmind.language.tokenizer_v2 import ByteBPETokenizer
from nmind.language.neural_language_v2 import LanguageV2Config, LanguageV2Model, load_language_v2
from nmind.language.bpe_decoder import effective_token_ids

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "experiments" / "language_v1"
OUT = SRC / "v15"
PARENT = SRC / "candidate_v14_main.pt"

LEGACY = [
    ("用户：你好\n助手：", "你好！"),
    ("用户：你叫什么？\n助手：", "我叫N-MIND。"),
    ("用户：谢谢\n助手：", "不客气！"),
    ("用户：小明有红色苹果\n助手：", "小明的苹果是红色的。"),
    ("用户：小红有绿色苹果\n助手：", "小红的苹果是绿色的。"),
    ("用户：小刚有蓝色苹果\n助手：", "小刚的苹果是蓝色的。"),
    ("用户：小丽有黄色苹果\n助手：", "小丽的苹果是黄色的。"),
    ("用户：我叫小明\n助手：", "你好，小明！"),
    ("用户：我叫什么？\n助手：", "你叫小明。"),
]
CASES = [
    {"id":"a_ming_red", "facts":"小明有红色苹果，小红有绿色苹果。", "query":"小明", "answer":"红色"},
    {"id":"a_hong_green", "facts":"小明有红色苹果，小红有绿色苹果。", "query":"小红", "answer":"绿色"},
    {"id":"b_ming_green", "facts":"小明有绿色苹果，小红有红色苹果。", "query":"小明", "answer":"绿色"},
    {"id":"b_hong_red", "facts":"小明有绿色苹果，小红有红色苹果。", "query":"小红", "answer":"红色"},
]

def prompt(row):
    return f"用户：{row['facts']}\n用户：{row['query']}的苹果是什么颜色？\n助手："

def digest(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

def model_sha(model):
    h = hashlib.sha256()
    for n, v in sorted(model.state_dict().items()):
        h.update(n.encode()); h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def answer_ids(tok, text):
    return tok.encode(text)

def candidate_score(model, tok, text, answer):
    """Exact causal score: logits at prompt boundary predict answer token 0."""
    pids = tok.encode(text); aids = answer_ids(tok, answer)
    valid = torch.tensor(effective_token_ids(tok))
    with torch.inference_mode():
        logits, state = model(torch.tensor([pids]))
        scores = []
        per_token = []
        for tid in aids:
            lp = torch.log_softmax(logits[0, -1].index_select(0, valid), dim=-1)
            pos = int((valid == tid).nonzero(as_tuple=True)[0][0])
            per_token.append(float(lp[pos])); scores.append(float(lp[pos]))
            logits, state = model(torch.tensor([[tid]]), state)
    return {"token_ids": aids, "token_logprobs": per_token, "logprob_sum": sum(scores),
            "mean_logprob": sum(scores) / max(1, len(scores))}

def state_trace(model, tok, text):
    ids = tok.encode(text); traces=[]; state=None
    with torch.inference_mode():
        for i, tid in enumerate(ids):
            logits, state = model(torch.tensor([[tid]]), state)
            traces.append({
                "step": i, "input_token_id": tid,
                "h_mean_abs": float(state.h.abs().mean()),
                "h_max_abs": float(state.h.abs().max()),
                "workspace_mean_abs": float(state.ws["content"].abs().mean()),
                "workspace_max_abs": float(state.ws["content"].abs().max()),
                "lm_head_input_norm": float((state.h.mean(dim=1)).norm()),
                "top5_ids": torch.topk(logits[0, -1], 5).indices.tolist(),
                "top5_values": torch.topk(logits[0, -1], 5).values.tolist(),
            })
    return traces

def load_pair(tok):
    baseline, meta = load_language_v2(PARENT, tok)
    baseline.use_router_load_update = False
    cfg = LanguageV2Config(**{**baseline.v2_config.__dict__, "query_readout": True})
    variant = LanguageV2Model(cfg)
    old = baseline.state_dict()
    missing, unexpected = variant.load_state_dict(old, strict=False)
    if set(unexpected) or set(missing) != {"query_readout_proj.weight", "query_readout_proj.bias"}:
        raise RuntimeError({"missing": missing, "unexpected": unexpected})
    variant.use_router_load_update = False
    variant.eval(); baseline.eval()
    return baseline, variant, meta

def eval_model(model, tok):
    rows=[]
    for row in CASES:
        p=prompt(row); scores={a:candidate_score(model,tok,p,a) for a in ("红色","绿色")}
        pred=max(scores, key=lambda x: scores[x]["mean_logprob"])
        rows.append({**row, "prompt":p, "scores":scores, "predicted":pred,
                     "correct":pred == row["answer"], "trace":state_trace(model,tok,p)})
    return rows

def main():
    torch.set_num_threads(1)
    tok=ByteBPETokenizer.train([p+a for p,a in LEGACY], vocab_size=4096)
    baseline, variant, meta=load_pair(tok)
    OUT.mkdir(parents=True, exist_ok=True)
    baseline_rows=eval_model(baseline,tok); variant_rows=eval_model(variant,tok)
    def summary(rows):
        return {"accuracy": sum(x["correct"] for x in rows)/len(rows),
                "rows": [{"id":x["id"],"predicted":x["predicted"],"answer":x["answer"],"scores":x["scores"]} for x in rows],
                "optimizer_steps":0,"status":"ZERO_STEP"}
    audit={"status":"FROZEN_BEFORE_TRAINING","train":LEGACY,"sealed":CASES,
           "tokenizer_hash":tok.vocab_hash(),"tokenizer_vocab_size":tok.vocab_size,
           "effective_token_count":len(effective_token_ids(tok)),"sealed_hash":digest(CASES),
           "training_steps":0,"optimizer_steps":0,"leakage":False}
    (OUT/"V15_DATASET_AUDIT.json").write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding="utf-8")
    (OUT/"V15_ANSWER_PROBABILITIES.json").write_text(json.dumps({"status":"ZERO_STEP","parent":meta,"baseline":summary(baseline_rows),"variant":summary(variant_rows)},ensure_ascii=False,indent=2),encoding="utf-8")
    (OUT/"V15_STATE_ANALYSIS.json").write_text(json.dumps({"status":"ZERO_STEP","cases":CASES,"baseline":baseline_rows,"variant":variant_rows},ensure_ascii=False,indent=2),encoding="utf-8")
    pre={"status":"PRE_REGISTERED_NOT_TRAINED","parent_checkpoint":str(PARENT),"parent_sha256":hashlib.sha256(PARENT.read_bytes()).hexdigest(),"tokenizer_sha256":tok.vocab_hash(),"sealed_sha256":digest(CASES),"baseline":"V1.4 architecture, mean h readout","variant":"query_readout=True; random query_readout_proj only","baseline_params":baseline.n_params(),"variant_params":variant.n_params(),"max_steps_per_arm":256,"total_optimization_steps":512,"already_executed_steps":0,"learning_rate":0.0005,"batch_size":2,"seeds":{"baseline":20261150,"variant":20261151},"wall_clock_limit_seconds":1800,"thresholds":{"sealed_accuracy_min":0.75,"must_exceed_fixed_answer":True,"must_exceed_recent_attribute":True,"eos_completion_rate_min":0.75,"old_skill_retention_min":0.75},"stop_conditions":["any data leakage","non-finite loss","checkpoint incompatibility","budget exceeded","old-skill regression below threshold"],"stable_modified":False}
    (OUT/"V15_PRE_REGISTRATION.json").write_text(json.dumps(pre,ensure_ascii=False,indent=2),encoding="utf-8")
    (OUT/"V15_BASELINE_EVAL.json").write_text(json.dumps(summary(baseline_rows),ensure_ascii=False,indent=2),encoding="utf-8")
    (OUT/"V15_VARIANT_EVAL.json").write_text(json.dumps(summary(variant_rows),ensure_ascii=False,indent=2),encoding="utf-8")
    (OUT/"V15_DIAGNOSIS.md").write_text("# V1.5 零步诊断\n\n见 JSON 原始证据。Baseline 使用 V1.4 Candidate；Variant 仅新增随机初始化的 query_readout_proj，未执行 optimizer.step。\n\n判定规则：只有候选答案条件对数概率在实体、颜色交换和人物顺序变化下正确反转，才可称为实体绑定；中间状态变化本身不是充分证据。\n",encoding="utf-8")
    (OUT/"V15_REPORT.md").write_text("# V1.5 报告\n\n状态：ZERO_STEP / NOT_TRAINED。答案条件概率、逐 token 状态轨迹、Baseline/Variant 零步评估和预登记已生成。新训练预算未获批准，optimizer.step=0。\n",encoding="utf-8")
    print(json.dumps({"baseline":summary(baseline_rows),"variant":summary(variant_rows),"baseline_params":baseline.n_params(),"variant_params":variant.n_params()},ensure_ascii=False))

if __name__ == "__main__": main()
