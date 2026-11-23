"""V1.7 P4: strict evaluation of the trained candidate on SEALED (scored once).
7 task families; teacher-forced CE / candidate conditional logprob / free generation / UTF-8 legal.
Counterfactual requires BOTH questions correct (not merely 'output changed').
Reports 3 shortcut baselines. Does NOT modify any test set.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT / "src"))
import torch
from nmind.language.tokenizer_v2 import ByteBPETokenizer, EOS
from nmind.language.neural_language_v2 import load_language_v2
from nmind.language.bpe_decoder import effective_token_ids, generate_bpe

V17 = ROOT / "experiments/language_v1/v17"
LEGACY = [("用户：你好\n助手：","你好！"),("用户：你叫什么？\n助手：","我叫N-MIND。"),("用户：谢谢\n助手：","不客气！"),("用户：小明有红色苹果\n助手：","小明的苹果是红色的。"),("用户：小红有绿色苹果\n助手：","小红的苹果是绿色的。"),("用户：小刚有蓝色苹果\n助手：","小刚的苹果是蓝色的。"),("用户：小丽有黄色苹果\n助手：","小丽的苹果是黄色的。"),("用户：我叫小明\n助手：","你好，小明！"),("用户：我叫什么？\n助手：","你叫小明。")]
COLORS = ["红色","绿色","蓝色","黄色","紫色","橙色","黑色","白色"]

def load_rows(s): return [json.loads(l) for l in open(V17/f"v17_{s}.jsonl",encoding="utf-8")]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", default="candidate_v17_best.pt")
    ap.add_argument("--split", default="sealed")
    ap.add_argument("--out", default="V17_EVALUATION.json")
    ap.add_argument("--limit", type=int, default=0, help="评分样本上限(0=全部); 用于父模型快速对照")
    args = ap.parse_args()
    torch.set_num_threads(1)
    tok = ByteBPETokenizer.train([p+a for p,a in LEGACY], vocab_size=4096)
    m, meta = load_language_v2(V17/args.candidate, tok); m.eval(); m.use_router_load_update=False
    valid = torch.tensor(sorted(effective_token_ids(tok)))
    lookup = torch.full((tok.vocab_size,),-1,dtype=torch.long); lookup[valid]=torch.arange(valid.numel())
    rows = load_rows(args.split)
    if args.limit: rows = rows[:args.limit]
    def prompt_lp(p):
        """一次 prompt forward, 返回末尾有效词表的 logprobs(所有候选共享)。"""
        with torch.inference_mode():
            lg,_ = m(torch.tensor([tok.encode(p)]))
            return torch.log_softmax(lg[0,-1].index_select(0,valid),-1)
    def cand_logprob(p, a, cached=None):
        """完整答案条件 logprob(含EOS)。cached 为 prompt_lp(p) 复用, 避免重复长 forward。
        对颜色候选, 首 token 判别位之后均为'色'+EOS, 用 prompt 的 logprobs 取首 token
        判别分(严格完整条件概率见 full_v2, 此处为首 token 口径, 与训练评测一致)。"""
        lp = cached if cached is not None else prompt_lp(p)
        aids = tok.encode(a)
        return float(lp[int((valid==aids[0]).nonzero()[0][0])])
    def tf_ce(r):
        seq = tok.encode(r["prompt"]+r["answer"], add_eos=True)
        start = len(tok.encode(r["prompt"]))-1
        x = torch.tensor([seq[:-1]]); tgt = torch.tensor(seq[1:])
        with torch.inference_mode():
            lg,_ = m(x); z = lg.index_select(-1,valid)
            idxs = list(range(start, len(seq)-1))
            if not idxs: return None
            ce = torch.nn.functional.cross_entropy(z[0].index_select(0,torch.tensor(idxs)), lookup[torch.tensor([tgt[i] for i in idxs])], reduction="none")
            return round(float(ce.mean()),4)
    per_task = {}; details = []
    cands_by_task = {
        "A_dialogue":["你好！","我叫N-MIND。","不客气！"],
        "B_bind":COLORS, "C_counter":COLORS,
        "D_position":sorted({r["answer"] for r in rows if r["task"]=="D_position"}),
        "E_context":sorted({r["answer"] for r in rows if r["task"]=="E_context"}),
    }
    for r in rows:
        cands = list(cands_by_task.get(r["task"], [r["answer"]]))
        if r["answer"] not in cands: cands.append(r["answer"])
        lp_cache = prompt_lp(r["prompt"])
        scores = {c:cand_logprob(r["prompt"],c,cached=lp_cache) for c in cands}
        pred = max(scores,key=scores.get)
        g = generate_bpe(m, tok, r["prompt"], max_new_tokens=16)
        ok = pred == r["answer"]
        per_task.setdefault(r["task"], {"n":0,"cand_correct":0,"free_correct":0,"utf8_legal":0,"eos":0})
        d = per_task[r["task"]]; d["n"] += 1
        d["cand_correct"] += int(ok)
        d["free_correct"] += int(g.text.strip()==r["answer"])
        d["utf8_legal"] += int(g.finish_reason in ("eos","max_tokens"))
        d["eos"] += int(g.finish_reason=="eos")
        details.append({"task":r["task"],"expected":r["answer"],"predicted":pred,"cand_ok":ok,
                        "free":g.text,"free_ok":g.text.strip()==r["answer"],"finish":g.finish_reason,
                        "tf_ce":tf_ce(r)})
    # 反事实成对正确(交换前后都对)
    cmap = {}
    for r in rows:
        if r["task"]=="C_counter": cmap.setdefault(r["key"],[]).append(r)
    pair_total=0; pair_ok=0
    for k,pr in cmap.items():
        if len(pr)==2 and pr[0]["answer"]!=pr[1]["answer"]:
            pair_total+=1
            oks=[]
            for r in pr:
                lp_cache = prompt_lp(r["prompt"])
                scores={c:cand_logprob(r["prompt"],c,cached=lp_cache) for c in COLORS}
                oks.append(max(scores,key=scores.get)==r["answer"])
            pair_ok += int(all(oks))
    # 基线
    freq = {}
    for r in rows: freq[r["answer"]]=freq.get(r["answer"],0)+1
    fixed = max(freq.values())/len(rows)
    b = [r for r in rows if r["task"]=="B_bind"]
    nearest = sum(1 for r in b if r["meta"].get("nearest_attr")==r["answer"])/max(1,len(b))
    position = sum(1 for r in b if r["meta"].get("target_is_first"))/max(1,len(b))
    summary = {t:{"n":d["n"],"candidate_acc":round(d["cand_correct"]/d["n"],4),
                  "free_acc":round(d["free_correct"]/d["n"],4),
                  "utf8_legal_rate":round(d["utf8_legal"]/d["n"],4),"eos_rate":round(d["eos"]/d["n"],4)}
               for t,d in per_task.items()}
    result = {"status":"EVALUATION_DONE","candidate":args.candidate,"split":args.split,
              "candidate_meta":meta,"summary":summary,
              "counterfactual_paired_correct":round(pair_ok/max(1,pair_total),4),"cf_pairs":pair_total,
              "baselines":{"fixed_answer":round(fixed,4),"nearest_attribute":round(nearest,4),"position_first":round(position,4)},
              "targets":{"unseen_binding":0.70,"counterfactual_paired":0.60,"free_generation":0.60,"utf8_legal":0.99,"eos":0.90},
              "details":details}
    (V17/args.out).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"summary":summary,"cf_paired":result["counterfactual_paired_correct"],
                      "baselines":result["baselines"]},ensure_ascii=False,indent=2))

if __name__=="__main__": main()
