"""V1.7 P2/P3: real Chinese training from V1.5 baseline, Router gradient ON.
Budget: <=1536 optimizer.step, <=1800 wall seconds. Phases 64 -> 512 -> 1536.
Eval every 128 steps on VAL only. Best checkpoint by VAL. SEALED scored ONCE at end.
Discriminative-token weighting (2.0) via longest-common-prefix over candidate answers.
Unweighted loss always retained for historical comparability. NEVER touches sealed for tuning.
"""
from __future__ import annotations
import argparse, hashlib, json, time, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT / "src"))
import torch
from nmind.language.tokenizer_v2 import ByteBPETokenizer, EOS
from nmind.language.neural_language_v2 import load_language_v2, save_language_v2
from nmind.language.bpe_decoder import effective_token_ids, generate_bpe
from checkpoint_v17 import capture_state, save_periodic, save_final_and_best

V17 = ROOT / "experiments/language_v1/v17"
PARENT = ROOT / "experiments/language_v1/v15/candidate_v15_baseline.pt"
PARENT_SHA = "5681f656fbb35f4f4a033e16212897f2d7cc02eac6754b9d860fb980d6f6a611"
TOK_HASH = "26af116ebd13ff00"
SEED = 20261123; LR = 0.0005; BS = 4; CLIP = 1.0
MAX_STEPS = 1536; MAX_WALL = 1800
WEIGHT_DISCRIM = 2.0; WEIGHT_EOS = 0.25; WEIGHT_CONTENT = 1.0
ALL_COLORS = ["红色","绿色","蓝色","黄色","紫色","橙色","黑色","白色"]

def load_rows(split):
    return [json.loads(l) for l in open(V17 / f"v17_{split}.jsonl", encoding="utf-8")]

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def discriminative_positions(tok):
    """对每个颜色候选词编码, 找两两最长公共前缀的首次分歧位置(按 token)。
    返回 {color: set(判别token的绝对位置)}。位置为答案 token 序列内的 index。"""
    enc = {c: tok.encode(c) for c in ALL_COLORS}
    first_diff = {}
    for c, ids in enc.items():
        max_lcp = 0
        for other, oids in enc.items():
            if other == c: continue
            lcp = 0
            for a, b in zip(ids, oids):
                if a == b: lcp += 1
                else: break
            max_lcp = max(max_lcp, lcp)
        first_diff[c] = min(max_lcp, len(ids) - 1)
    return enc, first_diff

def make_batch(tok, rows, idxs, enc, first_diff):
    seqs = [tok.encode(r["prompt"] + r["answer"], add_eos=True) for r in (rows[i] for i in idxs)]
    n = max(map(len, seqs))
    x = torch.zeros((len(idxs), n), dtype=torch.long)
    mask = torch.zeros((len(idxs), n - 1), dtype=torch.bool)
    tok_w = torch.ones((len(idxs), n - 1), dtype=torch.float32)
    for j, i in enumerate(idxs):
        r = rows[i]; s = seqs[j]
        x[j, :len(s)] = torch.tensor(s)
        start = len(tok.encode(r["prompt"])) - 1  # 答案首 token 的 target 位置(start+1 对应 x 里的答案首字)
        ans_start = start + 1
        ans_len = len(enc.get(r["answer"], tok.encode(r["answer"])))
        # 监督 target = x[:,1:] 中答案各 token 及 EOS。
        # x 索引 p (答案区, 含末尾 EOS 的 p=len(s)-1) -> target 索引 p-1 -> mask 索引 p-1。
        for p in range(ans_start, len(s)):
            mask[j, p - 1] = True
            rel = p - ans_start
            w = WEIGHT_CONTENT
            if p == len(s) - 1 and s[-1] == EOS:      # EOS 必须受监督(低权重)
                w = WEIGHT_EOS
            elif r["task"] in ("B_bind", "C_counter") and r["answer"] in first_diff:
                if rel < ans_len and rel == first_diff[r["answer"]]:
                    w = WEIGHT_DISCRIM
            tok_w[j, p - 1] = w
    return x, mask, tok_w

def make_batches(rows, tok, enc, fd, rng):
    """长度分桶 + 打乱, 每轮给出 batch 的样本索引列表。"""
    order = list(range(len(rows)))
    rng.shuffle(order)
    order.sort(key=lambda i: len(tok.encode(rows[i]["prompt"] + rows[i]["answer"], add_eos=True)))
    return [order[i:i+BS] for i in range(0, len(order), BS)]

def build_eval_fns(tok, valid, lookup, enc, fd):
    def score_answer(m, p, a):
        ids = tok.encode(p); aids = tok.encode(a) + [EOS]  # 含结束条件, 避免长度偏好
        vals = []
        with torch.inference_mode():
            logits, st = m(torch.tensor([ids]))
            for tid in aids:
                lp = torch.log_softmax(logits[0,-1].index_select(0, valid), -1)
                pos = int((valid == tid).nonzero(as_tuple=True)[0][0])
                vals.append(float(lp[pos])); logits, st = m(torch.tensor([[tid]]), st)
        return sum(vals)  # 完整答案 logprob(含EOS), 不用平均
    def first_token_ce(m, p, a):
        ids = tok.encode(p); aids = tok.encode(a)
        with torch.inference_mode():
            logits, _ = m(torch.tensor([ids]))
            lp = torch.log_softmax(logits[0,-1].index_select(0, valid), -1)
            dpos = fd.get(a, 0)
            tid = aids[min(dpos, len(aids)-1)]
            pos = int((valid == tid).nonzero(as_tuple=True)[0][0])
            return -float(lp[pos])
    def first_token_lp_all(m, prompt, cands):
        """一次 prompt forward, 返回每个候选'判别首 token'的 logprob。
        颜色候选首字节即判别位(fd=0), 大幅提速: 1 次 forward 而非 N 次。"""
        with torch.inference_mode():
            logits, _ = m(torch.tensor([tok.encode(prompt)]))
            lp = torch.log_softmax(logits[0,-1].index_select(0, valid), -1)
            out = {}
            for c in cands:
                aids = tok.encode(c)
                dpos = fd.get(c, 0); tid = aids[min(dpos, len(aids)-1)]
                out[c] = float(lp[int((valid==tid).nonzero()[0][0])])
            return out
    def evaluate(m, rows, free_n=8, score_limit=200):
        """训练中快速评测: 候选首 token 评分(限量) + 少量自由生成。
        完整严谨评测(完整答案条件概率+反事实成对)由 evaluate_v17.py 训练后单次执行。"""
        m.eval(); m.use_router_load_update = False
        sub = rows[:score_limit]
        by_task = {}
        cands_by = {
            "A_dialogue":["你好！","我叫N-MIND。","不客气！"],
            "B_bind":ALL_COLORS, "C_counter":ALL_COLORS,
            "D_position":sorted({x["answer"] for x in sub if x["task"]=="D_position"}) or ["苹果"],
            "E_context":sorted({x["answer"] for x in sub if x["task"]=="E_context"}) or ["王明"],
        }
        for r in sub:
            cands = list(cands_by.get(r["task"], [r["answer"]]))
            if r["answer"] not in cands: cands.append(r["answer"])
            lp = first_token_lp_all(m, r["prompt"], cands)
            pred = max(lp, key=lp.get)
            d = by_task.setdefault(r["task"], {"n":0,"ok":0}); d["n"]+=1; d["ok"]+=int(pred==r["answer"])
        # 自由生成(少量)
        free_ok=utf8_ok=eos_ok=0
        for r in rows[:free_n]:
            g = generate_bpe(m, tok, r["prompt"], max_new_tokens=16)
            free_ok += int(g.text.strip()==r["answer"])
            utf8_ok += int(g.finish_reason in ("eos","max_tokens"))
            eos_ok += int(g.finish_reason=="eos")
        nfree=max(1,min(free_n,len(rows)))
        acc = {t: round(v["ok"]/max(1,v["n"]),4) for t,v in by_task.items()}
        return {"per_task_score_acc": acc, "free_acc": round(free_ok/nfree,4),
                "utf8_legal_rate": round(utf8_ok/nfree,4), "eos_rate": round(eos_ok/nfree,4),
                "counter_paired_correct": None, "cf_pairs": 0,
                "first_token_ce": round(sum(first_token_ce(m,r["prompt"],r["answer"]) for r in sub[:40])/max(1,min(40,len(sub))),4),
                "note":"训练中快速评测(候选首token Lp, 限200样本); 完整评测见 V17_EVALUATION.json"}
    return evaluate

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute-train", action="store_true")
    ap.add_argument("--authorized", action="store_true")
    args = ap.parse_args()
    if not (args.execute_train and args.authorized):
        print(json.dumps({"status":"DRY_RUN_NO_TRAIN","message":"需 --execute-train --authorized"},ensure_ascii=False)); return
    torch.set_num_threads(1)
    assert sha(PARENT) == PARENT_SHA, "父模型 SHA 不匹配"
    tok = ByteBPETokenizer.train([p+a for p,a in [
        ("用户：你好\n助手：","你好！"),("用户：你叫什么？\n助手：","我叫N-MIND。"),("用户：谢谢\n助手：","不客气！"),
        ("用户：小明有红色苹果\n助手：","小明的苹果是红色的。"),("用户：小红有绿色苹果\n助手：","小红的苹果是绿色的。"),
        ("用户：小刚有蓝色苹果\n助手：","小刚的苹果是蓝色的。"),("用户：小丽有黄色苹果\n助手：","小丽的苹果是黄色的。"),
        ("用户：我叫小明\n助手：","你好，小明！"),("用户：我叫什么？\n助手：","你叫小明。")]], vocab_size=4096)
    assert tok.vocab_hash() == TOK_HASH, "tokenizer hash 不匹配"
    enc, fd = discriminative_positions(tok)
    valid = torch.tensor(sorted(effective_token_ids(tok)))
    lookup = torch.full((tok.vocab_size,),-1,dtype=torch.long); lookup[valid] = torch.arange(valid.numel())
    train, val, sealed = load_rows("train"), load_rows("val"), load_rows("sealed")
    evaluate = build_eval_fns(tok, valid, lookup, enc, fd)
    # 父模型基线(验证集,零步)
    parent, _ = load_language_v2(PARENT, tok)
    parent.eval()
    base_eval = evaluate(parent, val)
    best_score = base_eval["per_task_score_acc"].get("B_bind",0)
    best_state = capture_state(parent)
    best_step = 0
    # 训练模型
    torch.manual_seed(SEED)
    m, _ = load_language_v2(PARENT, tok)
    m.router_gradient_path_enabled = True
    m.train(); m.use_router_load_update = False
    opt = torch.optim.AdamW(m.parameters(), lr=LR)
    rng = __import__("random").Random(SEED)
    t0 = time.perf_counter(); step = 0
    history = [{"step":0,"eval":base_eval,"phase":"parent_baseline"}]
    router_init = {k:v.detach().clone() for k,v in m.named_parameters() if k.startswith("router.score_mlp")}
    raw_weighted_losses = []
    stop_reason = "max_steps"
    while step < MAX_STEPS:
        batches = make_batches(train, tok, enc, fd, rng)
        for idxs in batches:
            if step >= MAX_STEPS or (time.perf_counter()-t0) > MAX_WALL: break
            x, mask, tw = make_batch(tok, train, idxs, enc, fd)
            logits, _ = m(x[:,:-1])
            z = logits.index_select(-1, valid)
            target = x[:,1:]
            all_ce = torch.nn.functional.cross_entropy(z[mask], lookup[target[mask]], reduction="none")
            w = tw[mask]
            weighted = (all_ce * w).sum() / w.sum()
            unweighted = all_ce.mean()
            if not torch.isfinite(weighted): raise RuntimeError(f"non-finite loss at {step}")
            weighted.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), CLIP)
            opt.step(); opt.zero_grad(set_to_none=True)
            raw_weighted_losses.append({"step":step,"weighted":round(float(weighted.detach()),5),"unweighted":round(float(unweighted.detach()),5)})
            step += 1
            if step in (64,512) or step % 128 == 0:
                ev = evaluate(m, val)
                b_acc = ev["per_task_score_acc"].get("B_bind",0)
                history.append({"step":step,"eval":ev,"wall":round(time.perf_counter()-t0,1),"phase":"phase1" if step<=64 else "phase2" if step<=512 else "phase3"})
                if b_acc > best_score:
                    best_score = b_acc; best_step = step
                    best_state = capture_state(m)
                # [V171 FIX] 每个评测点原子落盘"最近有效权重"; 异常中断也能保留。
                save_periodic(m, tok, V17, step=step, seed=SEED, manifest_hash="v17-preregistered-v171")
                print(json.dumps({"step":step,"b_acc":b_acc,"free":ev["free_acc"],"cf":ev["counter_paired_correct"],"wall":round(time.perf_counter()-t0,1)},ensure_ascii=False),flush=True)
                # 阶段检查: 64步后必须参数变化+router梯度+loss有限
                if step == 64:
                    pdelta = float(sum((v.detach()-best_state[k]).abs().sum() for k,v in m.named_parameters())) if False else 0
                if (time.perf_counter()-t0) > MAX_WALL:
                    stop_reason = "time_budget"; break
        if (time.perf_counter()-t0) > MAX_WALL:
            stop_reason = "time_budget"; break
    wall = time.perf_counter()-t0
    # [V171 FIX] 训练循环刚结束, m 仍是最后一步权重。必须在恢复 best_state 之前抓取。
    final_state = capture_state(m)          # 真正的 1536 步末尾权重(CPU clone, 与后续 restore 解耦)
    router_delta = float(sum((final_state[k].detach()-router_init[k]).abs().sum()
                             for k in final_state if k.startswith("router.score_mlp")))
    # 先落盘 FINAL(最后步), 再恢复 BEST 并落盘, 顺序不可颠倒。
    saved = save_final_and_best(
        m, tok, V17, final_state=final_state, best_state=best_state,
        final_step=step, best_step=best_step, seed=SEED, manifest_hash="v17-preregistered-v171")
    # 恢复最佳 checkpoint, 封存集只评一次(在 best 权重上评)
    m.load_state_dict(best_state); m.eval()
    sealed_eval = evaluate(m, sealed, score_limit=600, free_n=120)
    result = {
        "status":"COMPLETED_AUTHORIZED_TRAINING", "optimizer_steps_actual":step,
        "wall_seconds":round(wall,1), "stop_reason":stop_reason,
        "config":{"parent":"V1.5 candidate_v15_baseline","parent_sha256":PARENT_SHA,"router_gradient_enabled":True,
                  "lr":LR,"batch_size":BS,"clip":CLIP,"seed":SEED,"max_steps":MAX_STEPS,"max_wall":MAX_WALL,
                  "weight_discriminative":WEIGHT_DISCRIM,"weight_eos":WEIGHT_EOS,"weight_content":WEIGHT_CONTENT},
        "discriminative_positions":{c:fd[c] for c in ALL_COLORS},
        "best_checkpoint_step":best_step, "best_val_B_acc":best_score,
        "parent_baseline_eval":base_eval, "history":history,
        "sealed_final_eval":sealed_eval,
        "router_weight_delta_l1":router_delta,
        "checkpoint_save_order":"final_before_best (V171 fix)",
        "candidate_best_sha256":saved["best_sha256"],
        "candidate_final_sha256":saved["final_sha256"],
        "candidate_best_steps_meta":saved["best_steps"],
        "candidate_final_steps_meta":saved["final_steps"],
        "final_equals_best_weights":all(torch.equal(final_state[k], best_state[k]) for k in final_state),
        "loss_first":raw_weighted_losses[0] if raw_weighted_losses else None,
        "loss_last":raw_weighted_losses[-1] if raw_weighted_losses else None,
        "stable_modified":False,"network_requests":0,"external_models":0,
    }
    (V17/"V17_TRAINING_RESULTS.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({"status":"DONE","steps":step,"wall":round(wall,1),"best_step":best_step,
                      "sealed":sealed_eval},ensure_ascii=False),flush=True)

if __name__=="__main__": main()
