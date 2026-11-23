"""N-MIND V1.6 A/B Router-gradient CPU training.
ONLY variable: router_gradient_path_enabled (A=False, B=True). Same seed, same parent, same data.
DEFAULT = dry-run (validation only). Real training requires --execute-train --authorized.
NO external pretrained models. N-Cell/BlockRouter/GlobalWorkspace preserved.
"""
from __future__ import annotations
import argparse, hashlib, json, random, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
import torch
from nmind.language.tokenizer_v2 import ByteBPETokenizer
from nmind.language.neural_language_v2 import load_language_v2, save_language_v2
from nmind.language.bpe_decoder import effective_token_ids, generate_bpe

V16 = ROOT / "experiments/language_v1/v16"
CKPT = ROOT / "experiments/language_v1/v15/candidate_v15_baseline.pt"
PARENT_SHA = "5681f656fbb35f4f4a033e16212897f2d7cc02eac6754b9d860fb980d6f6a611"
TOK_HASH = "26af116ebd13ff00"
SEED = 20261123
LR = 0.0005
BS = 2
MAX_STEPS = 256
MAX_TOTAL = 512
# VERBATIM from train_v15.py
LEGACY = [("用户：你好\n助手：","你好！"),("用户：你叫什么？\n助手：","我叫N-MIND。"),("用户：谢谢\n助手：","不客气！"),("用户：小明有红色苹果\n助手：","小明的苹果是红色的。"),("用户：小红有绿色苹果\n助手：","小红的苹果是绿色的。"),("用户：小刚有蓝色苹果\n助手：","小刚的苹果是蓝色的。"),("用户：小丽有黄色苹果\n助手：","小丽的苹果是黄色的。"),("用户：我叫小明\n助手：","你好，小明！"),("用户：我叫什么？\n助手：","你叫小明。")]
TRAIN = [("用户：小明有红色苹果，小红有绿色苹果。\n用户：小明的苹果是什么颜色？\n助手：","红色"),("用户：小明有红色苹果，小红有绿色苹果。\n用户：小红的苹果是什么颜色？\n助手：","绿色"),("用户：小明有绿色苹果，小红有红色苹果。\n用户：小明的苹果是什么颜色？\n助手：","绿色"),("用户：小明有绿色苹果，小红有红色苹果。\n用户：小红的苹果是什么颜色？\n助手：","红色")]
SEALED = [("用户：小刚有蓝色书包，小丽有黄色书包。\n用户：小丽的书包是什么颜色？\n助手：","黄色"),("用户：小刚有黄色杯子，小丽有蓝色杯子。\n用户：小刚的杯子是什么颜色？\n助手：","黄色"),("用户：小丽有绿色苹果，小明有蓝色苹果。\n用户：小明的苹果是什么颜色？\n助手：","蓝色"),("用户：小红有黄色书包，小刚有绿色书包。\n用户：小红的书包是什么颜色？\n助手：","黄色")]
CF_PATH = V16 / "V16_COUNTERFACTUAL.json"

def sha_file(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def build_tokenizer(): return ByteBPETokenizer.train([p + a for p, a in LEGACY], vocab_size=4096)
def load_cf():
    d = json.loads(CF_PATH.read_text(encoding="utf-8"))
    rows = [(x["prompt"], x["answer"]) for x in d["rows"]]
    return rows, d["sha256"]

def preflight():
    """Environment + parent integrity. Training must never start if this fails."""
    errs = []
    if not CKPT.exists(): errs.append(f"parent checkpoint missing: {CKPT}")
    model_sha = sha_file(CKPT) if CKPT.exists() else None
    if model_sha != PARENT_SHA: errs.append(f"parent SHA mismatch: {model_sha} != {PARENT_SHA}")
    tok = build_tokenizer()
    if tok.vocab_hash() != TOK_HASH: errs.append(f"tokenizer hash mismatch: {tok.vocab_hash()} != {TOK_HASH}")
    try:
        m, meta = load_language_v2(CKPT, tok)
        n_params = m.n_params()
        if n_params != 279363: errs.append(f"param count mismatch: {n_params} != 279363")
        if m.v2_config.query_readout: errs.append("parent has query_readout=True (unexpected QA params)")
        bad = [k for k, v in m.state_dict().items() if v.is_floating_point() and not torch.isfinite(v).all().item()]
        if bad: errs.append(f"nonfinite tensors: {bad}")
    except Exception as e:
        errs.append(f"strict load failed: {type(e).__name__}: {e}")
    # independent counterfactual file integrity
    try:
        cf_rows, cf_sha = load_cf()
        canon = hashlib.sha256(json.dumps([[p,a] for p,a in cf_rows],ensure_ascii=False,sort_keys=True).encode()).hexdigest()
        if canon != cf_sha: errs.append(f"counterfactual file hash mismatch: {canon} != {cf_sha}")
    except Exception as e:
        errs.append(f"counterfactual file error: {e}")
    return tok, errs, {"parent_sha256": model_sha, "tokenizer_hash": tok.vocab_hash()}

def make_batch_masks(tok, batch):
    seq = [tok.encode(p + a, add_eos=True) for p, a in batch]
    n = max(map(len, seq))
    x = torch.zeros((BS, n), dtype=torch.long)
    mask = torch.zeros((BS, n - 1), dtype=torch.bool)
    for i, (p, a) in enumerate(batch):
        x[i, :len(seq[i])] = torch.tensor(seq[i])
        start = len(tok.encode(p)) - 1
        mask[i, start:len(seq[i]) - 1] = True
    return x, mask

def train_arm(tok, enabled, name, valid, lookup):
    torch.manual_seed(SEED); random.seed(SEED)
    # reload parent from disk each arm -> bit-identical initial weights (no cross-arm carry-over)
    m, _ = load_language_v2(CKPT, tok)
    m.router_gradient_path_enabled = enabled
    m.train(); m.use_router_load_update = False
    opt = torch.optim.AdamW(m.parameters(), lr=LR)
    router_before = {k: v.detach().clone() for k, v in m.named_parameters() if k.startswith("router.score_mlp")}
    losses = []
    t0 = time.perf_counter()
    for step in range(MAX_STEPS):
        batch = [TRAIN[(step * BS + i) % len(TRAIN)] for i in range(BS)]
        x, mask = make_batch_masks(tok, batch)
        logits, _ = m(x[:, :-1])
        z = logits.index_select(-1, valid)
        target = x[:, 1:]
        loss = torch.nn.functional.cross_entropy(z[mask], lookup[target[mask]])
        if not torch.isfinite(loss): raise RuntimeError(f"non-finite loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        losses.append(float(loss.detach()))
    m.eval(); m.use_router_load_update = False
    router_delta = float(sum((v.detach() - router_before[k]).abs().sum() for k, v in m.named_parameters() if k.startswith("router.score_mlp")))
    return m, {"name": name, "router_gradient_path_enabled": enabled, "seed": SEED,
               "steps": len(losses), "loss_first": losses[0], "loss_last": losses[-1],
               "losses": losses, "wall_seconds": time.perf_counter() - t0,
               "router_weight_delta_l1": router_delta}

def score_answer(m, tok, p, a, valid, lookup):
    """Full conditional logprob of answer tokens (NOT free-generation score)."""
    ids = tok.encode(p); aids = tok.encode(a); vals = []
    with torch.inference_mode():
        logits, st = m(torch.tensor([ids]))
        for tid in aids:
            lp = torch.log_softmax(logits[0, -1].index_select(0, valid), -1)
            pos = int((valid == tid).nonzero(as_tuple=True)[0][0])
            vals.append(float(lp[pos]))
            logits, st = m(torch.tensor([[tid]]), st)
    return sum(vals) / max(1, len(vals))

def first_token_ce(m, tok, p, a, valid, lookup):
    """CE of the discriminative answer FIRST token only."""
    ids = tok.encode(p); aids = tok.encode(a)
    with torch.inference_mode():
        logits, _ = m(torch.tensor([ids]))
        lp = torch.log_softmax(logits[0, -1].index_select(0, valid), -1)
        pos = int((valid == aids[0]).nonzero(as_tuple=True)[0][0])
        return -float(lp[pos])

def evaluate(m, tok, valid, lookup, cf_rows):
    colors = ("红色", "绿色", "蓝色", "黄色")
    out = []
    for rows, label in ((SEALED, "sealed"), (cf_rows, "independent_counterfactual")):
        for p, a in rows:
            scores = {c: score_answer(m, tok, p, c, valid, lookup) for c in colors}
            pred = max(scores, key=scores.get)
            g = generate_bpe(m, tok, p, max_new_tokens=16)
            out.append({"set": label, "prompt": p, "expected": a, "predicted_by_score": pred,
                        "score_correct": pred == a, "free_generation": g.text,
                        "free_correct": g.text.strip() == a, "finish_reason": g.finish_reason})
    # first-token CE by color on TRAIN
    ce_by_color = {}
    for p, a in TRAIN:
        ce_by_color.setdefault(a, []).append(first_token_ce(m, tok, p, a, valid, lookup))
    ce_mean = {c: sum(v)/len(v) for c, v in ce_by_color.items()}
    sealed = [x for x in out if x["set"] == "sealed"]
    cf = [x for x in out if x["set"] == "independent_counterfactual"]
    # counterfactual follow rate: within each pair, answers must flip with color swap
    cf_follow = None
    if len(cf) >= 4:
        flipped = 0; pairs = 0
        for i in range(0, len(cf), 2):
            pair = cf[i:i+2]
            if len(pair) == 2 and pair[0]["expected"] != pair[1]["expected"]:
                pairs += 1
                if pair[0]["predicted_by_score"] != pair[1]["predicted_by_score"]: flipped += 1
        cf_follow = flipped / max(1, pairs)
    return {"rows": out, "sealed_score_accuracy": sum(x["score_correct"] for x in sealed)/max(1,len(sealed)),
            "sealed_free_accuracy": sum(x["free_correct"] for x in sealed)/max(1,len(sealed)),
            "cf_score_accuracy": sum(x["score_correct"] for x in cf)/max(1,len(cf)),
            "counterfactual_follow_rate": cf_follow,
            "answer_first_token_ce_by_color": ce_mean}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute-train", action="store_true", help="actually run optimizer.step")
    ap.add_argument("--authorized", action="store_true", help="explicit authorization flag")
    args = ap.parse_args()
    torch.set_num_threads(1)
    tok, errs, pre = preflight()
    if errs:
        print(json.dumps({"status": "BLOCKED_PREFLIGHT", "errors": errs, "preflight": pre}, ensure_ascii=False, indent=2))
        sys.exit(1)
    cf_rows, cf_sha = load_cf()
    valid = torch.tensor(sorted(effective_token_ids(tok)))
    lookup = torch.full((tok.vocab_size,), -1, dtype=torch.long); lookup[valid] = torch.arange(valid.numel())
    if not (args.execute_train and args.authorized):
        print(json.dumps({"status": "DRY_RUN_VALIDATION_ONLY", "optimizer_steps": 0,
                          "preflight": pre, "counterfactual_sha256": cf_sha,
                          "would_train": {"arms": 2, "steps_per_arm": MAX_STEPS, "total": MAX_TOTAL,
                                          "seed": SEED, "lr": LR, "batch_size": BS,
                                          "only_variable": "router_gradient_path_enabled"},
                          "message": "默认仅校验。真实训练须显式 --execute-train --authorized"}, ensure_ascii=False, indent=2))
        return
    if MAX_TOTAL > 512: raise RuntimeError("budget exceeds 512 optimizer steps")
    report = {"status": "COMPLETED_AUTHORIZED_TRAINING", "preflight": pre, "seed": SEED,
              "total_optimizer_steps": 0, "arms": {}, "stable_modified": False, "network_requests": 0}
    for enabled, name, cand in ((False, "baseline_frozen_router", "candidate_v16_baseline_frozen_router.pt"),
                                (True, "fixed_router", "candidate_v16_fixed_router.pt")):
        m, arm = train_arm(tok, enabled, name, valid, lookup)
        ev = evaluate(m, tok, valid, lookup, cf_rows)
        arm["evaluation"] = ev
        report["arms"][name] = arm
        report["total_optimizer_steps"] += arm["steps"]
        save_language_v2(V16 / cand, m, tok, steps=arm["steps"], seed=SEED, manifest_hash="v16-preregistered")
        arm["candidate_file"] = cand
    a, b = report["arms"]["baseline_frozen_router"], report["arms"]["fixed_router"]
    report["comparison"] = {
        "router_grad_enabled_only_difference": True,
        "sealed_accuracy": {"baseline": a["evaluation"]["sealed_score_accuracy"], "fixed": b["evaluation"]["sealed_score_accuracy"]},
        "counterfactual_follow": {"baseline": a["evaluation"]["counterfactual_follow_rate"], "fixed": b["evaluation"]["counterfactual_follow_rate"]},
        "answer_first_token_ce": {"baseline": a["evaluation"]["answer_first_token_ce_by_color"], "fixed": b["evaluation"]["answer_first_token_ce_by_color"]},
        "router_weight_delta_l1": {"baseline": a["router_weight_delta_l1"], "fixed": b["router_weight_delta_l1"]},
    }
    (V16 / "V16_TRAINING_REPORT.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "COMPLETED_AUTHORIZED_TRAINING", "total_optimizer_steps": report["total_optimizer_steps"],
                      "comparison": report["comparison"]}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
