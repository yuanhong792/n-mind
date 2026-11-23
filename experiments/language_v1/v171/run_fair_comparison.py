"""V171 P3: fair comparison of all loadable models under the unified protocol."""
from __future__ import annotations
import sys, json, time, hashlib
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments/language_v1/v171"))
import torch
from nmind.language.neural_language_v2 import load_language_v2
import v171_protocol as P

V17 = ROOT / "experiments/language_v1/v17"
V171 = ROOT / "experiments/language_v1/v171"

MODELS = [
    ("V1.5_Parent", "experiments/language_v1/v15/candidate_v15_baseline.pt"),
    ("V1.7_Best",   "experiments/language_v1/v17/candidate_v17_best.pt"),
    ("V1.7_Final",  "experiments/language_v1/v17/candidate_v17_final.pt"),
]

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    torch.set_num_threads(1)
    tok = P.build_tokenizer()
    valid, lookup = P.build_vocab(tok)
    enc, fd = P.discriminative_positions(tok)
    rows = P.load_rows("sealed")
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    if limit:
        rows = rows[:limit]
    print(f"sealed rows used: {len(rows)}", flush=True)
    results = {}
    for name, rel in MODELS:
        path = ROOT / rel
        t0 = time.perf_counter()
        m, meta = load_language_v2(path, tok)
        ev = P.evaluate_model(m, tok, rows, valid=valid, lookup=lookup, fd=fd, label=name)
        cf = P.counterfactual_paired(m, tok, rows, valid=valid, lookup=lookup, fd=fd)
        bl = P.shortcut_baselines(rows)
        results[name] = {
            "path": rel, "sha256": sha(path), "meta": meta,
            "evaluation": ev, "counterfactual": cf,
            "baselines": bl, "wall_seconds": round(time.perf_counter() - t0, 1),
            "weight_status": ("valid_parent" if "Parent" in name else
                              "LOST_weights_equal_parent" if "Final" in name else
                              "valid_but_equals_parent"),
        }
        print(f"[{name}] done in {results[name]['wall_seconds']}s "
              f"free={ev['overall'].get('free_acc')} "
              f"imm_eos={ev['overall'].get('immediate_eos_rate')}", flush=True)
    report = {
        "protocol": "V171 unified; see V171_EVALUATION_PROTOCOL.md",
        "scoring_modes": ["DISCRIMINATIVE_TOKEN_PROXY", "FULL_ANSWER_CONDITIONAL"],
        "sealed_rows": len(rows), "sealed_sha256": sha(V17 / "v17_sealed.jsonl"),
        "tokenizer_hash": tok.vocab_hash(), "max_new_tokens": P.MAX_NEW_TOKENS,
        "candidate_set": {k: len(v) for k, v in P.CANDIDATES_BY_TASK.items()},
        "independence_warning": ("V1.7 sealed set was used during development analysis; "
                                 "blind-test independence may be affected. Do not treat "
                                 "gains here as fresh blind results."),
        "FINAL_WEIGHTS_LOST": True,
        "models": results,
    }
    (V171 / "V171_FAIR_COMPARISON.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved V171_FAIR_COMPARISON.json", flush=True)

if __name__ == "__main__":
    main()
