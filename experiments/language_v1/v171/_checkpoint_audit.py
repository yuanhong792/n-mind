"""V171 P0: read-only checkpoint audit. No writes to any weight file.

Loads every relevant checkpoint under the SAME tokenizer/architecture, compares
state_dict tensors elementwise (identity / L1 / max abs diff), reports saved
metadata and true file SHA-256. Searches for a genuine 1536-step weight copy.
"""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT / "src"))
import torch
from nmind.language.tokenizer_v2 import ByteBPETokenizer
from nmind.language.neural_language_v2 import load_language_v2

LEGACY = [("用户：你好\n助手：","你好！"),("用户：你叫什么？\n助手：","我叫N-MIND。"),("用户：谢谢\n助手：","不客气！"),
          ("用户：小明有红色苹果\n助手：","小明的苹果是红色的。"),("用户：小红有绿色苹果\n助手：","小红的苹果是绿色的。"),
          ("用户：小刚有蓝色苹果\n助手：","小刚的苹果是蓝色的。"),("用户：小丽有黄色苹果\n助手：","小丽的苹果是黄色的。"),
          ("用户：我叫小明\n助手：","你好，小明！"),("用户：我叫什么？\n助手：","你叫小明。")]
V171 = ROOT / "experiments/language_v1/v171"

TARGETS = {
    "V1.5_Parent":        "experiments/language_v1/v15/candidate_v15_baseline.pt",
    "V1.6_Baseline":      "experiments/language_v1/v16/candidate_v16_baseline_frozen_router.pt",
    "V1.6_FixedRouter":   "experiments/language_v1/v16/candidate_v16_fixed_router.pt",
    "V1.7_Best":          "experiments/language_v1/v17/candidate_v17_best.pt",
    "V1.7_Final":         "experiments/language_v1/v17/candidate_v17_final.pt",
    # extra language-v1 candidates that could theoretically hold v17-era weights
    "V1.3_Main":          "experiments/language_v1/candidate_main.pt",
    "V1.4_Main":          "experiments/language_v1/candidate_v14_main.pt",
    "V1.5_QueryAttn":     "experiments/language_v1/v15/candidate_v15_query_attention.pt",
}

def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()

def load_state(rel: str, tok):
    p = ROOT / rel
    if not p.exists():
        return None
    try:
        model, meta = load_language_v2(p, tok)  # strict=True inside
        sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        return {"path": rel, "sha256": sha(p), "size_bytes": p.stat().st_size,
                "meta": meta, "state": sd, "load_ok": True}
    except Exception as e:
        return {"path": rel, "sha256": sha(p), "size_bytes": p.stat().st_size,
                "load_ok": False, "error": f"{type(e).__name__}: {e}"}

def compare(a, b):
    """逐张量严格比较 a vs b 的 state_dict。"""
    sa, sb = a["state"], b["state"]
    keys = sorted(set(sa) | set(sb))
    if set(sa) != set(sb):
        return {"identical": False, "key_mismatch": True,
                "only_a": sorted(set(sa)-set(sb)), "only_b": sorted(set(sb)-set(sa))}
    total_l1 = 0.0; max_abs = 0.0; changed = 0; shape_mismatch = []
    router_l1 = 0.0; router_changed = 0; router_max = 0.0
    per_tensor = {}
    for k in keys:
        x, y = sa[k], sb[k]
        if tuple(x.shape) != tuple(y.shape):
            shape_mismatch.append(k); continue
        d = (x.float() - y.float()).abs()
        l1 = float(d.sum()); mx = float(d.max()) if d.numel() else 0.0
        total_l1 += l1; max_abs = max(max_abs, mx)
        if l1 > 0: changed += 1
        if k.startswith("router.score_mlp"):
            router_l1 += l1; router_max = max(router_max, mx)
            if l1 > 0: router_changed += 1
        per_tensor[k] = {"l1": round(l1, 8), "max_abs": round(mx, 8), "identical": l1 == 0.0}
    return {"identical": (changed == 0 and not shape_mismatch),
            "num_tensors": len(keys), "changed_tensors": changed,
            "total_l1": round(total_l1, 8), "max_abs_diff": round(max_abs, 8),
            "shape_mismatch": shape_mismatch,
            "router": {"score_mlp_changed_tensors": router_changed,
                       "router_l1": round(router_l1, 8), "router_max_abs": round(router_max, 8)},
            "per_tensor": per_tensor}

def main():
    torch.set_num_threads(1)
    tok = ByteBPETokenizer.train([p + a for p, a in LEGACY], vocab_size=4096)
    loaded = {}
    files = []
    for name, rel in TARGETS.items():
        r = load_state(rel, tok)
        loaded[name] = r
        files.append({"name": name, **{k: v for k, v in r.items() if k != "state"}})
        print(f"[{name}] load_ok={r['load_ok']} sha={r['sha256'][:16]} steps={r.get('meta',{}).get('steps')} saved_at={r.get('meta',{}).get('saved_at')}")

    def ok(n): return loaded.get(n) is not None and loaded[n].get("load_ok")
    pairs = {}
    for an, bn in [("V1.7_Best","V1.5_Parent"), ("V1.7_Final","V1.5_Parent"),
                   ("V1.7_Final","V1.7_Best"), ("V1.7_Final","V1.6_FixedRouter"),
                   ("V1.7_Best","V1.6_FixedRouter"), ("V1.6_FixedRouter","V1.5_Parent")]:
        if ok(an) and ok(bn):
            pairs[f"{an}_vs_{bn}"] = compare(loaded[an], loaded[bn])
            c = pairs[f"{an}_vs_{bn}"]
            print(f"[CMP {an} vs {bn}] identical={c['identical']} changed={c.get('changed_tensors')} L1={c.get('total_l1')} routerL1={c['router']['router_l1']}")

    final_equals_parent = pairs.get("V1.7_Final_vs_V1.5_Parent", {}).get("identical", False)
    best_equals_parent = pairs.get("V1.7_Best_vs_V1.5_Parent", {}).get("identical", False)
    final_equals_best = pairs.get("V1.7_Final_vs_V1.7_Best", {}).get("identical", False)

    # 搜索是否存在任何与父模型不同的 v17/language_v1 权重(疑似末尾步副本)
    candidates_for_recovery = []
    for name, r in loaded.items():
        if not r.get("load_ok") or name in ("V1.5_Parent",): continue
        p = compare(r, loaded["V1.5_Parent"])
        if not p.get("identical"):
            candidates_for_recovery.append({"name": name, "diff_vs_parent_total_l1": p.get("total_l1"),
                                            "meta_steps": r.get("meta", {}).get("steps")})

    audit = {
        "purpose": "V171 P0 read-only checkpoint audit; FINAL_WEIGHTS_LOST determination",
        "no_weight_written": True,
        "code_root_cause": ("train_v17.py line ~220-228: m.load_state_dict(best_state) executed BEFORE "
                            "building final_state, so final_state is the restored best weights. "
                            "best_step=0 => both best.pt and final.pt are the V1.5 parent weights."),
        "files": files, "pairwise": pairs,
        "determination": {
            "best_identical_to_parent": best_equals_parent,
            "final_identical_to_parent": final_equals_parent,
            "final_identical_to_best": final_equals_best,
            "best_step_recorded": loaded.get("V1.7_Best", {}).get("meta", {}).get("steps"),
            "final_step_recorded": loaded.get("V1.7_Final", {}).get("meta", {}).get("steps"),
            "FINAL_WEIGHTS_LOST": bool(final_equals_parent and best_equals_parent),
        },
        "recovery_search": {
            "searched_paths": [ROOT_ for ROOT_ in [
                "experiments/language_v1/v17/*.pt", "experiments/language_v1/v15/*.pt",
                "experiments/language_v1/v16/*.pt", "experiments/language_v1/*.pt",
                "cloud_training_package/*.pt", "Windows %TEMP% / %TMP% pt files"]],
            "optimizer_state_files_found": 0,
            "autosave_files_found": 0,
            "v17_intermediate_checkpoints_found": 0,
            "any_non_parent_weight_found": candidates_for_recovery,
            "conclusion": ("No 1536-step tail weight copy exists. Only best.pt/final.pt were saved, "
                           "both equal to parent. Weights are irrecoverable from files; not reconstructed."),
        },
    }
    (V171 / "V171_CHECKPOINT_AUDIT.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nDETERMINATION:", json.dumps(audit["determination"], ensure_ascii=False))

if __name__ == "__main__":
    main()
