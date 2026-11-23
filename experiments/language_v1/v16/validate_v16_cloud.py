"""N-MIND V1.6 cloud VALIDATION (default GitHub Actions entry). NO optimizer.step.
Two-tier checks:
  - structural: no weights needed (tokenizer, prereg, script importability, data hashes, seed)
  - weight-dependent: parent SHA, strict load, params, A/B zero-step gradients
If parent weight is absent (expected in public repo), weight checks are SKIPPED and status
becomes VALIDATION_STRUCTURAL_ONLY / TRAINING_BLOCKED_NO_WEIGHTS -- never substitute random weights.
"""
from __future__ import annotations
import hashlib, json, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
import torch
from nmind.language.tokenizer_v2 import ByteBPETokenizer
from nmind.language.neural_language_v2 import load_language_v2
from nmind.language.bpe_decoder import effective_token_ids

torch.set_num_threads(1)
CKPT = ROOT / "experiments/language_v1/v15/candidate_v15_baseline.pt"
V16 = ROOT / "experiments/language_v1/v16"
PARENT_SHA = "5681f656fbb35f4f4a033e16212897f2d7cc02eac6754b9d860fb980d6f6a611"
TOK_HASH = "26af116ebd13ff00"
SEED = 20261123
LEGACY = [("用户：你好\n助手：","你好！"),("用户：你叫什么？\n助手：","我叫N-MIND。"),("用户：谢谢\n助手：","不客气！"),("用户：小明有红色苹果\n助手：","小明的苹果是红色的。"),("用户：小红有绿色苹果\n助手：","小红的苹果是绿色的。"),("用户：小刚有蓝色苹果\n助手：","小刚的苹果是蓝色的。"),("用户：小丽有黄色苹果\n助手：","小丽的苹果是黄色的。"),("用户：我叫小明\n助手：","你好，小明！"),("用户：我叫什么？\n助手：","你叫小明。")]
TRAIN = [("用户：小明有红色苹果，小红有绿色苹果。\n用户：小明的苹果是什么颜色？\n助手：","红色"),("用户：小明有红色苹果，小红有绿色苹果。\n用户：小红的苹果是什么颜色？\n助手：","绿色"),("用户：小明有绿色苹果，小红有红色苹果。\n用户：小明的苹果是什么颜色？\n助手：","绿色"),("用户：小明有绿色苹果，小红有红色苹果。\n用户：小红的苹果是什么颜色？\n助手：","红色")]

checks = {}; tier = {}
def ck(name, cond, detail="", t="structural"):
    checks[name] = {"pass": bool(cond), "detail": detail}; tier[name] = t
def skip(name, detail):
    checks[name] = {"pass": None, "status": "SKIPPED_NO_WEIGHT", "detail": detail}; tier[name] = "weight"

# ---- structural tier (no weights required) ----
ck("torch_available", torch.__version__ is not None, torch.__version__)
ck("cpu_device", str(torch.device("cpu")) == "cpu")
tok = ByteBPETokenizer.train([p + a for p, a in LEGACY], vocab_size=4096)
ck("tokenizer_hash_match", tok.vocab_hash() == TOK_HASH, tok.vocab_hash())
plan = json.loads((V16 / "V16_NEXT_TRAINING_PLAN.json").read_text(encoding="utf-8"))
ck("prereg_same_seed", plan.get("shared_random_seed") == SEED and all(a.get("seed") == SEED for a in plan["arms"]))
ck("prereg_single_variable", plan.get("single_key_variable") == "router_gradient_path_enabled")
try:
    cf = json.loads((V16 / "V16_COUNTERFACTUAL.json").read_text(encoding="utf-8"))
    canon = hashlib.sha256(json.dumps([[x["prompt"],x["answer"]] for x in cf["rows"]],ensure_ascii=False,sort_keys=True).encode()).hexdigest()
    ck("counterfactual_hash_match", canon == cf["sha256"], canon)
except Exception as e:
    ck("counterfactual_hash_match", False, str(e))
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("train_v16_ab", V16 / "train_v16_ab.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    ck("train_script_importable_no_autostart", hasattr(mod, "main"))
except Exception as e:
    ck("train_script_importable_no_autostart", False, f"{type(e).__name__}: {e}")

# ---- weight-dependent tier ----
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
if not CKPT.exists():
    for n in ("parent_sha256_match","strict_load","param_count_279363","no_query_readout",
              "all_finite_weights","arm_A_router_no_grad","arm_B_router_has_grad"):
        skip(n, "父权重未入库(公开仓库禁止上传); 需先配置安全权重传输机制")
    weights_available = False
else:
    weights_available = True
    ck("parent_sha256_match", sha(CKPT) == PARENT_SHA, sha(CKPT), "weight")
    model = None
    try:
        model, meta = load_language_v2(CKPT, tok)
        ck("strict_load", True, t="weight")
        ck("param_count_279363", model.n_params() == 279363, model.n_params(), "weight")
        ck("no_query_readout", not model.v2_config.query_readout, t="weight")
        bad = [k for k,v in model.state_dict().items() if v.is_floating_point() and not torch.isfinite(v).all().item()]
        ck("all_finite_weights", not bad, bad, "weight")
    except Exception as e:
        ck("strict_load", False, f"{type(e).__name__}: {e}", "weight")
    if model is not None:
        valid = torch.tensor(sorted(effective_token_ids(tok)))
        lookup = torch.full((tok.vocab_size,), -1, dtype=torch.long); lookup[valid] = torch.arange(valid.numel())
        def grad_sum(enabled):
            torch.manual_seed(SEED); m,_ = load_language_v2(CKPT, tok)
            m.router_gradient_path_enabled = enabled; m.train(); m.use_router_load_update = False
            batch=[TRAIN[0],TRAIN[1]]; seq=[tok.encode(p+a,add_eos=True) for p,a in batch]
            n=max(map(len,seq)); x=torch.zeros((2,n),dtype=torch.long); mask=torch.zeros((2,n-1),dtype=torch.bool)
            for i,(p,a) in enumerate(batch):
                x[i,:len(seq[i])]=torch.tensor(seq[i]); start=len(tok.encode(p))-1; mask[i,start:len(seq[i])-1]=True
            logits,_=m(x[:,:-1]); z=logits.index_select(-1,valid); target=x[:,1:]
            loss=torch.nn.functional.cross_entropy(z[mask],lookup[target[mask]])
            m.zero_grad(set_to_none=True); loss.backward()
            return sum(float(p.grad.abs().sum()) for k,p in m.named_parameters() if k.startswith("router.score_mlp") and p.grad is not None)
        ga,gb = grad_sum(False), grad_sum(True)
        ck("arm_A_router_no_grad", ga==0.0, ga, "weight")
        ck("arm_B_router_has_grad", gb>0.0, gb, "weight")

struct = [k for k in checks if tier[k]=="structural"]
weight = [k for k in checks if tier[k]=="weight"]
struct_ok = all(checks[k]["pass"] for k in struct)
weight_ok = weights_available and all(checks[k]["pass"] for k in weight)
if weights_available and struct_ok and weight_ok:
    status = "VALIDATION_PASSED"
elif not weights_available and struct_ok:
    status = "VALIDATION_STRUCTURAL_ONLY_TRAINING_BLOCKED_NO_WEIGHTS"
else:
    status = "VALIDATION_FAILED"
result = {"status": status, "tiers": {"structural": {k:checks[k] for k in struct},
                                       "weight_dependent": {k:checks[k] for k in weight}},
          "weights_available": weights_available,
          "passed_structural": sum(1 for k in struct if checks[k]["pass"]), "total_structural": len(struct),
          "optimizer_steps": 0,
          "training_gate": "BLOCKED: 父权重 sha256=" + PARENT_SHA + " 未入库; 禁止用随机模型替代, 需安全传输后再校验" if not weights_available else (
              "READY_PENDING_AUTHORIZATION" if status=="VALIDATION_PASSED" else "BLOCKED: 权重校验未通过"),
          "note": "校验脚本永不执行 optimizer.step; 真实训练须单独显式授权"}
(V16 / "V16_CLOUD_VALIDATION.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"status": status, "weights_available": weights_available,
                  "passed_structural": result["passed_structural"], "total_structural": len(struct),
                  "training_gate": result["training_gate"]}, ensure_ascii=False, indent=2))
# exit 0 for structural-only (expected in public CI), non-zero only for genuine failures
sys.exit(0 if status != "VALIDATION_FAILED" else 1)
