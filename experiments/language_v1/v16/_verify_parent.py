"""V1.6 P0: verify parent checkpoint + tokenizer before any cloud prep.
ZERO optimizer steps. Read-only diagnosis."""
from __future__ import annotations
import hashlib, json, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
import torch
from nmind.language.tokenizer_v2 import ByteBPETokenizer
from nmind.language.neural_language_v2 import load_language_v2
from nmind.language.bpe_decoder import generate_bpe

torch.set_num_threads(1)
CKPT = ROOT / "experiments/language_v1/v15/candidate_v15_baseline.pt"
EXPECT_MODEL_SHA = "5681f656fbb35f4f4a033e16212897f2d7cc02eac6754b9d860fb980d6f6a611"
EXPECT_TOK_HASH = "26af116ebd13ff00"
EXPECT_PARAMS = 279363
# V1.5 ORIGINAL LEGACY corpus - verbatim from train_v15.py line 11
LEGACY = [("用户：你好\n助手：","你好！"),("用户：你叫什么？\n助手：","我叫N-MIND。"),("用户：谢谢\n助手：","不客气！"),("用户：小明有红色苹果\n助手：","小明的苹果是红色的。"),("用户：小红有绿色苹果\n助手：","小红的苹果是绿色的。"),("用户：小刚有蓝色苹果\n助手：","小刚的苹果是蓝色的。"),("用户：小丽有黄色苹果\n助手：","小丽的苹果是黄色的。"),("用户：我叫小明\n助手：","你好，小明！"),("用户：我叫什么？\n助手：","你叫小明。")]

r = {"checkpoint": str(CKPT.relative_to(ROOT))}
# 1. exists
r["file_exists"] = CKPT.exists()
if not r["file_exists"]:
    print(json.dumps(r, ensure_ascii=False, indent=2)); sys.exit(1)
# 2. sha256
h = hashlib.sha256(CKPT.read_bytes()).hexdigest()
r["sha256"] = h
r["sha256_match"] = (h == EXPECT_MODEL_SHA)
r["file_size_bytes"] = CKPT.stat().st_size
# 3. tokenizer rebuild from LEGACY
tok = ByteBPETokenizer.train([p + a for p, a in LEGACY], vocab_size=4096)
tok_hash = tok.vocab_hash()
r["tokenizer_hash"] = tok_hash
r["tokenizer_hash_match"] = (tok_hash == EXPECT_TOK_HASH)
# 4. strict load
load_err = None
try:
    model, meta = load_language_v2(CKPT, tok)
    r["strict_load"] = "OK"
except Exception as e:
    load_err = f"{type(e).__name__}: {e}"
    r["strict_load"] = "FAIL"
    r["strict_load_error"] = load_err
if load_err:
    print(json.dumps(r, ensure_ascii=False, indent=2)); sys.exit(1)
model.eval(); model.use_router_load_update = False
# 5. param count
n_params = model.n_params()
r["param_count"] = n_params
r["param_count_match"] = (n_params == EXPECT_PARAMS)
# 6. no query-attention params
state_keys = list(model.state_dict().keys())
qa_keys = [k for k in state_keys if "query_readout" in k]
r["query_attention_param_keys"] = qa_keys
r["no_query_attention_params"] = (len(qa_keys) == 0)
r["config_query_readout"] = model.v2_config.query_readout
# 7. NaN / Inf scan over all tensors
nonfinite = []
for k, v in model.state_dict().items():
    t = v.detach().cpu()
    if t.is_floating_point() and not torch.isfinite(t).all().item():
        nonfinite.append(k)
r["nonfinite_tensors"] = nonfinite
r["all_finite"] = (len(nonfinite) == 0)
# 8. inference with LEGACY tokenizer
infer_ok = True; infer_detail = {}
try:
    p = "用户：小明有红色苹果，小红有绿色苹果。\n用户：小红的苹果是什么颜色？\n助手："
    g = generate_bpe(model, tok, p, max_new_tokens=16)
    infer_detail = {"generated": g.text, "token_ids": g.token_ids,
                    "finish_reason": g.finish_reason,
                    "invalid_candidate_ids": g.invalid_candidate_ids}
    r["inference"] = infer_detail
except Exception as e:
    infer_ok = False; r["inference_error"] = f"{type(e).__name__}: {e}"
# invalid_candidate_ids are invalid BPE slots the decoder filtered out (expected);
# inference is OK iff we got non-empty valid UTF-8 text that finished at EOS without byte error.
r["inference_ok"] = (infer_ok
                     and bool(infer_detail.get("generated"))
                     and infer_detail.get("finish_reason") == "eos"
                     and infer_detail.get("byte_decode_error") is not True)
# checkpoint embedding hash sanity (as stored)
import torch as _t
blob = _t.load(CKPT, map_location="cpu", weights_only=True)
r["checkpoint_tokenizer_hash_field"] = blob.get("tokenizer_hash")
r["checkpoint_field_matches_rebuilt"] = (blob.get("tokenizer_hash") == tok_hash)
r["checkpoint_saved_meta"] = {k: blob.get(k) for k in ("steps","seed","manifest_hash","saved_at")}
# verdict
checks = ["file_exists","sha256_match","tokenizer_hash_match","param_count_match",
          "no_query_attention_params","all_finite","inference_ok","checkpoint_field_matches_rebuilt"]
r["all_checks_passed"] = all(r.get(c) is True for c in checks)
r["failed_checks"] = [c for c in checks if r.get(c) is not True]
out = ROOT / "experiments/language_v1/v16/_parent_verification.json"
out.write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(r, ensure_ascii=False, indent=2))
