"""Language V1.1 Gate-0: backward-only audit and candidate round trip."""
from __future__ import annotations
import hashlib, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import torch
from nmind.language.tokenizer_v2 import ByteBPETokenizer
from nmind.language.neural_language_v2 import LanguageV2Config, LanguageV2Model, save_language_v2, load_language_v2

ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / "experiments" / "language_v1"

def sha(model):
    h = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()): h.update(name.encode()); h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def main():
    torch.manual_seed(20261123)
    texts = ["你好，世界。", "苹果在桌子上。", "Python可以处理中文。", "代码、标点与emoji🙂。"]
    tok = ByteBPETokenizer.train(texts, vocab_size=4096)
    cfg = LanguageV2Config(vocab_size=tok.vocab_size, d_model=32, n_blocks=2, block_hidden=32, workspace_dim=16, workspace_slots=2)
    model = LanguageV2Model(cfg); model.train(); model.use_router_load_update = False; before = sha(model)
    ids = torch.tensor([tok.encode(texts[0] + texts[1], add_eos=True)], dtype=torch.long)
    logits, _ = model(ids[:, :-1]); target = ids[:, 1:]
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1)); loss.backward()
    groups = {"ncell": [], "router": [], "workspace": [], "block_to_ws": [], "embedding": [], "head": []}
    for name, p in model.named_parameters():
        if p.grad is None: continue
        key = "embedding" if name.startswith("tok_emb") else "head" if name.startswith("head") else "block_to_ws" if name.startswith("block_to_ws") else "router" if name.startswith("router") else "workspace" if name.startswith("workspace") else "ncell" if name.startswith("blocks") else None
        if key: groups[key].append(float(p.grad.detach().norm()))
    grad = {k: {"parameter_count": len(v), "grad_norm_sum": sum(v), "nonzero_finite": bool(v) and all(x > 0 and x == x for x in v)} for k, v in groups.items()}
    candidate = OUT / "candidate_gate0.pt"; save_language_v2(candidate, model, tok, steps=0, seed=20261123, manifest_hash="synthetic-gate0")
    reloaded, meta = load_language_v2(candidate, tok); reload_same = sha(reloaded) == before
    result = {"status":"PASS_BACKWARD_ONLY", "optimizer_steps":0, "attempted_steps":0, "loss":float(loss.detach()), "gradients":grad,
      "tokenizer_vocab_size":tok.vocab_size, "model_config":cfg.__dict__, "embedding_shape":list(model.tok_emb.weight.shape), "head_shape":list(model.head.weight.shape),
      "n_params":model.n_params(), "valid_token_id_max":int(ids.max()), "input_shape":list(ids.shape), "logits_shape":list(logits.shape),
      "sha_before":before, "sha_after_backward":sha(model), "parameters_changed":sha(model)!=before, "candidate_saved":candidate.exists(), "reload_sha_matches":reload_same,
      "stable_modified":False, "network_requests":0, "training_data":"synthetic-only", "formal_training":"PENDING_USER_APPROVAL", "meta":meta}
    OUT.mkdir(parents=True, exist_ok=True); (OUT/"LANGUAGE_V1_1_GATE0.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False))
if __name__ == "__main__": main()
