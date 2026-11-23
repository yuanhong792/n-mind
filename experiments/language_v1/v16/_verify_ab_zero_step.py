"""V1.6 A/B zero-step validation. NO optimizer.step anywhere.
Verifies: identical init, identical forward, A=no router grad, B=nonzero router grad,
other modules receive grads in both arms."""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
import torch
from nmind.language.tokenizer_v2 import ByteBPETokenizer
from nmind.language.neural_language_v2 import load_language_v2

torch.set_num_threads(1)
LEGACY = [("用户：你好\n助手：","你好！"),("用户：你叫什么？\n助手：","我叫N-MIND。"),("用户：谢谢\n助手：","不客气！"),("用户：小明有红色苹果\n助手：","小明的苹果是红色的。"),("用户：小红有绿色苹果\n助手：","小红的苹果是绿色的。"),("用户：小刚有蓝色苹果\n助手：","小刚的苹果是蓝色的。"),("用户：小丽有黄色苹果\n助手：","小丽的苹果是黄色的。"),("用户：我叫小明\n助手：","你好，小明！"),("用户：我叫什么？\n助手：","你叫小明。")]
TRAIN = [("用户：小明有红色苹果，小红有绿色苹果。\n用户：小明的苹果是什么颜色？\n助手：","红色"),("用户：小明有红色苹果，小红有绿色苹果。\n用户：小红的苹果是什么颜色？\n助手：","绿色"),("用户：小明有绿色苹果，小红有红色苹果。\n用户：小明的苹果是什么颜色？\n助手：","绿色"),("用户：小明有绿色苹果，小红有红色苹果。\n用户：小红的苹果是什么颜色？\n助手：","红色")]
CKPT = ROOT / "experiments/language_v1/v15/candidate_v15_baseline.pt"
SEED = 20261123  # SAME seed for both arms (fixed from V1.6 prereg)
tok = ByteBPETokenizer.train([p + a for p, a in LEGACY], vocab_size=4096)
valid = torch.tensor(sorted(__import__("nmind.language.bpe_decoder", fromlist=["effective_token_ids"]).effective_token_ids(tok)))
lookup = torch.full((tok.vocab_size,), -1, dtype=torch.long); lookup[valid] = torch.arange(valid.numel())

def build(enabled: bool):
    torch.manual_seed(SEED)
    parent, _ = load_language_v2(CKPT, tok)
    m = parent
    m.router_gradient_path_enabled = enabled
    m.train(); m.use_router_load_update = False
    return m

def init_snapshot(m):
    return {k: v.detach().clone() for k, v in m.state_dict().items()}

A = build(False); B = build(True)
# 1. identical init
sa, sb = init_snapshot(A), init_snapshot(B)
init_identical = all(torch.equal(sa[k], sb[k]) for k in sa)
# 2. identical forward (same batch, same seed)
batch = [TRAIN[0], TRAIN[1]]
seq = [tok.encode(p + a, add_eos=True) for p, a in batch]
n = max(map(len, seq))
x = torch.zeros((2, n), dtype=torch.long)
mask = torch.zeros((2, n - 1), dtype=torch.bool)
for i, (p, a) in enumerate(batch):
    x[i, :len(seq[i])] = torch.tensor(seq[i])
    start = len(tok.encode(p)) - 1
    mask[i, start:len(seq[i]) - 1] = True

def fwd(m):
    logits, _ = m(x[:, :-1])
    z = logits.index_select(-1, valid)
    target = x[:, 1:]
    loss = torch.nn.functional.cross_entropy(z[mask], lookup[target[mask]])
    return logits, loss

la, lossa = fwd(A); lb, lossb = fwd(B)
forward_identical = torch.equal(la, lb) and torch.equal(lossa, lossb)

def grad_audit(m, loss):
    m.zero_grad(set_to_none=True)
    loss.backward()  # ONLY backward, no optimizer.step
    g = {}
    for name, p in m.named_parameters():
        if p.grad is None:
            g[name] = 0.0
        else:
            g[name] = float(p.grad.detach().abs().sum())
    return g

ga = grad_audit(A, lossa)
gb = grad_audit(B, lossb)
router_keys = [k for k in gb if k.startswith("router.score_mlp")]
A_router_sum = sum(ga[k] for k in router_keys)
B_router_sum = sum(gb[k] for k in router_keys)
# known pre-registered UNWIRED component from V1.6 root-cause report (secondary finding):
# workspace.utility_mlp computes utility but write() result is not consumed by forward.
# It has no gradient in BOTH arms and is unrelated to the A/B variable -> disclose,
# exclude from the "active modules must receive gradient" check.
KNOWN_UNWIRED = ("workspace.utility_mlp.weight", "workspace.utility_mlp.bias")
# active trainable modules: everything except router.score_mlp and known-unwired
other_keys = [k for k in gb if not k.startswith("router.score_mlp") and k not in KNOWN_UNWIRED]
A_other_all_grad = all(ga[k] > 0 for k in other_keys)
B_other_all_grad = all(gb[k] > 0 for k in other_keys)
# which active modules have zero grad (report honestly)
A_other_zero = [k for k in other_keys if ga[k] == 0.0]
B_other_zero = [k for k in other_keys if gb[k] == 0.0]
# disclose known-unwired grad state explicitly for both arms
unwired_grad_A = {k: ga[k] for k in KNOWN_UNWIRED}
unwired_grad_B = {k: gb[k] for k in KNOWN_UNWIRED}

r = {
    "status": "ZERO_STEP_VALIDATION_COMPLETED",
    "optimizer_steps": 0,
    "seed_both_arms": SEED,
    "parent_checkpoint_sha256": "5681f656fbb35f4f4a033e16212897f2d7cc02eac6754b9d860fb980d6f6a611",
    "conditions": {
        "1_initial_weights_identical": init_identical,
        "2_forward_outputs_identical": forward_identical,
        "3_arm_A_router_no_task_gradient": (A_router_sum == 0.0),
        "4_arm_B_router_nonzero_task_gradient": (B_router_sum > 0.0),
        "5_other_modules_receive_gradient": (A_other_all_grad and B_other_all_grad),
    },
    "router_grad_abs_sum": {"arm_A_baseline_frozen_router": A_router_sum, "arm_B_fixed_router": B_router_sum},
    "router_score_mlp_grads": {"arm_A": {k: ga[k] for k in router_keys}, "arm_B": {k: gb[k] for k in router_keys}},
    "loss": {"arm_A": float(lossa.detach()), "arm_B": float(lossb.detach())},
    "active_modules_zero_grad_A": A_other_zero,
    "active_modules_zero_grad_B": B_other_zero,
    "known_unwired_component": {
        "keys": list(KNOWN_UNWIRED),
        "grad_A": unwired_grad_A,
        "grad_B": unwired_grad_B,
        "reason": "V1.6 根因报告 secondary finding: workspace.utility_mlp 计算 utility 但 write() 未被 forward 消费；两组均无梯度，与 A/B 变量无关，不计入主动模块梯度判定，据实披露",
    },
    "note": "条件5的主动模块 = 除 router.score_mlp 与已知未接线 utility_mlp 外的所有含参模块；零梯度项据实列出，不掩盖",
}
checks = list(r["conditions"].values())
r["all_five_conditions_met"] = all(checks)
out = ROOT / "experiments/language_v1/v16/V16_AB_ZERO_STEP.json"
out.write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({
    "all_five_conditions_met": r["all_five_conditions_met"],
    "conditions": r["conditions"],
    "router_grad_A": A_router_sum, "router_grad_B": B_router_sum,
    "other_zero_grad_A": A_other_zero, "other_zero_grad_B": B_other_zero,
}, ensure_ascii=False, indent=2))
