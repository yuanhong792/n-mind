"""V1.6 架构层诊断：Router 梯度是否为0 + h_mean 是否对查询敏感。冻结参数，不更新权重。"""
from __future__ import annotations
import sys, json, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
import torch
from nmind.language.tokenizer_v2 import ByteBPETokenizer
from nmind.language.bpe_decoder import effective_token_ids
from nmind.language.neural_language_v2 import load_language_v2

ROOT=Path(__file__).resolve().parents[3]; SRC=ROOT/"experiments/language_v1"; V15=SRC/"v15"; OUT=SRC/"v16"
LEGACY=[("用户：你好\n助手：","你好！"),("用户：你叫什么？\n助手：","我叫N-MIND。"),("用户：谢谢\n助手：","不客气！"),
("用户：小明有红色苹果\n助手：","小明的苹果是红色的。"),("用户：小红有绿色苹果\n助手：","小红的苹果是绿色的。"),
("用户：小刚有蓝色苹果\n助手：","小刚的苹果是蓝色的。"),("用户：小丽有黄色苹果\n助手：","小丽的苹果是黄色的。"),
("用户：我叫小明\n助手：","你好，小明！"),("用户：我叫什么？\n助手：","你叫小明。")]
TRAIN=[("用户：小明有红色苹果，小红有绿色苹果。\n用户：小明的苹果是什么颜色？\n助手：","红色"),
("用户：小明有红色苹果，小红有绿色苹果。\n用户：小红的苹果是什么颜色？\n助手：","绿色"),
("用户：小明有绿色苹果，小红有红色苹果。\n用户：小明的苹果是什么颜色？\n助手：","绿色"),
("用户：小明有绿色苹果，小红有红色苹果。\n用户：小红的苹果是什么颜色？\n助手：","红色")]
def main():
    torch.set_num_threads(1)
    tok=ByteBPETokenizer.train([p+a for p,a in LEGACY],vocab_size=4096)
    valid=torch.tensor(effective_token_ids(tok))
    lookup=torch.full((tok.vocab_size,),-1,dtype=torch.long); lookup[valid]=torch.arange(valid.numel())
    model,meta=load_language_v2(V15/"candidate_v15_baseline.pt",tok)

    # Router/各层梯度：lr=0 优化器只反传不更新
    model.train(); model.use_router_load_update=False
    p,a=TRAIN[0]; seq=tok.encode(p+a,add_eos=True); start=len(tok.encode(p))-1; n=len(seq)
    x=torch.zeros((1,n),dtype=torch.long); x[0,:n]=torch.tensor(seq)
    logits,_=model(x[:,:-1]); z=logits.index_select(-1,valid); target=x[:,1:]
    mask=torch.zeros((1,n-1),dtype=torch.bool); mask[0,start:n-1]=True
    loss=torch.nn.functional.cross_entropy(z[mask],lookup[target[mask]])
    model.zero_grad(); loss.backward()
    grad={}
    for name,param in model.named_parameters():
        grad[name]="NO_GRAD" if param.grad is None else round(float(param.grad.abs().sum()),8)
    model.eval()

    q_ming="用户：小明有红色苹果，小红有绿色苹果。\n用户：小明的苹果是什么颜色？\n助手："
    q_hong="用户：小明有红色苹果，小红有绿色苹果。\n用户：小红的苹果是什么颜色？\n助手："
    def feat(prompt):
        ids=tok.encode(prompt)
        with torch.inference_mode():
            logits,st=model(torch.tensor([ids]))
        return st.h.mean(dim=1).squeeze(0), logits[0,-1]
    fm,lm=feat(q_ming); fh,lh=feat(q_hong)
    cos_h=float(torch.nn.functional.cosine_similarity(fm.unsqueeze(0),fh.unsqueeze(0)))
    cos_logit=float(torch.nn.functional.cosine_similarity(lm.unsqueeze(0),lh.unsqueeze(0)))
    h_l2=float((fm-fh).norm()); logit_l2=float((lm-lh).norm())

    router_grads={k:v for k,v in grad.items() if 'router' in k.lower()}
    payload={
        "loss_used":round(float(loss),6),
        "router_grad_abs_sum":router_grads,
        "router_has_zero_grad": all(v==0 or v=="NO_GRAD" for v in router_grads.values()),
        "grad_abs_sum_all_layers":grad,
        "h_mean_cosine_ming_vs_hong":round(cos_h,6),
        "h_mean_l2_diff":round(h_l2,6),
        "final_logit_cosine":round(cos_logit,6),
        "final_logit_l2_diff":round(logit_l2,6),
        "note":"router梯度=0表示路由不接收语言监督; h/logit两查询余弦≈1、L2≈0表示查询实体完全不影响表示与读出"
    }
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"_diag_router.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(payload,ensure_ascii=False,indent=2))

if __name__=="__main__": main()
