"""V1.6 P1-P4 深挖：数据捷径审计 + 首字loss收敛性 + 探针(冻结表示线性可解码性)。"""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
import torch
from nmind.language.tokenizer_v2 import ByteBPETokenizer, EOS
from nmind.language.bpe_decoder import effective_token_ids
from nmind.language.neural_language_v2 import load_language_v2

ROOT = Path(__file__).resolve().parents[3]; SRC = ROOT / "experiments/language_v1"; V15 = SRC/"v15"; OUT = SRC/"v16"
LEGACY = [
    ("用户：你好\n助手：", "你好！"), ("用户：你叫什么？\n助手：", "我叫N-MIND。"),
    ("用户：谢谢\n助手：", "不客气！"), ("用户：小明有红色苹果\n助手：", "小明的苹果是红色的。"),
    ("用户：小红有绿色苹果\n助手：", "小红的苹果是绿色的。"), ("用户：小刚有蓝色苹果\n助手：", "小刚的苹果是蓝色的。"),
    ("用户：小丽有黄色苹果\n助手：", "小丽的苹果是黄色的。"), ("用户：我叫小明\n助手：", "你好，小明！"),
    ("用户：我叫什么？\n助手：", "你叫小明。"),
]
TRAIN = [
    ("用户：小明有红色苹果，小红有绿色苹果。\n用户：小明的苹果是什么颜色？\n助手：", "红色"),
    ("用户：小明有红色苹果，小红有绿色苹果。\n用户：小红的苹果是什么颜色？\n助手：", "绿色"),
    ("用户：小明有绿色苹果，小红有红色苹果。\n用户：小明的苹果是什么颜色？\n助手：", "绿色"),
    ("用户：小明有绿色苹果，小红有红色苹果。\n用户：小红的苹果是什么颜色？\n助手：", "红色"),
]

def main():
    torch.set_num_threads(1)
    tok = ByteBPETokenizer.train([p+a for p,a in LEGACY], vocab_size=4096)
    valid = torch.tensor(effective_token_ids(tok))
    lookup = torch.full((tok.vocab_size,),-1,dtype=torch.long); lookup[valid]=torch.arange(valid.numel())

    # ===== 数据捷径审计 =====
    from collections import Counter
    ans_freq = Counter(a for p,a in TRAIN)
    # 每个人对应的颜色（联合事实里）
    fact_color = {"小明":Counter(), "小红":Counter()}
    for p,a in TRAIN:
        for person,color in [("小明","红色"),("小红","绿色")]:
            fact_color[person][color]+=1
        for person,color in [("小明","绿色"),("小红","红色")]:
            fact_color[person][color]+=1
    # 训练 walker：step%4 决定 (p,a)。检查"最近属性"策略能否命中
    # TRAIN[i] 的事实里，最后提到的颜色；答案是否等于最近颜色
    recent_hits=0; fixed_red=0; fixed_green=0
    for p,a in TRAIN:
        if a=="红色": fixed_red+=1
        if a=="绿色": fixed_green+=1
    # 分析: 每条 prompt 首个事实颜色 vs 答案
    first_fact_vs_ans=[]
    for p,a in TRAIN:
        first_color = "红色" if "小明有红色" in p else "绿色"
        first_fact_vs_ans.append({"answer":a,"first_mentioned_color":first_color,"equal":a==first_color})
    shortcut = {
        "answer_frequency": dict(ans_freq),
        "fixed_red_accuracy_on_train": fixed_red/len(TRAIN),
        "fixed_green_accuracy_on_train": fixed_green/len(TRAIN),
        "first_fact_color_vs_answer": first_fact_vs_ans,
        "query_template_count": 1,
        "answer_length_tokens": {a:len(tok.encode(a)) for a in set(a for _,a in TRAIN)},
        "note": "事实陈述顺序固定(小明在前)在两组内;答案首字候选仅红/绿两类的一个token;尾字色+EOS恒定"
    }

    # ===== 冻结表示线性探针 =====
    # 提取每个 prompt 最后一个 token 位置的隐蔽表示 h.mean，看能否线性区分答案
    model, meta = load_language_v2(V15/"candidate_v15_baseline.pt", tok)
    model.eval(); model.use_router_load_update=False
    feats=[]; labels=[]
    with torch.inference_mode():
        for p,a in TRAIN:
            ids=tok.encode(p)
            logits,st=model(torch.tensor([ids]))
            feat=st.h.mean(dim=1).squeeze(0)   # [block_hidden]
            feats.append(feat); labels.append(0 if a=="红色" else 1)
    X=torch.stack(feats); y=torch.tensor(labels)
    # 简单线性探针: 用最小二乘/逻辑回归(梯度下降, 只用feat不动模型)
    torch.manual_seed(0)
    W=torch.zeros(X.shape[1],2,requires_grad=True); b=torch.zeros(2,requires_grad=True)
    opt=torch.optim.Adam([W,b],lr=0.05)
    for _ in range(300):
        opt.zero_grad(); loss=torch.nn.functional.cross_entropy(X@W+b,y); loss.backward(); opt.step()
    acc=float(((X@W+b).argmax(-1)==y).float().mean())
    probe={"probe_type":"linear_on_frozen_h_mean", "train_acc_on_4_samples":round(acc,4),
           "note":"探针仅证明冻结表示可解码性, 不计入N-MIND自主语言能力; 4样本过拟合风险极高"}

    OUT.mkdir(parents=True,exist_ok=True)
    payload={"shortcut_audit":shortcut,"frozen_probe":probe}
    (OUT/"_diag_shortcut.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(payload,ensure_ascii=False,indent=2))

if __name__=="__main__": main()
