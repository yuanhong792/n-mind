"""V1.6 探查2：逐 token 展开 train_v15 的实际监督目标与 loss 构成。"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
import torch
from nmind.language.tokenizer_v2 import ByteBPETokenizer, EOS

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
]
def tid_bytes(tok, t):
    """把一个 token id 展开成原始字节（不解码 utf-8）。"""
    reverse={4+i:i for i in range(256)}
    for i,pair in enumerate(tok.merges): reverse[260+i]=pair
    out=bytearray()
    def exp(x):
        if x>=260:
            for c in reverse[x]: exp(c)
        else:
            out.append(x-4 if x>=4 else 0)
    exp(t); return bytes(out)

def show(tok, ids):
    parts=[]
    for t in ids:
        if t==EOS: parts.append("<EOS>")
        elif t==0: parts.append("<PAD>")
        else:
            b=tid_bytes(tok,t)
            try: parts.append(b.decode("utf-8"))
            except Exception: parts.append(f"<bytes:{b.hex()}>")
    return parts

def main():
    tok=ByteBPETokenizer.train([p+a for p,a in LEGACY],vocab_size=4096)
    for p,a in TRAIN:
        joint=tok.encode(p+a,add_eos=True)
        pre=tok.encode(p)
        start=len(pre)-1
        n=len(joint)
        print("="*70)
        print("答案:",a," len(pre)=",len(pre)," start=",start," n=",n)
        print("完整 sequence (joint_ids) 逐 token:")
        for i,t in enumerate(joint):
            mark=""
            if i==start: mark+=" <== v15 mask 起点(位置)"
            print(f"  x[{i:2d}]={t:3d} {show(tok,[t])[0]!r}{mark}")
        print("target = x[1:]，mask 位置 =", list(range(start,n-1)))
        tgt_pos=list(range(start,n-1))
        print("被监督的 target token (即模型要预测的):")
        for k in tgt_pos:
            tid=joint[k+1]
            print(f"  logits位置{k} -> target x[{k+1}]={tid:3d} {show(tok,[tid])[0]!r}")
        print()

if __name__=="__main__": main()
