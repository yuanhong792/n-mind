"""V1.6 探查3：对比训练口径(joint encode)与评分口径(answer alone encode)的 token id 是否一致。"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
import torch, json
from nmind.language.tokenizer_v2 import ByteBPETokenizer, EOS

LEGACY = [
    ("用户：你好\n助手：", "你好！"), ("用户：你叫什么？\n助手：", "我叫N-MIND。"),
    ("用户：谢谢\n助手：", "不客气！"), ("用户：小明有红色苹果\n助手：", "小明的苹果是红色的。"),
    ("用户：小红有绿色苹果\n助手：", "小红的苹果是绿色的。"), ("用户：小刚有蓝色苹果\n助手：", "小刚的苹果是蓝色的。"),
    ("用户：小丽有黄色苹果\n助手：", "小丽的苹果是黄色的。"), ("用户：我叫小明\n助手：", "你好，小明！"),
    ("用户：我叫什么？\n助手：", "你叫小明。"),
]
def main():
    tok=ByteBPETokenizer.train([p+a for p,a in LEGACY],vocab_size=4096)
    results=[]
    # 对 V15 的 4 个封存题 + 4 个训练题都检查
    prompts=[
        ("用户：小明有红色苹果，小红有绿色苹果。\n用户：小明的苹果是什么颜色？\n助手：","红色"),
        ("用户：小明有红色苹果，小红有绿色苹果。\n用户：小红的苹果是什么颜色？\n助手：","绿色"),
        ("用户：小刚有蓝色书包，小丽有黄色书包。\n用户：小丽的书包是什么颜色？\n助手：","黄色"),
        ("用户：小丽有绿色苹果，小明有蓝色苹果。\n用户：小明的苹果是什么颜色？\n助手：","蓝色"),
    ]
    for p,a in prompts:
        ans_alone=tok.encode(a)             # score() 用的
        pre=tok.encode(p)
        joint=tok.encode(p+a,add_eos=True)
        # 联合编码中答案部分(去掉EOS)的 token：joint[len(pre):-1]
        joint_ans=joint[len(pre):-1] if len(joint)>len(pre)+1 else []
        match = ans_alone==joint_ans
        results.append({
            "answer":a,
            "answer_alone_ids":ans_alone,
            "joint_answer_ids_tail":joint_ans,
            "len_pre":len(pre),"len_joint":len(joint),
            "match":match,
        })
    print(json.dumps(results,ensure_ascii=False,indent=2))

if __name__=="__main__": main()
