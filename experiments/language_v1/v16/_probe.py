"""V1.6 探查脚本：验证 ByteBPE 跨边界合并导致的 token 对齐问题。"""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
import torch
from nmind.language.tokenizer_v2 import ByteBPETokenizer
from nmind.language.bpe_decoder import effective_token_ids

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "experiments/language_v1"
V15 = SRC / "v15"

LEGACY = [
    ("用户：你好\n助手：", "你好！"),
    ("用户：你叫什么？\n助手：", "我叫N-MIND。"),
    ("用户：谢谢\n助手：", "不客气！"),
    ("用户：小明有红色苹果\n助手：", "小明的苹果是红色的。"),
    ("用户：小红有绿色苹果\n助手：", "小红的苹果是绿色的。"),
    ("用户：小刚有蓝色苹果\n助手：", "小刚的苹果是蓝色的。"),
    ("用户：小丽有黄色苹果\n助手：", "小丽的苹果是黄色的。"),
    ("用户：我叫小明\n助手：", "你好，小明！"),
    ("用户：我叫什么？\n助手：", "你叫小明。"),
]
TRAIN = [
    ("用户：小明有红色苹果，小红有绿色苹果。\n用户：小明的苹果是什么颜色？\n助手：", "红色"),
    ("用户：小明有红色苹果，小红有绿色苹果。\n用户：小红的苹果是什么颜色？\n助手：", "绿色"),
    ("用户：小明有绿色苹果，小红有红色苹果。\n用户：小明的苹果是什么颜色？\n助手：", "绿色"),
    ("用户：小明有绿色苹果，小红有红色苹果。\n用户：小红的苹果是什么颜色？\n助手：", "红色"),
]

def token_byte_len(tok, tid):
    """该 token 解码后的 UTF-8 字节数。"""
    try:
        return len(tok.decode([tid]).encode("utf-8"))
    except Exception:
        return None

def main():
    torch.set_num_threads(1)
    tok = ByteBPETokenizer.train([p + a for p, a in LEGACY], vocab_size=4096)
    print("tokenizer_hash:", tok.vocab_hash(), "vocab:", tok.vocab_size, "merges:", len(tok.merges))
    print("effective_token_count:", len(effective_token_ids(tok)))
    print()
    out = []
    for idx, (p, a) in enumerate(TRAIN):
        pre_ids = tok.encode(p)                 # 训练脚本用来算 start 的：单独编码 prompt
        ans_ids = tok.encode(a, add_eos=True)   # lm_update 里 encode_rows 的正确做法：答案单独编码 + EOS
        joint_ids = tok.encode(p + a, add_eos=True)  # train_v15 实际送入模型的：联合编码
        # train_v15 的做法
        start_v15 = len(tok.encode(p)) - 1
        n = len(joint_ids)
        # 正确做法：答案 span 应基于 pre_ids 长度
        correct_start = len(pre_ids)
        # 联合编码里，前缀 token 到底占几个？用字节数反推 prompt 的边界
        p_bytes = len(p.encode("utf-8"))
        cum = 0
        joint_prefix_len = None
        for j, tid in enumerate(joint_ids):
            bl = token_byte_len(tok, tid)
            if bl is None:
                break
            cum += bl
            if cum >= p_bytes:
                joint_prefix_len = j + 1  # prompt 占用了 j+1 个 token
                break
        # v15 掩码实际覆盖的 target 位置（mask[i,start:n-1] 对应 target=x[:,1:] 的位置 start..n-2）
        v15_mask_positions = list(range(start_v15, n - 1))
        correct_mask_positions = list(range(correct_start - 1, n - 1))
        row = {
            "idx": idx,
            "answer": a,
            "prompt_bytes": p_bytes,
            "len_pre_ids_alone": len(pre_ids),
            "len_ans_ids_alone_eos": len(ans_ids),
            "len_joint_ids": n,
            "joint_prefix_tokens_for_prompt": joint_prefix_len,
            "v15_start": start_v15,
            "correct_start": correct_start,
            "v15_mask_positions": v15_mask_positions,
            "correct_mask_positions": correct_mask_positions,
            "v15_supervised_targets": [tok.decode([joint_ids[k]]) for k in v15_mask_positions if k < n],
            "correct_supervised_targets": [tok.decode([joint_ids[k]]) if joint_ids[k] != 2 else "<eos>" for k in correct_mask_positions if k < n],
            "misaligned": v15_mask_positions != correct_mask_positions,
        }
        out.append(row)
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
