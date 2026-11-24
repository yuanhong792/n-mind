"""M2 SEP 显式边界 Tokenizer。

- 字符 ID 与 M1 完全一致（ASCII 32..126 + \\n\\t → id 4..100），历史数据可直接复用。
- 新增专用边界符 SEP=101：序列 = prompt_ids + [SEP] + answer_ids + [EOS]。
- SEP 不出现在任何正常字符中，确保 prompt 与 answer 边界唯一、无前缀吞并。
- 训练与推理共用同一 encode；推理输入固定为 prompt_ids + [SEP]，模型在 SEP 之后开始生成答案。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

PAD, BOS, EOS, UNK, SEP = 0, 1, 2, 3, 101
SPECIAL = {PAD: "<pad>", BOS: "<bos>", EOS: "<eos>", UNK: "<unk>", SEP: "<sep>"}

# 与 M1 完全一致的字符表与 ID 分配
_CHARS = [chr(c) for c in range(32, 127)] + ["\n", "\t"]
_SYMBOLS = sorted(set(_CHARS))
ID2CHAR = {i: ch for i, ch in enumerate(_SYMBOLS, start=4)}
CHAR2ID = {ch: i for i, ch in ID2CHAR.items()}
VOCAB_SIZE = 102  # 0..100 为 M1 原有 + 101 为 SEP

VERSION = "nmind-math-m2-sep/1"


def vocab_sha() -> str:
    payload = {"version": VERSION, "vocab_size": VOCAB_SIZE,
               "special": SPECIAL, "chars": list(ID2CHAR.values()), "sep": SEP}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


class M2Tokenizer:
    version = VERSION

    def __init__(self):
        self.vocab_size = VOCAB_SIZE

    @property
    def vocab_sha(self) -> str:
        return vocab_sha()

    def encode(self, text: str) -> list[int]:
        """仅编码普通字符（不含 SEP/EOS）；字符 ID 与 M1 完全一致。"""
        if not isinstance(text, str):
            raise TypeError("text must be str")
        out = []
        for ch in text:
            tid = CHAR2ID.get(ch)
            if tid is None:
                raise ValueError(f"未定义字符 {ch!r} (ord={ord(ch)})；禁止静默 UNK")
            out.append(tid)
        return out

    def decode(self, ids, skip_special: bool = True) -> str:
        parts: list[str] = []
        for value in ids:
            i = int(value)
            if i in ID2CHAR:
                parts.append(ID2CHAR[i])
            elif i == SEP:
                if not skip_special:
                    parts.append("|")
            elif i in SPECIAL:
                if not skip_special:
                    parts.append(SPECIAL[i])
            else:
                raise ValueError(f"未定义 token id {i}")
        return "".join(parts)

    # ---- SEP 协议核心 ----

    def encode_pair(self, prompt: str, answer: str) -> tuple[list[int], dict]:
        """训练序列：prompt_ids + [SEP] + answer_ids + [EOS]。
        返回 (seq, meta)，meta 记录 answer 区域起止，供 loss 掩码使用。"""
        p = self.encode(prompt)
        a = self.encode(answer)
        seq = p + [SEP] + a + [EOS]
        # answer 监督目标位于 seq 下标 answer_start..answer_end-1，
        # 由前一位置预测；即 seq[answer_start-1](SEP) 预测 seq[answer_start](a[0])。
        return seq, {
            "prompt_len": len(p),
            "sep_index": len(p),
            "answer_start": len(p) + 1,
            "answer_len": len(a),
            "eos_index": len(seq) - 1,
            "seq_len": len(seq),
        }

    def encode_inference(self, prompt: str) -> list[int]:
        """推理输入：prompt_ids + [SEP]，模型从 SEP 位置开始生成答案。"""
        return self.encode(prompt) + [SEP]

    def encode_legacy(self, prompt: str, answer: str) -> list[int]:
        """无 SEP 的历史协议（仅用于前缀冲突审计对照，禁止用于 M2 训练）。"""
        return self.encode(prompt) + self.encode(answer) + [EOS]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({
            "format": "nmind-math-m2-tokenizer/1", "version": VERSION,
            "vocab_size": VOCAB_SIZE, "sha256": vocab_sha(),
            "sep": SEP, "special": SPECIAL,
            "id2char": {str(k): v for k, v in ID2CHAR.items()},
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "M2Tokenizer":
        blob = json.loads(Path(path).read_text(encoding="utf-8"))
        if blob.get("version") != VERSION or blob.get("sha256") != vocab_sha():
            raise ValueError("M2 tokenizer 版本/SHA 不匹配")
        return cls()


def allowed_ids() -> list[int]:
    """生成时允许的输出：EOS + 全部普通字符（SEP 不作为生成目标）。"""
    return [EOS] + [i for i in range(4, 101)]
