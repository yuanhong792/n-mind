"""M3 字符级 Tokenizer：与 M1/M2 字符 ID 完全一致 + SEP 边界。

序列协议（统一修复后）：prompt_ids + [SEP] + answer_ids + [EOS]
字符表：ASCII 32..126 + \\n\\t → id 4..100；SEP=101；EOS=2；PAD=0。
所有正确率统计都按此协议，legacy 仅在 M2.1 审计中作为对照。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

PAD, BOS, EOS, UNK, SEP = 0, 1, 2, 3, 101
SPECIAL = {PAD: "<pad>", BOS: "<bos>", EOS: "<eos>", UNK: "<unk>", SEP: "<sep>"}
_CHARS = [chr(c) for c in range(32, 127)] + ["\n", "\t"]
_SYMBOLS = sorted(set(_CHARS))
ID2CHAR = {i: ch for i, ch in enumerate(_SYMBOLS, start=4)}
CHAR2ID = {ch: i for i, ch in ID2CHAR.items()}
VOCAB_SIZE = 102
VERSION = "nmind-math-m3-sep/1"


def vocab_sha() -> str:
    payload = {"version": VERSION, "vocab_size": VOCAB_SIZE,
               "special": SPECIAL, "chars": list(ID2CHAR.values()), "sep": SEP}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class M3Tokenizer:
    version = VERSION

    def __init__(self):
        self.vocab_size = VOCAB_SIZE

    @property
    def vocab_sha(self) -> str:
        return vocab_sha()

    def encode(self, text: str) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be str")
        out = []
        for ch in text:
            tid = CHAR2ID.get(ch)
            if tid is None:
                raise ValueError(f"未定义字符 {ch!r}")
            out.append(tid)
        return out

    def decode(self, ids, skip_special=True) -> str:
        parts = []
        for value in ids:
            i = int(value)
            if i in ID2CHAR:
                parts.append(ID2CHAR[i])
            elif i == SEP and not skip_special:
                parts.append("|")
            elif i in SPECIAL and not skip_special:
                parts.append(SPECIAL[i])
        return "".join(parts)

    def encode_pair(self, prompt: str, answer: str):
        p = self.encode(prompt); a = self.encode(answer)
        seq = p + [SEP] + a + [EOS]
        return seq, {"prompt_len": len(p), "answer_start": len(p) + 1,
                     "answer_len": len(a), "eos_index": len(seq) - 1}

    def encode_inference(self, prompt: str) -> list[int]:
        return self.encode(prompt) + [SEP]

    def save(self, path):
        Path(path).write_text(json.dumps({
            "format": "nmind-math-m3-tokenizer/1", "version": VERSION,
            "vocab_size": VOCAB_SIZE, "sha256": vocab_sha(), "sep": SEP,
            "id2char": {str(k): v for k, v in ID2CHAR.items()},
        }, ensure_ascii=False, indent=2), encoding="utf-8")


def allowed_ids():
    return [EOS] + [i for i in range(4, 101)]
