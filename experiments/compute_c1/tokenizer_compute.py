"""C1.0 计算领域字符级 Tokenizer。

- 固定小词表（<=128），无 BPE merge，杜绝跨 prompt/answer 合并。
- 覆盖数字、大小写字母、运算/逻辑/控制符、空白与特殊符号。
- 无损 round-trip；训练与推理共用同一个 encode。
- 版本与 SHA 冻结；checkpoint 必须记录。
不影响任何现有中文 tokenizer。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

PAD, BOS, EOS, UNK = 0, 1, 2, 3
SPECIAL = {PAD: "<pad>", BOS: "<bos>", EOS: "<eos>", UNK: "<unk>"}

# 所有可打印 ASCII（32..126）+ 换行/制表；字符级，无合并。
_CHARS = [chr(c) for c in range(32, 127)]
_EXTRA = ["\n", "\t"]
# 特殊符号只用 ID 表示，不占用字符表；全部可打印 ASCII 都可用。
_SYMBOLS = sorted(set(_CHARS + _EXTRA))
ID2CHAR = {i: ch for i, ch in enumerate(_SYMBOLS, start=4)}
CHAR2ID = {ch: i for i, ch in ID2CHAR.items()}
VOCAB_SIZE = max(ID2CHAR.keys()) + 1
VERSION = "nmind-compute-char/1"


def vocab_sha() -> str:
    payload = {"version": VERSION, "vocab_size": VOCAB_SIZE,
               "special": SPECIAL, "chars": list(ID2CHAR.values())}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class ComputeTokenizer:
    version = VERSION

    def __init__(self):
        self.vocab_size = VOCAB_SIZE

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("text must be str")
        out = ([BOS] if add_bos else [])
        for ch in text:
            tid = CHAR2ID.get(ch)
            if tid is None:
                raise ValueError(f"未定义字符 {ch!r} (ord={ord(ch)})；计算词表禁止静默 UNK")
            out.append(tid)
        if add_eos:
            out.append(EOS)
        return out

    def decode(self, ids: list[int], *, skip_special: bool = True) -> str:
        parts: list[str] = []
        for value in ids:
            i = int(value)
            if i in ID2CHAR:
                parts.append(ID2CHAR[i])
            elif i in SPECIAL:
                if not skip_special:
                    parts.append(SPECIAL[i])
            else:
                raise ValueError(f"未定义 token id {i}")
        return "".join(parts)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({
            "format": "nmind-compute-tokenizer/1", "version": VERSION,
            "vocab_size": VOCAB_SIZE, "sha256": vocab_sha(),
            "id2char": {str(k): v for k, v in ID2CHAR.items()},
        }, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ComputeTokenizer":
        blob = json.loads(Path(path).read_text(encoding="utf-8"))
        if blob.get("version") != VERSION or blob.get("sha256") != vocab_sha():
            raise ValueError("计算 tokenizer 版本/SHA 不匹配")
        return cls()

    @property
    def vocab_sha(self) -> str:
        return vocab_sha()


def effective_ids(tok: ComputeTokenizer) -> list[int]:
    return [PAD, BOS, EOS, UNK] + sorted(ID2CHAR.values())
