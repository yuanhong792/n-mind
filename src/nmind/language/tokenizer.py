"""自有字符级分词器（M5-Language 任务书 P0）：中文逐字、ASCII 词元聚合、UNK 语义明确。

* 词表只从允许的训练集构建；冻结后持久化 sha256；评测/推理一律不扩词表——
  未见汉字映射为 <unk>（UTF-8 byte fallback 留作 v2 配置项，默认关并如实记录）。
* 特殊 token 语义：PAD=0 填充｜BOS=1 序列起始｜EOS=2 轮次/序列结束｜UNK=3 未见字符。
* 切分规则：CJK 汉字与全角标点逐字；连续 ASCII 字母/数字聚合为一个词元（小写）；
  空格、换行、西文标点逐字符。换行 \\n 是轮次边界的自然组成部分，单独成词元。
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

PAD, BOS, EOS, UNK = 0, 1, 2, 3
SPECIAL_TOKENS: dict[str, int] = {"<pad>": PAD, "<bos>": BOS, "<eos>": EOS, "<unk>": UNK}
_TOKENIZER_VERSION = "char-tokenizer-v1"

_ASCII_RE = re.compile(r"[A-Za-z0-9]")


def _tokenize_text(text: str) -> list[str]:
    tokens: list[str] = []
    buf = ""

    def flush() -> None:
        nonlocal buf
        if buf:
            tokens.append(buf.lower())
            buf = ""

    for ch in text:
        if _ASCII_RE.match(ch):
            buf += ch
        else:
            flush()
            tokens.append(ch)
    flush()
    return tokens


class CharTokenizer:
    def __init__(self, vocab: dict[str, int]):
        for name, idx in SPECIAL_TOKENS.items():
            if vocab.get(name) != idx:
                raise ValueError(f"词表缺少/错位特殊 token {name}（应为 {idx}）")
        self.vocab = dict(vocab)
        self.id2tok = {i: t for t, i in self.vocab.items()}

    @classmethod
    def train_from_texts(cls, texts: list[str], *, max_size: int = 2000) -> "CharTokenizer":
        counts: Counter = Counter()
        for t in texts:
            counts.update(_tokenize_text(t))
        vocab = dict(SPECIAL_TOKENS)
        for tok, _c in counts.most_common():
            if tok in vocab:
                continue
            if len(vocab) >= max_size:
                break
            vocab[tok] = len(vocab)
        return cls(vocab)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False,
               max_len: int | None = None) -> list[int]:
        ids: list[int] = []
        if add_bos:
            ids.append(BOS)
        for tok in _tokenize_text(text):
            ids.append(self.vocab.get(tok, UNK))
        if add_eos:
            ids.append(EOS)
        if max_len is not None:
            ids = ids[:max_len]
        return ids

    def decode(self, ids: list[int], *, skip_special: bool = True) -> str:
        out: list[str] = []
        for i in ids:
            tok = self.id2tok.get(int(i), "<unk>")
            if tok == "<unk>":
                out.append("\ufffd")         # 未见字符如实显示为替换符，不静默丢弃
                continue
            if skip_special and tok in SPECIAL_TOKENS:
                continue                     # PAD/BOS/EOS 不进文本
            out.append(tok)
        return "".join(out)

    def vocab_hash(self) -> str:
        payload = json.dumps(self.vocab, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"format": "nmind-tokenizer/1",
                                  "version": _TOKENIZER_VERSION,
                                  "vocab": self.vocab,
                                  "vocab_hash": self.vocab_hash()},
                                 ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        blob = json.loads(Path(path).read_text(encoding="utf-8"))
        if blob.get("format") != "nmind-tokenizer/1":
            raise ValueError(f"不是 nmind tokenizer（format={blob.get('format')!r}）")
        tok = cls(blob["vocab"])
        if tok.vocab_hash() != blob.get("vocab_hash"):
            raise ValueError("tokenizer 词表哈希不一致：文件被改动，拒绝加载")
        return tok
