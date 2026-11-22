"""Reversible UTF-8 byte tokenizer for Language V2.

IDs 0..3 remain reserved for padding and sequence boundaries; every byte is
represented exactly, so inference never silently turns unseen Chinese into UNK.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path

PAD, BOS, EOS, UNK = 0, 1, 2, 3
BYTE_OFFSET = 4
BYTE_COUNT = 256

class TokenizerV2:
    version = "nmind-tokenizer-v2-utf8-byte"
    def __init__(self) -> None: self.vocab_size = BYTE_OFFSET + BYTE_COUNT
    def encode(self, text: str, *, add_bos=False, add_eos=False, max_len=None) -> list[int]:
        if not isinstance(text, str): raise TypeError("text must be str")
        ids = ([BOS] if add_bos else []) + [BYTE_OFFSET + b for b in text.encode("utf-8", "strict")]
        if add_eos: ids.append(EOS)
        return ids[:max_len] if max_len is not None else ids
    def decode(self, ids: list[int], *, skip_special=True) -> str:
        raw = bytearray()
        out: list[str] = []
        for value in ids:
            i = int(value)
            if BYTE_OFFSET <= i < BYTE_OFFSET + BYTE_COUNT: raw.append(i - BYTE_OFFSET); continue
            if raw: out.append(bytes(raw).decode("utf-8", "strict")); raw.clear()
            if not skip_special and i in (PAD, BOS, EOS, UNK): out.append({PAD:"<pad>",BOS:"<bos>",EOS:"<eos>",UNK:"<unk>"}[i])
            elif not skip_special: raise ValueError(f"unknown token id {i}")
        if raw: out.append(bytes(raw).decode("utf-8", "strict"))
        return "".join(out)
    def vocab_hash(self) -> str:
        return hashlib.sha256(json.dumps({"version":self.version,"vocab_size":self.vocab_size},sort_keys=True).encode()).hexdigest()[:16]
    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"format":"nmind-tokenizer/2","version":self.version,"vocab_size":self.vocab_size,"vocab_hash":self.vocab_hash()}),encoding="utf-8")
    @classmethod
    def load(cls, path: str | Path) -> "TokenizerV2":
        blob=json.loads(Path(path).read_text(encoding="utf-8")); obj=cls()
        if blob.get("format") != "nmind-tokenizer/2" or blob.get("vocab_hash") != obj.vocab_hash() or blob.get("vocab_size") != obj.vocab_size: raise ValueError("TokenizerV2 hash/version mismatch")
        return obj
