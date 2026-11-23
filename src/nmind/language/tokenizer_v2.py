"""Reversible UTF-8 byte tokenizer for Language V2.

IDs 0..3 remain reserved for padding and sequence boundaries; every byte is
represented exactly, so inference never silently turns unseen Chinese into UNK.
"""
from __future__ import annotations
import hashlib, json
from pathlib import Path
from collections import Counter

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


class ByteBPETokenizer:
    """Small dependency-free byte-level BPE tokenizer.

    The base alphabet is all 256 byte values, so encoding is reversible even
    for Unicode that was absent from the training corpus.  Merge learning is
    deliberately deterministic and offline; it consumes already approved
    text supplied by the caller and never downloads a vocabulary.
    """
    version = "nmind-tokenizer-v2-byte-bpe"

    def __init__(self, merges: list[tuple[int, int]] | None = None, *, vocab_size: int = 4096):
        if vocab_size < BYTE_OFFSET + BYTE_COUNT:
            raise ValueError("vocab_size must leave room for all byte tokens")
        self.vocab_size = int(vocab_size)
        self.merges = [tuple(map(int, p)) for p in (merges or [])]

    @classmethod
    def train(cls, texts: list[str], *, vocab_size: int = 4096) -> "ByteBPETokenizer":
        if not texts:
            raise ValueError("at least one approved text is required")
        sequences = [list(t.encode("utf-8", "strict")) for t in texts if t]
        merges: list[tuple[int, int]] = []
        used: set[tuple[int, int]] = set()
        next_id = BYTE_OFFSET + BYTE_COUNT
        limit = vocab_size - next_id
        for _ in range(limit):
            counts = Counter((a, b) for seq in sequences for a, b in zip(seq, seq[1:]))
            if not counts:
                break
            pair, count = min(counts.items(), key=lambda item: (-item[1], item[0]))
            if pair in used:
                counts.pop(pair, None)
                continue
            if count < 2:
                break
            merges.append(pair)
            used.add(pair)
            for seq_i, seq in enumerate(sequences):
                out: list[int] = []
                i = 0
                while i < len(seq):
                    if i + 1 < len(seq) and (seq[i], seq[i + 1]) == pair:
                        out.append(next_id); i += 2
                    else:
                        out.append(seq[i]); i += 1
                sequences[seq_i] = out
            next_id += 1
        return cls(merges, vocab_size=vocab_size)

    def _merge_map(self) -> dict[tuple[int, int], int]:
        return {pair: BYTE_OFFSET + BYTE_COUNT + i for i, pair in enumerate(self.merges)}

    def encode(self, text: str, *, add_bos=False, add_eos=False, max_len=None) -> list[int]:
        ids = list(text.encode("utf-8", "strict"))
        mapping = self._merge_map()
        for pair in self.merges:
            token = mapping[pair]
            out: list[int] = []
            i = 0
            while i < len(ids):
                if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
                    out.append(token); i += 2
                else:
                    out.append(ids[i]); i += 1
            ids = out
        result = ([BOS] if add_bos else []) + [BYTE_OFFSET + x if x < 256 else x for x in ids]
        if add_eos: result.append(EOS)
        return result[:max_len] if max_len is not None else result

    def decode(self, ids: list[int], *, skip_special=True) -> str:
        reverse = {BYTE_OFFSET + i: i for i in range(256)}
        for i, pair in enumerate(self.merges):
            reverse[BYTE_OFFSET + BYTE_COUNT + i] = pair
        expanded: list[int] = []
        for value in ids:
            value = int(value)
            if value in reverse:
                def expand(item: int) -> None:
                    if item >= BYTE_OFFSET + BYTE_COUNT:
                        for child in reverse[item]: expand(child)
                    else:
                        expanded.append(item if value >= BYTE_OFFSET + BYTE_COUNT else item - BYTE_OFFSET)
                expand(value)
            elif value in (PAD, BOS, EOS, UNK):
                if not skip_special: raise ValueError("special token in decoded text")
            else:
                raise ValueError(f"unknown token id {value}")
        return bytes(expanded).decode("utf-8", "strict")

    def vocab_hash(self) -> str:
        payload = {"version": self.version, "vocab_size": self.vocab_size, "merges": self.merges}
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"format": "nmind-tokenizer/2-bpe", "version": self.version,
            "vocab_size": self.vocab_size, "merges": self.merges, "vocab_hash": self.vocab_hash()}), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ByteBPETokenizer":
        blob = json.loads(Path(path).read_text(encoding="utf-8"))
        obj = cls(blob["merges"], vocab_size=blob["vocab_size"])
        if blob.get("format") != "nmind-tokenizer/2-bpe" or blob.get("vocab_hash") != obj.vocab_hash():
            raise ValueError("ByteBPETokenizer hash/version mismatch")
        return obj
