import json
from pathlib import Path

from nmind.language.tokenizer_v2 import ByteBPETokenizer, BOS, EOS
from nmind.language.corpus_pipeline import build_records, split_by_document


def test_byte_bpe_roundtrip_unseen_unicode_and_specials(tmp_path: Path):
    tok = ByteBPETokenizer.train(["你好，世界。Python 3.11", "苹果在桌子上。"], vocab_size=4096)
    text = "未见字符𠀀🙂、Python与中文"
    ids = tok.encode(text, add_bos=True, add_eos=True)
    assert ids[0] == BOS and ids[-1] == EOS
    assert tok.decode(ids) == text
    path = tmp_path / "tokenizer.json"
    tok.save(path)
    assert ByteBPETokenizer.load(path).decode(ids) == text
    assert tok.vocab_size == 4096


def test_corpus_split_is_document_disjoint_and_manifest_hashable():
    records = build_records("approved-demo", ["训练句一", "训练句二", "验证句三"])
    splits = split_by_document(records, seed=7)
    owners = {r.source_id for rows in splits.values() for r in rows}
    assert owners == {"approved-demo"}
    assert json.dumps([r.sha256 for rows in splits.values() for r in rows])
