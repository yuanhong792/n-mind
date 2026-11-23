"""V1.7 P0 regression: UTF-8 prefix state machine + token_bytes expansion."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nmind.language.bpe_decoder import utf8_prefix_status
from nmind.language.tokenizer_v2 import ByteBPETokenizer

def test_utf8_prefix_status():
    assert utf8_prefix_status(b"") == "complete"
    assert utf8_prefix_status(b"abc") == "complete"
    assert utf8_prefix_status("红".encode()) == "complete"
    assert utf8_prefix_status("红".encode()[:1]) == "incomplete"
    assert utf8_prefix_status("红".encode()[:2]) == "incomplete"
    assert utf8_prefix_status(b"\xa3") == "invalid"          # V1.6 实际故障字节
    assert utf8_prefix_status(b"\xc0\x80") == "invalid"      # overlong
    assert utf8_prefix_status(b"\xed\xa0\x80") == "invalid"  # surrogate
    assert utf8_prefix_status(b"\xf4\x90\x80\x80") == "invalid"  # >U+10FFFF

def test_token_bytes_roundtrip():
    tok = ByteBPETokenizer.train(["你好世界红色绿色"], vocab_size=4096)
    for ch in "你好世界红色绿色":
        ids = tok.encode(ch)
        buf = b"".join(tok.token_bytes(t) for t in ids)
        assert buf == ch.encode("utf-8")
        assert utf8_prefix_status(buf) == "complete"

def test_partial_han_across_tokens():
    tok = ByteBPETokenizer.train(["红"], vocab_size=260)  # no merges -> pure bytes
    ids = tok.encode("红")
    assert len(ids) == 3
    buf = bytearray()
    assert utf8_prefix_status(bytes(buf)) == "complete"
    buf.extend(tok.token_bytes(ids[0])); assert utf8_prefix_status(bytes(buf)) == "incomplete"
    buf.extend(tok.token_bytes(ids[1])); assert utf8_prefix_status(bytes(buf)) == "incomplete"
    buf.extend(tok.token_bytes(ids[2])); assert utf8_prefix_status(bytes(buf)) == "complete"
