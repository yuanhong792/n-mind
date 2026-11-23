"""V1.7 P0: frozen V1.6 model, old vs new decoder, isolate decoder-only gain. ZERO training."""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT / "src"))
import torch
from nmind.language.tokenizer_v2 import ByteBPETokenizer
from nmind.language.neural_language_v2 import load_language_v2
from nmind.language.bpe_decoder import generate_bpe, generate_bpe_legacy, utf8_prefix_status

torch.set_num_threads(1)
LEGACY = [("用户：你好\n助手：","你好！"),("用户：你叫什么？\n助手：","我叫N-MIND。"),("用户：谢谢\n助手：","不客气！"),("用户：小明有红色苹果\n助手：","小明的苹果是红色的。"),("用户：小红有绿色苹果\n助手：","小红的苹果是绿色的。"),("用户：小刚有蓝色苹果\n助手：","小刚的苹果是蓝色的。"),("用户：小丽有黄色苹果\n助手：","小丽的苹果是黄色的。"),("用户：我叫小明\n助手：","你好，小明！"),("用户：我叫什么？\n助手：","你叫小明。")]
SEALED = ["用户：小刚有蓝色书包，小丽有黄色书包。\n用户：小丽的书包是什么颜色？\n助手：", "用户：小刚有黄色杯子，小丽有蓝色杯子。\n用户：小刚的杯子是什么颜色？\n助手：", "用户：小丽有绿色苹果，小明有蓝色苹果。\n用户：小明的苹果是什么颜色？\n助手：", "用户：小红有黄色书包，小刚有绿色书包。\n用户：小红的书包是什么颜色？\n助手："]
# UTF-8 状态机自检
sm_test = {"empty": utf8_prefix_status(b""), "ascii": utf8_prefix_status(b"abc"),
           "full_han": utf8_prefix_status("红".encode()), "partial_han": utf8_prefix_status("红".encode()[:2]),
           "lead_byte": utf8_prefix_status("红".encode()[:1]), "bad_start": utf8_prefix_status(b"\xa3"),
           "overlong": utf8_prefix_status(b"\xc0\x80"), "surrogate": utf8_prefix_status(b"\xed\xa0\x80")}

tok = ByteBPETokenizer.train([p+a for p,a in LEGACY], vocab_size=4096)
m, _ = load_language_v2(ROOT / "experiments/language_v1/v16/candidate_v16_baseline_frozen_router.pt", tok)
m.eval(); m.use_router_load_update = False
rows = []
for p in SEALED:
    old = generate_bpe_legacy(m, tok, p, max_new_tokens=16)
    new = generate_bpe(m, tok, p, max_new_tokens=16, utf8_safe=True)
    rows.append({"prompt": p,
                 "old": {"text": old.text, "finish": old.finish_reason, "tokens": old.token_ids, "byte_hex": old.raw_byte_hex},
                 "new": {"text": new.text, "finish": new.finish_reason, "tokens": new.token_ids,
                         "byte_hex": new.raw_byte_hex, "rejected": len(new.rejected_utf8_tokens),
                         "incomplete_steps": new.utf8_incomplete_steps}})
def utf8_ok(rec_text, finish): return finish in ("eos","max_tokens") and "byte_decode_error" not in finish
old_ok = sum(utf8_ok(r["old"]["text"], r["old"]["finish"]) for r in rows)
new_ok = sum(utf8_ok(r["new"]["text"], r["new"]["finish"]) for r in rows)
result = {"status": "DECODER_COMPARE_DONE", "model": "V1.6 candidate_v16_baseline_frozen_router (FROZEN, no weight update)",
          "optimizer_steps": 0, "utf8_state_machine_selftest": sm_test,
          "n_cases": len(rows), "old_utf8_legal": old_ok, "new_utf8_legal": new_ok,
          "decoder_only_gain": f"{old_ok}/{len(rows)} -> {new_ok}/{len(rows)}",
          "attribution_note": "该提升仅来自 decoder UTF-8 过滤，模型神经权重完全未变，不代表模型学到新知识",
          "rows": rows}
(ROOT / "experiments/language_v1/v17").mkdir(parents=True, exist_ok=True)
out = ROOT / "experiments/language_v1/v17/V17_DECODER_FIX.json"
out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"sm_selftest": sm_test, "old_legal": old_ok, "new_legal": new_ok, "gain": result["decoder_only_gain"]}, ensure_ascii=False, indent=2))
