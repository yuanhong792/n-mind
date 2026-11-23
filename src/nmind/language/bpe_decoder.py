from __future__ import annotations
from dataclasses import dataclass, field
import torch
from .tokenizer_v2 import ByteBPETokenizer, EOS

@dataclass
class BPEGeneration:
    token_ids: list[int] = field(default_factory=list)
    text: str = ""
    finish_reason: str = "max_tokens"
    invalid_candidate_ids: list[int] = field(default_factory=list)

def generate_bpe(model, tokenizer: ByteBPETokenizer, prompt: str, *, max_new_tokens=32):
    """Greedy BPE generation with an explicit active-vocabulary mask.

    A trained model head has ``vocab_size`` slots, but a corpus-trained BPE
    can define fewer merges. Undefined slots are not silently decoded as
    ``<unk>``; they are recorded and excluded from sampling.
    """
    model.eval(); ids = tokenizer.encode(prompt); x = torch.tensor([ids], dtype=torch.long)
    active = set(range(4, 260)) | {EOS}
    active.update(260 + i for i in range(len(tokenizer.merges)))
    rec = BPEGeneration(); blocked=[]
    with torch.inference_mode():
        logits, state = model(x)
        for _ in range(max_new_tokens):
            last = logits[0, -1].clone(); invalid = [i for i in range(last.numel()) if i not in active]
            if invalid: blocked.extend(invalid)
            mask = torch.full_like(last, float("-inf")); valid = torch.tensor(sorted(active), dtype=torch.long)
            mask[valid] = last[valid]; tid = int(torch.argmax(mask))
            if tid == EOS: rec.finish_reason="eos"; break
            rec.token_ids.append(tid); logits, state = model(torch.tensor([[tid]]), state)
    rec.invalid_candidate_ids = sorted(set(blocked))
    try: rec.text = tokenizer.decode(rec.token_ids)
    except (ValueError, UnicodeDecodeError): rec.finish_reason="byte_decode_error"
    return rec
