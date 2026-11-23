import torch
from nmind.language.tokenizer_v2 import ByteBPETokenizer, EOS
from nmind.language.bpe_decoder import effective_token_ids, constrained_causal_loss

def test_effective_vocab_masks_unused_slots_and_accepts_eos():
    tok=ByteBPETokenizer.train(["你好。苹果。🙂"],vocab_size=4096)
    ids=effective_token_ids(tok)
    assert EOS in ids and max(ids) < tok.vocab_size and len(ids) < tok.vocab_size
    logits=torch.randn(1,3,tok.vocab_size)
    labels=torch.tensor([[ids[0],ids[-1],EOS]])
    loss=constrained_causal_loss(logits,labels,tok)
    assert torch.isfinite(loss)

def test_undefined_training_label_is_rejected():
    tok=ByteBPETokenizer.train(["你好"],vocab_size=4096)
    with __import__('pytest').raises(ValueError):
        constrained_causal_loss(torch.randn(1,1,4096),torch.tensor([[4095]]),tok)
