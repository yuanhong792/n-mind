# N-MIND Language V2 report

## Current result

P0 implementation and offline validation are complete. This round created a reversible UTF-8 byte tokenizer, a Language V2 checkpoint/config envelope using the existing N-Cell/Router/Workspace components, an independent 80-pair (160-case) counterfactual data generator, and a zero-step dry-run path.

No optimizer was created or stepped. Gate-B 005 remains paused. No network access, external model, teacher model, distillation, stable-checkpoint write, or Bot runtime replacement was performed.

## Historical audit

M5, M5.1 and M5.2 evidence remains in `experiments/m5_language/`; the prior reports and checkpoints were not overwritten. The new audit index is `experiments/language_v2/audit.json`. Historical results are evidence only and are not relabeled as Language V2 results.

## Offline verification

Run from the repository with Python 3.11 and CPU PyTorch:

```powershell
$env:PYTHONPATH = "src"
& "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe" -m pytest -q tests/test_language_v2.py
& "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe" -c "from nmind.language.train_v2 import dry_run; print(dry_run())"
```

The dry run reports `optimizer_steps_executed: 0`. The tokenizer round-trips Chinese, punctuation, emoji, paths and supplementary Unicode without an UNK substitution.

## Not yet claimed

Language understanding, fact updates, negation, pronoun tracking, multi-turn quality and free generation are `NOT_RUN`; random initialization is not presented as learned capability. Integration with Bot V1.1 remains optional and disabled until an independently authorized training/evaluation budget exists.

## Next authorized round

Use `experiments/language_v2/preregistration.json`. A future run must keep the 80 counterfactual pairs and add sealed update/negation/pronoun/multiturn/free-generation sets, report neural-only vs memory/retrieval/tool arms separately, and preserve the zero-step gate until explicit budget approval.
