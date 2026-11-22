# N-MIND Language V2 Gate-2 report

## Execution status

The Gate-1 precondition passed after verifying the LanguageWorld oracle, Tokenizer V2 round-trip, split isolation, offline mode and BotRuntime control path. The controlled one-round experiment then executed exactly 1056 optimizer steps: main 512, permuted-label 512 and lr0 32. All checkpoints were written under `experiments/language_v2/gate2/runs/`; stable and historical checkpoints were not written or promoted.

## Training evidence

Main loss: 5.525930404663086 → 0.7190589308738708, 512 steps, 25,344 tokens, CPU wall time 175.18 s, parameter hash changed.

Permuted-label loss: 5.566213130950928 → 0.6984408497810364, 512 steps, 25,344 tokens, CPU wall time 172.77 s, parameter hash changed.

LR0 loss: 5.58607292175293 → 5.590318202972412, 32 steps, 1,584 tokens, CPU wall time 11.07 s, parameter hash unchanged. This confirms the optimizer control path did not alter parameters at zero learning rate.

## Sealed neural-only evaluation

All memory retrieval, knowledge retrieval, tools and rule answer backfill were disabled. A single greedy continuation evaluation was performed over the sealed set and 80 counterfactual pairs. Main, permuted-label and lr0 all produced exact-match 0.0 and counterfactual pair pass 0/80. Detailed summary is `experiments/language_v2/gate2/sealed_evaluation_summary.json`; the evaluator retains per-item predictions in its runtime output protocol and does not alter sealed data.

The honest conclusion is `TRAIN_SIGNAL_ONLY_BINDING_FAIL`: the main loss decreased, but no unseen-situation generalization or fact-binding evidence was obtained. The labels-permuted control also reduced loss, indicating the current autoregressive objective/data construction can learn surface/token statistics without demonstrating the requested semantic skill. This is a data/objective/evaluation limitation signal, not evidence that N-Cell learned binding.

## Safety

- optimizer steps: 1056 total, within the authorized 1056.
- network/downloads: 0; unreviewed public corpora were not loaded.
- external pretrained models/APIs: 0.
- stable checkpoint: read-only and not promoted.
- Gate-B 005: paused.
- single seed: results are diagnostic only, not a stability claim.

## Reproduction

```powershell
$env:PYTHONPATH = "src"
& "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe" experiments\language_v2\gate2\run_gate2.py --preflight-only --offline
& "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe" -m pytest -q tests/test_language_v2_gate2.py tests/test_language_v2_gate2_eval.py tests/test_language_v2_gate1.py tests/test_language_v2.py tests/test_language_v2_pipeline.py
```

The training command was the same runner with `--i-authorize-gate2 --offline`; rerunning it would violate the single-round/no-extra-training rule and is intentionally not requested.
