# N-MIND N-Cell Architecture V3.7 — opt-in structure experiment

Source revision: `3ec5fd702ced72116bf01919e05b44e9939c5318`.

This branch implements three independently selectable candidate structural changes:

| Flag | Default | Effect | Additional trainable parameters |
|---|---|---|---|
| `LMConfig.use_reset_gate` | `False` | Adds reset gate to NCell candidate state `W_h @ (r * h)`; dense and sparse paths share math | `NCell.W_r/b_r` only when enabled |
| `LMConfig.use_router_ste` | `False` | Sends `BlockRouter.route().weights` into sparse NCell STE; task gradients reach score MLP | None |
| `LMConfig.readout_mode` | `"mean"` | `"query_attention"` uses current token + prior Workspace read to select recurrent blocks for prediction | `readout_query/key/ws` only when enabled |

**Important experimental boundary:** None of these changes proves Chinese language understanding, relation binding, retention, or continual learning. They are *trainable candidate architectural mechanisms*, not trained capabilities. No pretrained teacher, model API, network access, or optimizer step was used to author this branch.

## Compatibility rules

1. All three default options retain the prior module parameters and strict `nmind-lm/1` state_dict keys.
2. Old checkpoints without these config flags load with legacy defaults. New candidates persist their architecture in `LMConfig.to_dict()`, and cannot silently load into the wrong shape.
3. An opt-in readout reset/router variant **changes the mathematical computation** and must be treated as a **new candidate**, not an unchanged continuation of a trained Legacy checkpoint; compare from the same declared initialization or explicitly record migration.
4. Never overwrite stable or historical sealed evaluation sets.
5. `use_router_ste=True` preserves the hard forward output while enabling an approximate selected-route gradient. It does not learn selection for unchosen blocks from their own state updates and it is not equivalent to full Soft Mixture.
6. Query attention reads the previous Workspace state; it cannot access target answers. `readout_ws` adds a direct learned Workspace-to-output pathway.
7. The reset-gate candidate increases parameters; comparisons must disclose total parameter count and ideally include a parameter-matched baseline.

## Local verification (must be run on the user's Windows machine)

```powershell
python -m pytest tests/test_ncell_architecture_v37.py -q
python -m pytest tests/test_ncell.py tests/test_router.py tests/test_router_ste.py tests/test_language_m5.py tests/test_gateb004_fixes.py -q
python -m pytest tests -q
```

The GitHub connector provides file reads/writes but **does not execute the project's test environment**. Treat the above tests as `NOT_RUN` until real CI or local execution reports evidence.

## Next experiment (separate authorization and metrics)

- Zero-training: compare legacy vs optional architectures with deterministic seed, logits and gradient norms; verify full forward compatibility and persistence.
- Candidate controlled train: same data/sealed splits; 2x2 router-STE × query-readout ablation; reset-gate separate arm due to extra parameters.
- Metrics: true free generation full-answer exact; strict Unicode/UTF-8 and EOS; context-blind and counterfactual entity-swap controls; neuron/source attribution; GPU/CPU latency; retention.
- Teacher/Memory/Tool must be disconnected for `NEURAL_ONLY` evaluation. Do not infer learning benefit from non-zero gradients or loss alone.

## Known limitations

- The legacy CharTokenizer and the separate byte-level MINLEARN-2 pipeline need distinct evaluation contracts. Do not claim the opt-in CharLM architecture fixes MINLEARN-2's 100% invalid UTF-8 before reproducing that exact pipeline.
- Hard Top-K uses selected weights with STE only when enabled; changing it to full differentiable routing is a different design experiment.
- `err` and `unc` in this language path remain fixed at 0.5; adding calibrated prediction-error signals is separate research.
- The checkpoint format identifier remains `nmind-lm/1` because the existing envelope is unchanged; architecture is serialized as optional flags. Versioning can be revised before promotion.
