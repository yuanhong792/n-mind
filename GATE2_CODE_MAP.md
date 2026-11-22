# Gate-2 code map

- `src/nmind/language/gate2_dataset.py`: deterministic L0/L1 train/dev/sealed records and 80 counterfactual pairs.
- `src/nmind/language/gate2_preflight.py`: fail-closed preflight and resource/environment audit.
- `src/nmind/language/gate2_trainer.py`: bounded candidate-only optimizer loop and parameter hash audit.
- `src/nmind/language/gate2_evaluator.py`: neural-only greedy continuation, exact match and paired counterfactual score.
- `experiments/language_v2/gate2/run_gate2.py`: single preflight-gated experiment entrypoint.
- `src/nmind/bot/runtime.py`: atomic Windows status publication and pause polling contention fix.

Memory, retrieval, tools and rule answers are not imported by the Gate-2 runner.
