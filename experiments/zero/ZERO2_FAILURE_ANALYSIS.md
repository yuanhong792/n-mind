# ZERO-2 preparation and ZERO-1 audit

No new training was run. ZERO-1 used `target - (W h + b)` with one-hot output targets and normalized denominator `||h||²+eps`; output logits were not converted to probabilities. The hidden feature matrix has numerical rank 16/16 for all three seeds, so low rank is not established as the sole cause. The curves show non-monotonic interference (especially seed 202: 0.869 at step 50, 0.377 at step 128). The available evidence supports instability and target/update scaling as hypotheses, not a single proven root cause.

ZERO-0 sealed sets are independently generated from the training RNG stream and contain 512 unique random rows per seed. Class-1 rates are train/sealed: seed 101 0.453/0.438, seed 202 0.297/0.252, seed 303 0.492/0.553; initial differences therefore include genuine class prevalence differences but are not caused by duplicate data. No hidden rule tensors are passed to models; evaluation uses `no_grad` and does not mutate weights.
