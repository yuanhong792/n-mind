# N-MIND ZERO-0 initial report

## Result

ZERO-A (178 trainable parameters, random initialization) learned the E1 hidden nonlinear rule in 128 online updates per seed. Sealed accuracy after training was 0.8789, 0.9277 and 0.8984 for seeds 101, 202 and 303. Initial accuracies were 0.6543, 0.5215 and 0.3379. All seeds had non-zero weight change. Frozen controls remained at their initial predictions.

The run used 384 of the 768 permitted optimizer updates and PyTorch 2.10.0+cpu. Evaluation used an independent 512-example sealed set and did not update weights. Checkpoints were written under `experiments/zero/checkpoints/`.

## Interpretation

The first target is met: a randomly initialized micro-network acquired a measurable ability and generalized to unseen samples from the same hidden rule. The result is a positive control for the environment, not a claim of autonomous learning. New-rule learning, environment change, continual retention, local plasticity and meta-learning remain untested.

## Next smallest bottleneck

Run ZERO-B against the same E1 protocol, with a preregistered local update and the same 128-update/seed budget. Keep ZERO-A as the positive control; if ZERO-B fails, retain the standard MLP and diagnose the local update rather than increasing architectural complexity.
