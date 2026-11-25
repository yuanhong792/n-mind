# ZERO-1 final report

ZERO-B2 completed the authorized 384 updates (3 seeds × 128). It made genuine local output-synapse changes without autograd, but failed the candidate acceptance target: sealed accuracy was 0.6641 / 0.3770 / 0.6973, mean 0.5794, versus ZERO-A mean 0.9017. Checkpoints reload bitwise to the same predictions.

Conclusion: ZERO-B2 does not replace ZERO-A. The evidence supports a narrow statement: normalized local Delta can change weights and sometimes improve same-rule predictions, but this configuration is unstable and not a competitive learning mechanism under the current budget. The next bottleneck is the update scaling/target formulation or a diagnostic linear readout upper bound; no additional training is authorized by this run.
