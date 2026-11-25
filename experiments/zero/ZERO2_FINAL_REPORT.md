# ZERO-2 status

Preparation only. T2/T3 cross-task training is not run because the requested new 768-update budget is a proposal, not authorization. The executable entry point is `python -m experiments.zero.zero2` (dry preparation); it refuses training unless `--authorize` is explicitly supplied.

The next decision is whether to authorize the paired 3-seed T2/T3 run. Its primary result will be the learning-curve difference between fresh and inherited ZERO-A on independent environment B, with environment A retention reported separately.
