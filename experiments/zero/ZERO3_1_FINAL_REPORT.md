# ZERO-3.1 causal validation final report

The authorized run completed 672/672 task-network updates on CPU. One initial PyTorch API error occurred before any parameter update; its single reservation was explicitly resolved in the persistent ledger and the corrected run completed with 672 reservations and 672 completions.

Mean final accuracy / learning-curve AUC across 36 paired sealed evaluations:

| condition | final accuracy | AUC |
|---|---:|---:|
| C0 tuned fixed SGD | 0.7044 | 0.6108 |
| C1 historical M1 | 0.4922 | 0.4736 |
| C2 historical M2 | 0.5846 | 0.5242 |
| C3 fixed historical mean rate | 0.5833 | 0.5225 |
| C4 random M2 | 0.5768 | 0.5173 |

C2 is slightly above C3 (+0.0017 AUC) and C4 (+0.0069), both below the preregistered +0.01 engineering threshold. C0 is substantially better than C2. Therefore the result is a negative causal result for competitive dynamic learning: the 65-parameter controller has a small signal over the random controller, but no demonstrated independent advantage over a tuned fixed learning rate and no evidence of a useful general learning strategy.

The historical controller weights were frozen during adaptation. New task networks, support samples, query samples and feedback orders were paired across conditions. Full raw curves, learning rates and budget events are in `results/ZERO3_1_RESULTS.json` and `results/ZERO_BUDGET_LEDGER.jsonl`.
