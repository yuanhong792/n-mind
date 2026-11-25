# ZERO-3.1 historical audit

The historical result is preserved at `results/ZERO3_RESULTS.json`; this audit does not rewrite it.

Static reconstruction:

- `zero3_m0.py` loops 3 rates × 3 seeds × 6 tasks × 8 inner updates = **432** task-network updates. Its JSON says 144, so the historical M0 actual count is **ACTUAL_BUDGET_UNVERIFIED**.
- `zero3.py` M1 and M2 each run 3 seeds × 6 tasks × (8 inner + 1 outer) = 162 updates. Together: 324.
- `zero3.py` evaluates 3 seeds × 2 controllers × 4 tasks × 8 inner updates = 192 task-network updates, not the recorded 288.
- Static total from the checked-in loops is 432 + 324 + 192 = **948**, excluding the separate preflight. The historical hard-coded `actual_updates=768` is not evidence of execution.

The historical result is therefore retained with an `ACTUAL_BUDGET_UNVERIFIED` classification. No historical JSON was overwritten.
