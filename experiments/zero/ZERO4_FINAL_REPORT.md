# ZERO-4 preparation report

ZERO-4.1 completed the authorized 2568 updates on CPU. The functional inner-loop path was verified with non-zero controller gradients before execution. Mean A/B balanced accuracy after sequential A→B learning was: G0 0.602, G1 0.594, G2 0.622, G3 0.617, G4 0.620.

G2 exceeded G0 and G1 in this small run, but did not exceed G4. Therefore the result is not evidence of an independently useful learned selective-update strategy. G2's apparent gain is not separated from the random-controller/static-gate controls. Hidden/output gates are soft strengths; no hard zero-freeze claim is made. Full raw rows and budget state are in `results/ZERO4_RESULTS.json` and `results/ZERO4_BUDGET_STATE.json`.
