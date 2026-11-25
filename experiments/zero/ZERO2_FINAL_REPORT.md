# ZERO-2 status

The authorized run used exactly 768 new updates on CPU: T2 fresh ZERO-A and T3 inherited ZERO-A, three paired seeds and 128 B updates each. Mean B sealed accuracy was T2 0.8893 versus T3 0.8939. Per-seed transfer gains were -0.0078, -0.0195 and +0.0410. Learning-curve AUC was T2 0.8065 versus T3 0.8175; gains were mixed (-0.0349, +0.0109, +0.0569). This is weak, mixed preliminary evidence of a small transfer benefit, not a robust claim of general transfer.

T3 reached 80% on B at steps 100, 50 and 20; T2 reached it at 50, 50 and 50. T3 reached 90% only at step 128 for seeds 202 and 303; T2 reached it at step 100 for seed 202 and never for the other two. T3's A sealed accuracy after B was 0.5801/0.7188/0.6621. Because A and B have different answers without a context signal, these changes are interference evidence and not by themselves catastrophic-forgetting proof.

T3 included 128 historical A updates before the B phase. Optimizer state was reinitialized for both groups. The next smallest question is whether the small mixed gain survives a new independently pre-registered task family; no ZERO-3 or extra training is authorized here.

The next decision is whether to authorize the paired 3-seed T2/T3 run. Its primary result will be the learning-curve difference between fresh and inherited ZERO-A on independent environment B, with environment A retention reported separately.
