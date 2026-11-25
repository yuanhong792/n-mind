# ZERO-1 failure analysis

The preregistered B2 run completed 384 local updates. Sealed accuracy was 0.6641, 0.3770 and 0.6973 (mean 0.5794), below the 0.85 candidate threshold and below ZERO-A. All three local weights changed, with 0 backward calls and 0 optimizer steps. Reloaded checkpoints reproduced the same three accuracies.

The failure is not explained by duplicate samples or a collapsed class distribution: each train split has 128 unique rows, each sealed split 512 unique rows, and class-1 rates are 0.453/0.297/0.492 for training versus 0.438/0.252/0.553 sealed. The task is still the same hidden rule per seed; unfamiliar-rule transfer was not tested.

The update uses output error with a one-hot target and normalized Delta. It changes the output layer only; the random tanh encoder is frozen and receives no error signal. The large oscillations (seed 202 peaks at 0.869 at step 50 then falls to 0.377) indicate that the unit-gain update is too aggressive for this representation and target scale. This is a local-rule failure, not evidence that local learning is impossible. No extra training was run after the preregistered budget was exhausted.
