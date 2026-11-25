# E1 protocol

Each environment seed generates a hidden nonlinear rule from a private deterministic RNG. Training and sealed inputs use disjoint RNG streams. The model receives only features and labels for online training; it never receives the hidden rule. The sealed set contains 512 independently generated samples and is evaluated once after training.
