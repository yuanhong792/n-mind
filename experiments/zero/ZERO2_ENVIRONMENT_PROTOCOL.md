# ZERO-2 environment protocol

Environment A uses the existing E1 generator with seed `s`. Environment B uses the identical input/output form but an independent hidden rule and sample RNG with seed `s+1000`. A and B never expose their rule parameters. C and D remain reserved until a separate approved budget.

T2 and T3 use exactly the same B feature tensors and label order. T3 loads the actual ZERO-A checkpoint for the paired seed and reinitializes its optimizer; no checkpoint means T3 is invalid. Sealed sets are generated independently and read only at registered endpoints.
