# ZERO-3 protocol

Each task samples an independent hidden nonlinear rule with the shared E1 input/output structure. Support and query samples are independent. The task network is freshly initialized for every task; only controller parameters persist across meta-training. The controller sees only support loss, gradient norm, loss delta, step fraction, parameter norm and previous learning rate. It never sees hidden rule tensors, query labels during adaptation, or task IDs.

M0 uses a fixed learning rate selected from a preregistered grid on meta-development tasks. M1 learns one global scalar rate. M2 maps legal state features to a bounded positive rate. Final held-out tasks are evaluated with the controller frozen.
