# ZERO-4 minimal architecture

The task core remains ZERO-A (8→16→2, 178 parameters). Two parameter groups are explicit: hidden encoder and output layer. G0 updates both at full strength. G1 uses two frozen scalar strengths. G2 uses a frozen-at-evaluation controller that maps legal online state to two bounded strengths. G3 is an untrained random controller; G4 uses development-matched static strengths. No task answers, dictionaries or per-task checkpoints are retained as memory.
