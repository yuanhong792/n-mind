# V3.3 架构与决策

`V33LearningEngine` is a bounded non-neural controller. It reads the existing capability ledger, scores registered local task families, records candidate reasons, executes one existing Practice task, verifies with the deterministic Oracle, and writes review/failure evidence to SQLite. It never calls an optimizer, network client, or external model.

Priority factors are explicit: untested gap, recent failures, review phase, deterministic tie-break by skill id, and fixed per-task cost. User permission is a hard gate; disabled, paused, and stopped states do not execute tasks.
