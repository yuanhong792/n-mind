# Security and lifecycle

V3.2 binds to 127.0.0.1, keeps network disabled, does not execute document text, and does not call training or external LLMs. Browser actions use real local APIs. Task terminal states are protected by TaskService. Stable checkpoint and sealed evaluation data are read-only.
