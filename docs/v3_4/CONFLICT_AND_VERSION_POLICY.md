# 冲突与版本策略

同一 subject/predicate 的同时有效不同值产生 `DIRECT_CONTRADICTION` and remains `OPEN` until an explicit resolution. Different effective intervals are retained as history. Revocation is a tombstone-style invalidation; normal queries exclude revoked assertions and experiences.
