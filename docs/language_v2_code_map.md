# Language V2 code map

- `src/nmind/language/tokenizer_v2.py`: reversible UTF-8 byte tokenizer; IDs 0–3 preserve sequence specials and 4–259 cover every byte.
- `src/nmind/language/neural_language_v2.py`: new checkpoint envelope/config over the existing N-Cell, Router and Workspace language core; all weights are freshly initialized.
- `src/nmind/language/data_v2.py`: deterministic independent 80-pair, 160-case entity-binding corpus generator.
- `src/nmind/language/train_v2.py`: zero-step dry-run entrypoint. It intentionally does not construct an optimizer.
- `src/nmind/language/evaluate_v2.py`: honest NOT_RUN baseline separating neural, memory, retrieval and tool arms.

The old `CharTokenizer`, `NMindLM`, M5 checkpoints and Gate-B artifacts remain unchanged and load through their original APIs.
