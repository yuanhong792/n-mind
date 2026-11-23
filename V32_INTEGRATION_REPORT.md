# N-MIND V3.2 Integration Report

## Verified in a real browser

Chrome opened http://127.0.0.1:8765/. A JavaScript syntax error in the legacy inline script was reproduced with Node syntax checking and fixed. After reload, the browser visibly showed 本地 · V3.2, real connected/offline status, and the following actions:

- 2+3 → tool, 计算结果：5。
- one practice → Oracle: verified, 来源: tool/oracle, 训练步数: 0
- create safe task → real task UUID, queued
- enable autonomy → 自主学习开关: true

## Integrity

Network requests: 0. optimizer.step: 0. Stable checkpoint: unchanged. MINLEARN-2 and Gate-B 005 were not started. No external model or API was used.

## Test results

The final full pytest run completed with 257 passed, 3 skipped, 0 failed, and 1 warning. Collection reported 260 tests. The warning is the existing PyTorch scalar-conversion warning in `tests/test_pred_next_mode.py`.

## Coverage

The 276 baseline nodeid manifest is unavailable, so the exact 19-test difference remains unresolved and is explicitly recorded in V32_TEST_COVERAGE_DIFF.json. The current manifest contains 260 collected tests; the historical 276 nodeids were not present in the workspace and were not reconstructed.

## Limitations

D2, D4, D6 and D8 are partial/not-run as recorded in V32_BROWSER_E2E_RESULTS.json. The existing UI shell still needs a full dedicated memory/growth/task navigation pass. The CUA browser trace is persisted, but no local screenshot files were available from this environment; no screenshot-only claim is made. The Playwright CLI wrapper is provided but marked NOT_RUN because the package was not available without an install step.
