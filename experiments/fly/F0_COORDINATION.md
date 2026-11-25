# FLY 线并行会话隔离记录（F0 期间）

## 冻结时刻

- 时间：2026-11-25 15:4x（Asia/Shanghai）
- 分支：`experiment/fly-mushroom-body`
- **HEAD SHA（冻结点）：`3b471fa289c05e8a797ec6b4f98236b4cf41adf5`**
- 冻结前最后一次他人提交：`072bedc`（Caishen，`experiments/fly/PREREGISTRATION.md`，15:29:21）

## 冻结时未提交文件（本次已按原样提交为冻结快照）

| 文件 | 字节 | 状态 |
|---|---:|---|
| `experiments/fly/fly_core.py` | 9,940 | 未提交实现（编码/记忆/线性注意力） |
| `experiments/fly/fly_data.py` | 7,116 | 未提交数据加载 |
| `experiments/fly/eval_sanity.py` | 5,428 | 未提交自检套件 |
| `experiments/fly/EVAL_SANITY.json` | 2,937 | **失败证据：9/10 通过，low_p_recall 失败** |

原始 `EVAL_SANITY.json` 保留不动（另存 `EVAL_SANITY_BEFORE_F0.json` 作对照副本）。

## 实验状态（冻结时）

- `D_STAGE_STATUS = STOP_BUDGET_ESTIMATE`（其预注册 §5，估算 3.1–15.6 h > 60 min 预算）。
- `eval_sanity`：**FAIL**，`low_p_recall` 最差召回 0.0（P=10/50 要求全部 1.0）。
- 自检实际配置：`M=512, sparsity=0.02 (k=10), fan_in=6, eta=0.05, decay=1.0, clip=4.0, variant=A`，10 个种子 101–1010。

## 隔离规则（本轮执行期间生效）

1. 本轮（AutoClaw / F0）**只新增** `experiments/fly/f0_netroom/` 下的文件；不修改 Caishen 的四个文件。
2. `experiments/fly/fly_core.py` 等仅在**净室证据定位到实现错误**时才允许最小修改；修改前后 diff 全量留档。
3. Caishen 会话在本轮期间不应修改 `experiments/fly/` 下同名文件；如需并行，请改用独立目录。
4. 历史失败证据（`EVAL_SANITY.json`）不删除、不覆盖、不选择性报告。

## 交付映射

- 净室：`experiments/fly/f0_netroom/{F0_MATHEMATICAL_SPEC.md, F0_PREREGISTRATION.json, f0_netroom.py, F0_*.json}`
- 报告：`reports/F0_ROOT_CAUSE_REPORT.md`、`reports/F0_EVALSANITY_BEFORE_AFTER.md`
