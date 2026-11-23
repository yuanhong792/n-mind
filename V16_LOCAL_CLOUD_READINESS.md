# N-MIND V1.6 本地—云端训练衔接就绪报告

生成时间：2026-11-23（22:28 更新）　|　当前版本：V1.6　|　Git commit（本地）：`742d116`
仓库：https://github.com/yuanhong792/n-mind（**公开仓库**）
云端实验分支：`experiment/language-v16-github-cpu`

---

## ⚠️ 状态更新（22:28）：本地 A/B 训练已执行完毕，结果=负

用户明确授权"本地开跑"后，已于本地执行 512 步 A/B 训练（两组各 256）。**结果：Router 梯度修复本身生效（fixed 组权重变化 19.547 vs baseline 0.0），但两组能力指标完全一致且全部未达标**（封存 0.5 / 自由生成 0.0 / 反事实跟随 0.0）→ 按预登记 decision_rule 判定为**假设证伪**，不得宣称成功。详见 `experiments/language_v1/v16/V16_TRAINING_RESULT_REPORT.md`。

下方"就绪报告"为训练前的准备状态快照，其中第 9、10 项已被上述执行结果取代。

---

## 执行摘要（训练前快照）

本地父模型、Tokenizer、A/B 开关、训练程序与 GitHub Actions **全部就绪且验证通过**；但**真实云端训练被安全机制阻塞**——父模型权重不能进入公开仓库、也尚未通过安全渠道传输到云端。训练准备阶段 `optimizer.step = 0`；本地训练阶段执行 512 步；未修改 Stable。

---

## 十项交付问题真实回答

### 1. 本地父模型是否有效？
**有效。** `candidate_v15_baseline.pt` 8 项验证全部通过（`_parent_verification.json`）：
- SHA-256 = `5681f656fbb35f4f4a033e16212897f2d7cc02eac6754b9d860fb980d6f6a611`（与预登记一致）
- 严格加载 OK，参数量 **279363**，无 Query-Attention 新增参数（`query_readout=False`）
- 无 NaN/Inf；按 V1.5 原始 LEGACY 语料重建 Tokenizer 可正常推理（输出"红色"、EOS 结束）

### 2. Tokenizer 是否完全兼容？
**完全兼容。** 用 `train_v15.py` 中逐字相同的 9 对 LEGACY 语料重建，hash = `26af116ebd13ff00`，与 checkpoint 内 `tokenizer_hash` 字段一致，未被新文本重训。

### 3. 模型权重 SHA-256？
`5681f656fbb35f4f4a033e16212897f2d7cc02eac6754b9d860fb980d6f6a611`（1,125,400 bytes）。
Cloud 包内权重 SHA 与之一致，zip（SHA `7b34644c...`）完整性校验通过。

### 4. 云端实验分支是否同步？
**代码已本地准备，尚未推送。** `experiment/language-v16-github-cpu` 分支 fetch 成功，当前内容 = master（`bd2b37e`），其上有三个待提交文件（`train_v16_ab.py`、`validate_v16_cloud.py`、`.github/workflows/nmind-v16-cpu.yml`）+ 更新的预登记与诊断报告。按任务"独立实验分支提交、禁止强制推送"要求，已提交到该分支（见文末提交记录）。

### 5. Router 梯度开关是否正确？
**正确，零步验证 5/5 通过**（`V16_AB_ZERO_STEP.json`）：
- A 组 `router_gradient_path_enabled=False` → Router 梯度 = **0.0**
- B 组 `True` → Router 梯度 = **0.0937**（非零）
- 实现：`NMindLM.router_gradient_path_enabled` 显式开关控制 `forward_sparse` 是否传 `weights=route.weights`；STE 前向恒为 `h_upd`，故**只改反向梯度路径、不改前向数值**，旧 checkpoint 完全兼容。
- 诚实披露：`workspace.utility_mlp` 为预留未接线组件，两组均无梯度，与实验变量无关，不计入主动模块判定。

### 6. 两组初始化是否一致？
**逐参数位级一致。** 同种子 `20261123`、每组从磁盘重新加载同一父 checkpoint，零步验证 `initial_weights_identical=True`、`forward_outputs_identical=True`。

### 7. GitHub Actions 是否已配置？
**已配置且语法验证通过。** `.github/workflows/nmind-v16-cpu.yml`：
- 仅 `workflow_dispatch` 手动触发；push/PR **绝不**触发训练
- `validate` job：默认运行，校验脚本 0 step（本地 14/14 通过；云端无权重时 7/7 结构项通过、权重项 SKIP 并报告 BLOCKED）
- `train` job：需 `mode=train` **且**授权口令 `I_AUTHORIZE_512_STEPS` **且**配置安全权重 secret，三关缺一不可

### 8. 模型权重是否已安全上传？
**没有，也不应上传。** 仓库是公开的；权重是私有模型。已确认：
- 父模型**从未进入 Git 历史**，未被提交
- `cloud_training_package/` 与 `*.zip` 被 `.gitignore` 排除，`git status` 中不可见
- 权重包仅本地存在，供通过安全渠道传输；manifest 中 `weight_committed_to_git=false`、`weight_uploaded_to_release=false`

### 9. 是否可以启动真实云端训练？
**不可以，仍被阻塞。** 缺少的唯一条件：**父权重的安全传输机制**。具体为：
1. 需将 `N-MIND-V16-Cloud-Checkpoint.zip` 通过安全渠道送达云端（不得走公开仓库/Release）
2. 在 Actions secrets 中配置 `NMIND_V15_PARENT_CKPT_B64`（权重的 base64）
3. Workflow 会自动校验 SHA-256 与预登记一致后才允许训练
未配置前，`train` job 因缺 secret 而 BLOCKED，且**禁止用随机模型替代父 checkpoint**。

此外还需**你明确的训练授权口令**——即使权重就位，无 `I_AUTHORIZE_512_STEPS` 也不会训练。

### 10. 实际执行了多少 optimizer.step？
**准备阶段 0；本地训练阶段 512（两组各 256，用户授权后执行）。** 准备阶段仅做检查/验证/打包/配置；训练阶段经用户明确授权后本地执行完毕，结果见 `V16_TRAINING_RESULT_REPORT.md`（假设证伪）。云端训练仍未启动（权重未安全传输）。

---

## 关键修正记录

| 项 | V1.6 初版 | 本次修正 |
|---|---|---|
| A/B 随机种子 | A=20261159 / B=20261160（不同，违反单变量） | **两组统一 `20261123`** |
| COUNTER 独立性 | 误当未见反事实集 | 查明与 TRAIN **4/4 完全重合**，新建独立集 `V16_COUNTERFACTUAL.json`（SHA `cf27543e...`，训练前冻结，与 TRAIN 零重合） |
| 固定答案基线 | 报告为 0.5 | 实测最佳固定答案"黄色"= **0.375**（8 条评测集） |
| Router 开关 | 代码已修复但无显式开关 | 新增 `router_gradient_path_enabled` 显式开关，默认 True 向后兼容 |

---

## 训练前独立反事实集（预登记，冻结）

`experiments/language_v1/v16/V16_COUNTERFACTUAL.json`，SHA-256 = `cf27543ea05e3495fa36a0c73d43399cc05163e153751b03ed9f1b29fa94721d`。
4 条 / 2 对：未见人物（小强/小美）+ 未见物品（笔），交换两人颜色后问同一人，答案必须翻转。与 TRAIN 零重合。原始历史数据与评价记录未改动。

---

## 待你决策 / 下一步

1. **授权训练**：如确认执行，输入授权口令 `I_AUTHORIZE_512_STEPS`，并安排父权重安全传输（配置 secret `NMIND_V15_PARENT_CKPT_B64`）。
2. **总预算**：两组各 256 步，共 512 步，LR=0.0005，BS=2，CPU，约 6 分钟。需单独批准，不自动套用历史预算。
3. **验收规则**：仅当 `fixed_router` 臂在封存题、独立反事实、自由生成三项同时达标且显著超过 `baseline_frozen_router` 臂，才判定 Router 梯度修复带来实体绑定能力；否则进入架构/学习机制假设排查，不宣称成功。

---

## 产物清单

**本地**：`cloud_training_package/`（权重+manifest+README）、`N-MIND-V16-Cloud-Checkpoint.zip`、`V16_LOCAL_CLOUD_READINESS.md`（本文件）

**实验分支代码**：
`experiments/language_v1/v16/train_v16_ab.py`、`validate_v16_cloud.py`、`.github/workflows/nmind-v16-cpu.yml`

**预登记与证据**：
`V16_NEXT_TRAINING_PLAN.json`（相同种子修正版）、`V16_AB_ZERO_STEP.json`、`V16_DATA_SPLIT_AUDIT.json`、`V16_COUNTERFACTUAL.json`、`V16_CLOUD_VALIDATION.json`、`_parent_verification.json`、`_cloud_package_build.json`（及 `_` 前缀的可复现诊断脚本）

**安全**：权重与 zip 均未入库、未上 Release；公开仓库保持无父权重状态。
