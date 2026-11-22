# M4 增量学习首次 smoke 报告（m4-incremental-smoke-001）

日期：2026-11-22（AutoClaw 会话）｜任务书：`../N-MIND_87E4BBA5E78FADE4A0BCE588_BAE999_Codex8FE7BAAF%8D.md`
全部数字来自本机实际运行，证据 JSON 在本目录；判定等级严格按任务书第 7 节。

## 1. 停止了哪些旧重训任务

- H3b v2 / H4b cued v2 的 **10-seed 全量重训从未启动**（脚本 `m2_2_validation/train_ood_v2_full.py` 已写好但一次都没运行；本轮开始前经进程检查确认无任何训练进程）。
- 处置：脚本、3-seed smoke 结果、评测 JSON、报告全部保留，标记 **DEFERRED_NOT_CANCELLED**，待用户批准预算后执行；未删除任何中途成果。

## 2. 修复的根因及代码位置

- **根因**：`src/nmind/learning/trainer.py` 的 `train_continual()` 中，B 阶段调用 `train(cfg, epochs=ep, seed=seed, task=taskB, ...)`，而 `train()` 内部总是 `build_model(cfg, seed=seed, ...)`——同一 seed 只能重建**相同的初始权重**，A 阶段学到的参数从未进入 B 阶段。原返回的 `forgetting_occluded_accuracy` 实际比较的是"A 训练后模型"与"同 seed 全新初始化再学 B 的模型"，不是真实遗忘。
- **修复**（前后差异）：
  - 修复前：`rB = train(cfg, epochs=ep, seed=seed, task=taskB, ...)` → B 阶段重新初始化。
  - 修复后：A 训练 → `clone_snapshot(rA.model, rA.probe)` 快照 → 新建同结构对象并 `load_state_dict` 逐位覆盖 → `verify_bitwise_inheritance` 断言失败即 RuntimeError → `train(cfg, ..., model=b_model, probe=b_probe, seed=seed+1, task=taskB, ...)`。
  - `train()` 新增 `model:` / `probe:` 可选入参：提供时**跳过 build_model() 与权重重新初始化**；与 cfg 的 `event_numeric_dim` / `execution` / `router_ste` / `pred_mode` 不一致时**报错停止**（`init_mode="provided_model"` 如实记录）。
  - optimizer 为增量阶段**新建**（AdamW，动量不继承 A 阶段，明确记录）；训练数据采样 seed（B=seed+1）与模型初始化 seed（=seed）分开记录，不混用。
- 新增 `src/nmind/learning/online_adaptation.py`：`fit_existing_model()`（neural_adapt 唯一合法入口）、`AdaptationBudget` 硬预算守卫、逐位继承/冻结取证、transient buffer（`router.load_bias`、`block_load_count`）单独记录与复位。

## 3. 实验初始 checkpoint 与继承证据

- 父 checkpoint：`experiments/m2_ablation/C_sparse_ce_s20261131.pt`（M2 冻结稀疏基线族），state sha256 前缀 `7bb8c379c17bebb5`（完整值见 `lineage.json`）。
- lineage 吻合性交叉验证：父模型在 A 冻结测试集上的遮挡段准确率 **0.6594**，与 M2.2 定案报告记载的 C_sparse_ce 配对 10-seed 最差值 0.6594 一致（即该 checkpoint 就是那个最差 seed，未发生权重漂移或错档）。
- 每个实验臂开始前从父 checkpoint 重新加载并做逐位继承断言（`same_model_lineage` 全部通过；任一失败会直接中止实验）。
- 任务定义：A = `GridTask(goal_visible_steps=4, seed=20261131)`（与父训练同管线）；B 训练 = `reversed_control=True, seed=20261141`（96 条）；B 未见测试 = `reversed_control=True, seed=20261142`（32 条，学习阶段不可见）。

## 4. 实际权重变化清单（详见 parameter_deltas.json）

| 臂 | optimizer 步 | 允许更新参数变化 | 冻结参数违规 | N-Cell 核变化 |
|---|---:|---|---:|---|
| Control-0 | 0 | —（不更新） | — | 无 |
| Control-Memory | 0 | 0（参数逐位不变） | 0 | 无 |
| Neural-Head | 48 | **9/9** 个 adapter 参数（head/action_emb/block_to_ws）非零变化 | **0**（逐位） | 无（设计上不涉及） |
| Neural-Core | 48 | **15/15** 个开放参数非零变化，含 `blocks.W_g/W_z/W_h/W_w/b_g/b_c` 的块 5、1 切片 | **0**（逐位） | **有**（任务书要求的核心参数变化证明成立） |
| Negative-lr0 | 8 | **0**（lr=0，逐位不变，反向检验通过） | 0 | 无 |

- 梯度证据：首步梯度范数逐参数记录（grad_norms_first_step）；AdamW 对块切片组用 `weight_decay=0` + 梯度切片掩码，保证未授权块逐位不变。
- transient buffer：`load_bias` 在两个训练臂内有漂移（已记录 delta_l2），评测前复位到父值；`block_load_count` 为纯统计计数器，复位并单独记录，不计入参数变化证据。

## 5. B 新技能结果（未见测试集，冲突步口径，n=324）

B_before（父模型，旧策略）= **0.6019**；基线：测试标签多数类 **0.4321**、训练多数类迁移 0.4321、随机 0.25。

| 臂 | B_after（未见冲突步） | B_gain | 说明 |
|---|---:|---:|---|
| Control-0 | 0.6019 | 0 | 仅靠已有能力 |
| Control-Memory | 0.6265 | +2.5pp | 96 条 B 经验写入 SQLite + mem 向量注入；**参数逐位不变**，收益只能来自外部记忆（网络从未训练过读取 mem，小幅收益符合预期） |
| Neural-Head | **0.8889** | **+28.7pp** | 超 +10pp 预注册门槛且高于全部基线 → NEW_SKILL_SIGNAL_SINGLE_SEED |
| Neural-Core | 0.7191 | +11.7pp | 过 10pp 线但显著低于 Head；核心参数确有变化 |
| Negative-lr0 | 0.6019 | 0 | 反向检验：无学习收益、无参数变化 |

- 保存→重载一致性：两个神经臂的子 checkpoint 重载后 B 分数与保存前完全一致（reload_consistent=true）。
- 参考：遮挡段总准确率 B_before 0.7031（含巧合步 0.807——两规则在巧合步给出同一动作）。

## 6. A 旧技能保留（A 冻结测试集，遮挡段）

| 臂 | A_before | A_after | 遗忘 |
|---|---:|---:|---:|
| Control-0 / Memory / lr0 | 0.6594 | 0.6594 | 0 |
| Neural-Head | 0.6594 | **0.6609** | **+0.15pp（无遗忘）** |
| Neural-Core | 0.6594 | 0.6172 | −4.22pp（≤5pp 预注册门槛内，如实记录） |

## 7. 真实开销

- optimizer 步数：**104 / 120**（预注册上限内；分配 Head 48 + Core 48 + lr0 8）。
- 墙钟：**20.5s** 总计（上限 300s；护栏 240s 未触发）。单臂：Head 6.97s / Core 8.23s / lr0 1.27s。
- 设备 CPU（8 线程，torch 2.10.0+cu128 的 CPU 路径）；峰值 RSS **644.6MB**（主要为 PyTorch 运行时，< 2GB 上限）。
- 预算日志：见 `m4_incremental_smoke.json` 的 budget.log（逐臂记账）。

## 8. 通过 / 失败 / 未验证（任务书第 7 节等级）

- **PIPELINE_FIXED** ✅：A→同权重 B 继承、冻结、哈希、checkpoint 证明全部通过。
- **NEURAL_UPDATE_VERIFIED** ✅：合法参数变化发生，禁更参数逐位不变；**这不等于学会新技能**。
- **NEW_SKILL_SIGNAL_SINGLE_SEED** ✅（Neural-Head）：未见 B 冲突步 +28.7pp ≥ +10pp，高于多数类/旧策略/随机基线。**一个 seed 只能称信号**。
- **RETENTION_PASS_SINGLE_SEED** ✅（Neural-Head 零遗忘；Neural-Core −4.2pp 在 5pp 门槛内）。
- **RESEARCH_VALIDATED** ❌ 未授予：需独立新 seed、置信区间、OOD 迁移与预算公平对照后再评估。
- 诚实备注：B_before 冲突步 0.6019 偏高，原因是父模型在边界态的动作错误与反转规则标签巧合较多（其 A 准确率本身 0.6594）；预注册门槛与基线要求仍然全部满足，但任务区分度有限，结论只称信号。

## 9. 测试与 Git 提交

- 回归测试：**72 项全绿**（64 项既有 + `tests/test_true_continual.py` 8 项新断言：继承逐位 / 显式模型优先且冲突报错 / 允许参数变化且冻结逐位不变 / lr=0 与全冻结逐位不变 / 各臂同父快照 / reversed_control 有效冲突样本 / 子 checkpoint 往返一致 + 预算中止不触 stable / fresh_build 路径不变）。实际运行，非引用历史数字。
- Git 提交：本轮修复 + 实验 + 报告见提交 `（见仓库 log，m4 提交）`；H3b/H4b v2 脚本层以 DEFERRED 状态单独提交。
- 证据文件：`lineage.json`、`parameter_deltas.json`、`learning_controls.json`、`retention_eval.json`、`m4_incremental_smoke.json`（全部实际生成；无占位造假）。子 checkpoint 在 `ckpts/`（candidate 轨道；stable 未建、未触碰）。

## 10. 仍需用户批准的训练项目（未自行启动）

1. **H3b/H4b v2 10-seed 全量重训**（`train_ood_v2_full.py`，M2.2 收尾）：10 seed × 3 臂 × 课程/无课程 × 480 步 + 闭环成功率 + 长度外推，预估 ~2.5h CPU。收益假设：闭合 M2.2 的 H3b/H4b 模型层欠账，决定 M2.2 能否 CONDITIONAL→COMPLETE。
2. **多 seed 增量学习验证**：5-10 独立 seed × 同四臂（每 seed ~21s），预估 ~15 分钟。收益假设：把 NEW_SKILL_SIGNAL 升级为带置信区间的可复现结论。
3. **持续学习 A→B→A 全量运行**（修复后的 `train_continual`，含 replay 开关对照）：预估 ~30 分钟。收益假设：获得真实遗忘量基线，替代旧无效口径。
4. M3 Step5/6（同预算 GRU/Transformer 对照、持续学习四臂全量）维持排队，等前两项结论。
