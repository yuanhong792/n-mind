# ROADMAP.md — 里程碑、验收条件与当前状态

> 状态标记：✅ 已验收（有 JSON 证据）｜🔶 部分完成｜⬜ 未开始。验收以 `results/` 与 `experiments/` 的真实 JSON 为准。

## M0 环境与骨架 ✅

- [x] 目录骨架、`pyproject.toml`（src 布局、`nmind` 入口、pytest 配置）
- [x] `doctor` 体检：CPU 路径（torch 2.10.0+cpu）、8 线程、AVX2、参数 282,589
- [x] 单元测试 27 项（NCell 稠密/稀疏等价、路由器、工作空间读写掩码）
- 验收命令：`python -m nmind.cli doctor` / `python -m pytest tests -q`

## M1 可训练递归网络（0.5M–2M 原型）✅

- [x] Encoder / NCell×8 块 / 路由器 / 全局工作空间 / 预测器 / 决策头 / 治理器
- [x] 极小批次过拟合验收（experiments/2026-11-21_m1_overfit.json 实验 A）：
  32 条轨迹 800 次更新，loss 3.309→0.415，训练遮挡段准确率 **0.950**，测试遮挡段 **0.941**
- [x] checkpoint 保存→重载逐位一致（demo：`save_load_bitwise_identical: true`）
- [x] CPU 推理闭环（`run`：约 1.0–5.4ms/步，SQLite 记忆检索参与决策）
- 🔶 待办：把默认配置的训练预算提高（当前 6 epoch ≈ 36 次更新，遮挡段 0.459 ≈ 多数类基线 0.444，属欠训练）

## M2 公平对照实验 🔶

- [x] GRU / 稠密 N-Cell / 共享主干三组基线 + CPU 基准跑通并记录（results/baselines.json、benchmark_cpu.json）
- [x] 发现并修复评测层三处 bug（见 STATUS"已修复"）
- ⬜ **Transformer 基线未实现**（宁缺毋滥，不许拿没验证的实现充数）
- ⬜ 严格同损失对照：当前基线 fit() 只用 CE，N-MIND 主训练是 5 项联合损失——需要把稠密 N-Cell 在 5 项损失下重训，回答"稀疏路由 vs 稠密"谁拖了后腿（今天发现稀疏版 720 步遮挡 0.61 << 稠密版 72 步 0.93）
- ⬜ 多种子重复（≥3 seeds）

## M3 动态计算（HaltingGate / 预算）🔶

- [x] governor（max_step_ms / max_recurrence_depth / 降级计数）+ evaluate 的 budget-难度对照表
- [ ] 门控训练收益验证（当前 gated 与 fixed_k1 准确率相同：门控还没学到"难任务多算"）

## M4 长期记忆与持续学习 🔶

- [x] SQLite 情景记忆（FTS5）、检索器、巩固器、墓碑删除、replay 缓冲
- [x] `train --continual`（A→B→重测 A，replay 开关）实现完成；2026-11-22 修复 B 阶段权重继承缺陷（同 seed ≠ 继承），B 阶段从 A 后权重逐位克隆并内置断言
- [x] 增量学习最小闭环 smoke（experiments/m3_incremental/）：四臂对照 1-seed，Neural-Head 未见 B 冲突步 +28.7pp、A 零遗忘；NEURAL_UPDATE_VERIFIED；RESEARCH_VALIDATED 未授予（需多 seed）
- [x] 联网自主学习链路（模拟网页→验证→经验流→调度→候选更新→封存评测→审计）打通（experiments/m4_online/）；首轮 candidate REJECTED：复位协议下 B_OOD +2.9pp<10pp、A −3.75pp>3pp；关键发现：load_bias 漂移曾伪装 ~26pp“适应”，复位后消失（详见 M4_REPORT.md）；真实联网 NOT_TESTED（默认关，待批准）
- [x] **M4.1 稳定性预注册复验**（experiments/m4_online/run_m41_stability.py，λ_anchor=0+λ_replay=0.15+48 步）：A 遗忘 −3.75→−2.66pp（≤3pp ✅ 修复生效）；B_OOD(T=32) +1.55pp 仍 <10pp ❌；**诊断修正：瓶颈是长度外推而非 λ**（同集对比锚定配置反而更好 +10.4pp vs +5.8pp；on-distribution 增益随 seed 大幅波动）。候选隔离、stable 未动。报告 M4_1_REPORT.md
- 🔶 **M5-Language（2026-11-22 第四/五/六轮，当前活跃）**：中文受限领域神经语言交互栈建成（tokenizer/七 split 数据+oracle/NMindLM 自回归/自由解码/ChatSession/三条 CLI），首轮 200 步 PIPELINE_READY_LANGUAGE_NOT_VALIDATED（free-gen EM 3.61%）。**M5.1 诊断**：截断/路由死块排除，欠训练实锤（train TF-span-EM 12.93%≈sealed 12.55%）+ exposure bias 实锤（TF 12.55%→free 0%）；修正 v1 从零阶段误用局部更新口径的设计错误（改全参，TF 0→0.5）。阶段化验证：stage0 单实体 free-gen EM 0.50（过 EM 门槛、TF 0.5 未达 0.9 整门 FAIL）、stage1 两实体绑定 0.15 FAIL（低于随机线）、工具选择头未见问法 100% PASS（首个完整通过的神经子能力）、continual 按预登记 SKIPPED。与 M5-Network（互联网即知识空间）区分：Network=按需联网/工具/验证通路，Language=自有神经网络的中文生成能力。两者共用不可信输入隔离与授权约定
- 🔶 **M5.2 中文事实绑定（2026-11-22 第六轮，当前活跃）**：平衡数据+ack 多样化+反事实对照——**S0 单实体绑定 PASS（sealed free-gen EM 1.0/TF 1.0）**；**S1 EM 0.8375 过 EM 门槛但位置交换跟随率 8.33% → BINDING_FAIL**——**近因启发式实锤（答最近陈述位置而非按查询实体绑定，sealed EM 约一半为巧合）**；S2 移动更新零样本 8.33%（未训练任务）；GRU 对照同预算 0.0%（架构优势证据）；指标勘误：entity_selection/copy 基线两无效指标作废；预算 N-MIND 谱系 1250/1250 严格用满（GRU 600 步为诊断性追加已披露）。报告 M5_2_REPORT.md。下一步：近因启发式机制修复（查询重加权/步数加倍/structured_assist 臂）待批准
- 🔶 **M4.2（规划，待批准）**：A→B→C→D 连续技能学习 + 经验回放 + 遗忘矩阵，全部任务同一权重世系；候选方案：混合时域经验流（T=24+32）攻长度外推
- 🔶 **M5-Network（任务书已下，部分建成）**：online/ 数据通路已建并接入 M4；P1-P3（CapabilityTracker/ResourceManager/SkillRegistry）待排；真实联网与候选部署须批准
- ⬜ **M6（仅接口规划）**：多模态统一输入描述与模态路由接口，不启动大规模视觉/语音/视频训练
- ⬜ 多 seed 增量学习验证 + stable/candidate 双轨回测（需用户批准预算）
- ⬜ 持续学习 A→B→A 全量运行记录（train_continual 已修复，待批准排期）

## M5 语言能力试验 🔶

- 已从 `language.source: hash`（哈希袋词特征）推进到自建字符级 tokenizer + NMindLM 自回归语言模型（452k 参数，从零训练，无预训练权重）。
- M5-Language：管线建成但语言能力未验证（free-gen EM 3.61%）；M5.1 阶段化微型验证：stage0 EM 0.50 过 EM 门槛但 TF 整门 FAIL、stage1 绑定 0.15 FAIL、工具选择 100% PASS；**M5.2：S0 单实体绑定 PASS（EM 1.0/TF 1.0）、S1 EM 0.8375 但交换跟随率 8.33% BINDING_FAIL（近因启发式实锤）、S2 更新零样本 8.33%、GRU 对照 0.0% 支持架构方向**。下一步近因修复方案待批准。
- M5-Network：互联网即知识空间，online/ 通路已建并接入 M4；P1-P3 待排，真实联网默认关。

## M6 普通设备部署 ⬜

- ONNX 导出 / INT8 量化 / 可复现性能报告。前置：M3 验收。

## 已知阻塞

1. ~~torch 为 CPU 版 → RTX 5070（Blackwell, sm_120）不可用~~ → 2026-11-21 已装 cu128 轮子启用 GPU（小袁确认；GPU 冒烟：小模型下 GPU 无速度优势，先保留给更大规模，见 STATUS GPU 状态）。
2. ~~稀疏路由版训练显著落后~~ → 2026-11-21 M2.2 10-seed 配对定案：F10_ste 0.931±0.050，C 0.877±0.087；pred 分支 legacy 缺陷已修复（pred_mode="next"）。
3. M2.2 继续 `CONDITIONAL`：H3b v1 经复核存在隐藏障碍信息与 oracle 口径错误，已用显式地图+BFS oracle 的 v2 替代；旧 checkpoint 26 维输入与 v2 的 62 维不兼容，需完成障碍课程同口径重训。H4b 只保留为反转规则分布偏移诊断。2026-11-22 起其 10-seed 全量重训标记 **DEFERRED_NOT_CANCELLED**（脚本 train_ood_v2_full.py 就绪，待用户批准预算后执行）。
4. M3 预研已启动（v1 管线 + OOD 超捷径）；Step5/6（GRU/Transformer 对照、持续学习四臂）待排。
