# experiments/ — 实验记录索引

> 每个实验 = 一个可运行脚本 + 一个 JSON 结果。结论文字必须能追溯到 JSON 字段。

## 2026-11-21 M1 验收：过拟合健全性 + 训练预算扫描

- 脚本：`2026-11-21_m1_overfit.py`（seed 20261121，CPU，8 线程）
- 结果：`2026-11-21_m1_overfit.json`
- 结论：
  1. 实验 A（32 条轨迹、800 步）：loss 3.309→0.415，测试遮挡段准确率 **0.941**（随机 0.250）——梯度通路健全，架构可训练。
  2. 实验 B/C（96 条轨迹、240/720 步）：遮挡段停在 0.55–0.61，但记忆探针升到 **0.751**——目标信息在状态里，瓶颈在决策读出/联合损失优化。
  3. 多数动作基线 = 0.444：默认 demo 配置（36 步）的 0.459 属欠训练水平。
- 局限：单 seed；任务为 6×6 网格遮挡记忆，不外推到语言或其他域。

## 2026-11-21 基线对照（CLI 产出）

- 命令：`python -m nmind.cli baseline --json --epochs 12 --seed 20261121`
- 结果：`../results/baselines.json`
- 结论：同预算 72 步下，稠密 N-Cell（纯 CE）遮挡段 **0.933** > GRU 0.556；稀疏路由版 720 步仅 0.61 ⇒ 路由或联合损失拖累学习，待拆分验证。共享主干 29k 参数 ≥ 204k 私有参数（0.681 vs 0.663）。
- 公平性口径：基线 fit() 只用 CE；N-MIND 主训练为 5+1 项联合损失。严格对照需同损失重跑（M2 待办）。

## 2026-11-21 CPU 基准（CLI 产出）

- 命令：`python -m nmind.cli benchmark --json --seed 20261121`
- 结果：`../results/benchmark_cpu.json`
- 结论：T=48 序列，dense 35.4ms；sparse k=1 **21.1ms**（1.68×）、k=2 23.6ms、k=4 28.5ms（8 线程，AVX2，无 AVX-512）。稀疏有真实延迟收益；能耗未测量。

## 2026-11-21 GPU 冒烟（CLI 环境变更后首验）

- 脚本：`2026-11-21_gpu_smoke.py`；结果：`2026-11-21_gpu_smoke.json`
- 背景：小袁确认后安装 torch 2.10.0+cu128（download.pytorch.org 直连 ~400KB/s，改用阿里云镜像）；安装暴露并修复 `EventEncoder` 隐式 CPU 张量设备 bug。
- 结论：CUDA 可用（RTX 5070 sm_120，11.94GB，显存峰值 103MB）；同 seed 6 epoch demo **GPU 43.4s vs CPU 17.1s——当前 28 万参数规模 GPU 无优势**（kernel 启动开销主导）；GPU 训练→CPU 推理最大概率差 5.96e-08（float32 累加顺序噪声，等效逐位一致，满足"GPU 训练、CPU 推理"验收）。
- 局限：单次短训，未测 5-10M 参数与大批次场景；能耗未测量。

## 2026-11-21 任务套件（欠训练 demo 权重，仅验证管线）

- 命令：`python -m nmind.cli evaluate --json --seed 20261121`
- 结果：`../results/evaluate.json`
- 结论：状态预测/决策细分/记忆跨度曲线/预算-难度表全部出数；`gated` 与 `fixed_k1` 准确率相同 ⇒ 门控尚未学到"难任务多算"。权重为欠训练 demo checkpoint，数字只证明管线可运行，不代表能力上限。

## 2026-11-22 H3b/H4b 口径复核与 v2

- 脚本：`m2_2_validation/ood_h3b_h4b.py`；结果：`m2_2_validation/ood_h3b_h4b_v2.json`。
- H3b v1 的 oracle 使用模型不可见的障碍信息，且“BFS 可解”与曼哈顿贪心标签不一致，旧结果不进入验收。v2 使用 62 维显式障碍图、BFS 最短路标签、逐族隔离布局和内容级 manifest hash。
- 旧 26 维 checkpoint 与 H3b v2 不兼容，因此 JSON 明确记录 `not_run_incompatible_input_dim`；完成障碍课程重训前不报模型成绩。
- H4b 不再使用 `总准确率-巧合占比`。10-seed F10 的冲突步/巧合步/总准确率为 0.589/0.862/0.720；由于没有规则提示，该结果只表示反转规则分布偏移表现。另建 27 维显式规则提示的正常/反转配对数据，等待同口径重训。

## 2026-11-22 M4 增量学习首次 smoke（参数连续性修复后）

- 脚本：`m3_incremental/run_incremental_smoke.py`；结果：`m3_incremental/{lineage,parameter_deltas,learning_controls,retention_eval}.json` 与 `m4_incremental_smoke.json`；报告：`m3_incremental/m3_incremental_report.md`。
- 修复背景：`train_continual()` 的 B 阶段此前重新 build_model()（同 seed ≠ 权重继承），已改为从 A 后权重逐位克隆 + 内置断言；新增 `learning/online_adaptation.py`（fit_existing_model + 预算守卫 + 逐位取证）。
- 结论（1 预声明 seed，104/120 步，20.5s，CPU）：父 checkpoint = C_sparse_ce_s20261131（A 遮挡段 0.6594，与 M2.2 冻结记录一致）。新任务 B = reversed_control（未见测试 seed 20261142）：Neural-Head 未见冲突步 0.6019→0.8889（+28.7pp，超 +10pp 门槛且高于全部基线），A 零遗忘；Neural-Core +11.7pp 但 A −4.2pp；memory-only +2.5pp 且参数逐位不变；lr=0 反向检验逐位不变。判定：PIPELINE_FIXED / NEURAL_UPDATE_VERIFIED / NEW_SKILL_SIGNAL_SINGLE_SEED (Neural-Head) / RETENTION_PASS_SINGLE_SEED；RESEARCH_VALIDATED 不由 1-seed 授予。
- 局限：单 seed；B_before 冲突步 0.6019 偏高（父模型边界态错误与反转规则巧合），结论只称信号。

## 2026-11-22 H3b/H4b v2 全量重训（DEFERRED_NOT_CANCELLED）

- 脚本：`m2_2_validation/train_ood_v2_full.py`（已就绪未运行：10 seed × 3 臂 × 课程/无课程 × 480 步 + 闭环成功率 + 长度外推 + 配对统计）。
- 3-seed smoke（240 步）：`h3b_v2_retrain_smoke.json`（H3b 稀疏臂 ~0.5 欠训练、稠密 0.88）、`h4b_v2_retrain_smoke.json`（H4b cued：F10 0.898 > C 0.749，显式规则提示有效）。
- 状态：按 2026-11-22 纠偏指令暂停，待用户批准预算后执行（预估 ~2.5h CPU）。

## 2026-11-22 M4 联网自主神经学习全链路（首轮 REJECTED）

- 脚本：`m4_online/run_online_smoke.py`（全链路）+ `make_continuity_report.py`（P0 回归证据）；结果：`m4_online/{continuity_bug_regression,online_stream_smoke,benchmark_matrix,experience_manifest}.json`、`model_promotion_audit.jsonl`、`experiences.jsonl`；报告：`m4_online/M4_REPORT.md`。
- 链路 PASS：模拟网页规则页（control-reversed-v1）→ 解析/程序化验证 → 96 条 verified 经验 → 显式调度 → 四臂+负对照候选更新（96/120 步，48.6s，RSS 647MB）→ 封存评测 → 审计/回滚。
- 判定 REJECTED：复位协议下 B_OOD 冲突步 +2.9pp<+10pp、A 遗忘 −3.75pp>3pp；参数学习 PASS（17/19 变化、冻结 0 违规）；负对照 −45.6pp 无泄漏。**关键发现：router.load_bias 漂移曾伪装 ~26pp 适应（不复位 0.8889 / 复位 0.6244）——buffer 复位协议的定量必要性。**
- 失败诊断：λ_anchor=0.1+λ_replay=0.3 过度约束 40 步 adapter 学习；需新预注册（λ_anchor=0、λ_replay=0.15、48 步）并批准。真实联网 NOT_TESTED（默认关）。

## 2026-11-22 M4.1 稳定性预注册复验（遗忘修复、外推定位）

- 脚本：`m4_online/run_m41_stability.py`（预登记 λ_anchor=0、λ_replay=0.15、48 步）+ `make_m41_param_evidence.py`（剔除 buffer 重算参数证据）；结果：`m41_stability.json`、`m41_param_evidence.json`、`benchmark_matrix_m41.json`；报告：`M4_1_REPORT.md`。
- 判定 REJECTED（4/5）：A 遗忘 −2.66pp ✅（λ 调整修复遗忘半边）；B_OOD(T=32) +1.55pp ❌。**诊断修正：瓶颈是长度外推而非 λ**（同集对比锚定 +10.4pp vs 无锚 +5.8pp；M3 轮 +28.7pp 系另一数据 seed）。max 权重 delta 0.1116（6830 系 block_load_count 计数器累加，口径更正）。
- 待批准：混合时域经验流新预注册（T=24+32）攻长度外推。

## 2026-11-22 M5-Language 中文神经语言交互（管线就绪、能力未验证）

- 脚本：`../language/` 模块 + CLI `train-language`/`evaluate-language`/`chat`；结果：`m5_language/{train_language,eval_language}.json`、`data/`（七 split+manifest）、`chat_demo_transcript.txt`；报告：`M5_LANGUAGE_REPORT.md`。
- 判定 **PIPELINE_READY_LANGUAGE_NOT_VALIDATED**：管线全通（自由生成无 teacher forcing、无标签泄漏、会话隔离、词表冻结、buffer 复位协议），但单 seed 200 步自由生成 EM 仅 3.61%（sealed）/0~0.98%（OOD），预登记门槛未过；TF 字符准确率 74.41% 仅诊断口径。
- 规则基线在合成域接近满分——rule/神经分开计量的必要性实锤；CLI 演示保留全部错误样例。
- 修复真 bug：load_lm 缺 eval()（重载即漂移 buffer）；tokenizer UNK 被 decode 静默丢弃。
- 测试 92 项全绿（+test_language_m5.py 10 项）。
