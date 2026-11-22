# STATUS.md — 当前真实进度

更新：2026-11-22（AutoClaw 会话）。所有数字来自本机实际运行，证据在 `results/` 与 `experiments/`。

## 环境

- Windows 11，Python 3.12.10，torch **2.10.0+cu128**（2026-11-21 15:44 由 CPU 版升级，阿里云镜像），pytest 9.2.0，包已 `pip install -e .`
- i7-14700KF（28 逻辑核，AVX2，无 AVX-512）/ 32GB RAM / **RTX 5070 12GB 已启用**（sm_120，驱动 596.36，doctor 报设备 cuda）
- 云端 `mode: off`，开发全程零网络调用（pip 安装除外）
- Git：2026-11-21 初始化；**注意可能有并行会话（Codex）在同一仓库工作，提交时只 add 自己的文件**

## 可复现命令

```powershell
# M1 过拟合验收（约 2 分钟）
python experiments/2026-11-21_m1_overfit.py
# 基线与基准（证据：results/baselines.json、benchmark_cpu.json、evaluate.json）
python -m nmind.cli baseline --json --epochs 12 --seed 20261121
python -m nmind.cli benchmark --json --seed 20261121
python -m pytest tests -q          # 27 passed
```

## M4 增量学习：参数连续性修复与首次四臂 smoke（2026-11-22，AutoClaw）

- **P0 bug 修复（实锤并闭环）**：`train_continual()` 的 B 阶段此前调用 train() 重新 build_model()——同一 seed 只重建相同初始权重，A 阶段学到的参数从未进入 B，forgetting 指标无法解释为真实遗忘。修复：`train()` 新增 `model=`/`probe=` 显式入参（提供时跳过 build_model，语义冲突直接报错不静默重建）；`train_continual()` B 阶段改为从 A 后权重逐位克隆 + 内置继承断言；optimizer 增量阶段新建并记录；采样 seed 与初始化 seed 分开记录。
- **新增** `src/nmind/learning/online_adaptation.py`：`fit_existing_model`（neural_adapt 唯一合法入口）+ AdaptationBudget 硬预算守卫 + 逐位继承/冻结取证（torch.equal，不用容差）+ transient buffer（load_bias/block_load_count）单独记录与复位。
- **首次四臂增量学习 smoke**（experiments/m3_incremental/，1 预声明 seed，104/120 步，20.5s，RSS 645MB）：父 checkpoint = C_sparse_ce_s20261131（A 遮挡段 0.6594，与 M2.2 冻结记录一致）。新任务 B = reversed_control（训练 96 条 / 未见测试 32 条，seed 20261141/42）。结果：Neural-Head 未见 B 冲突步 0.6019→**0.8889（+28.7pp）**、A 0.6594→0.6609（零遗忘）；Neural-Core 0.7191（+11.7pp）但 A −4.2pp（门槛内）；memory-only +2.5pp 且参数逐位不变；lr=0 对照逐位不变。判定：PIPELINE_FIXED / NEURAL_UPDATE_VERIFIED / NEW_SKILL_SIGNAL_SINGLE_SEED (Neural-Head) / RETENTION_PASS_SINGLE_SEED；RESEARCH_VALIDATED 不由 1-seed 授予。报告：experiments/m3_incremental/m3_incremental_report.md。
- 测试 **72 项全绿**（64 项既有 + tests/test_true_continual.py 8 项新断言）。
- H3b/H4b v2 的 10-seed 全量重训 **DEFERRED_NOT_CANCELLED**（脚本 train_ood_v2_full.py 就绪未运行，约 2.5h CPU，待小袁批准）。

## M4 联网自主神经学习：全链路打通 + 首轮 REJECTED（2026-11-22 第二轮，AutoClaw）

- **链路 PASS**：src/nmind/online/（fetcher/fixtures/extractor/provenance/verifier/experiences/scheduler/promotion/explorer）+ learning/online_trainer.py（有界流更新：CE + λ_replay·CE_replay + λ_anchor·锚定）+ evaluation/lifelong.py。模拟网页夹具发布结构化规则 control-reversed-v1 → 解析/程序化验证 → 96 条 verified 经验流 → 调度触发候选更新（四臂+负对照，96/120 步，48.6s，RSS 647MB）→ 封存评测 → 晋升审计。
- **判定 candidate_status=REJECTED**：B_OOD 冲突步 gain +2.9pp < +10pp、A 遗忘 −3.75pp > 3pp；其余门槛过（继承逐位/冻结 0 违规/负对照 −45.6pp 无泄漏/neural>frozen）。候选隔离 ckpts/rejected/，stable 逐位未动。参数学习本身 PASS（17/19 非零变化）＝NEURAL_UPDATE_VERIFIED；可泛化技能 FAIL；真实联网 NOT_TESTED（门槛未过不触网）。
- **关键发现**：训练态 router.load_bias 漂移承载 ~26pp 的“表面适应”——不复位时 B_OOD 0.8889，复位后 0.6244。buffer 复位协议（任务书要求）防止把 buffer 漂移冒充参数学习，本轮给出定量证据。
- **失败诊断**：λ_anchor=0.1+λ_replay=0.3 在 40 步内过度约束 adapter（对照：上轮无锚定 48 步单 seed +28.7pp）；Hybrid 证明锚定+回放保 A（−0.31pp）但代价是 B 几乎不学。修复需新预注册并批准。
- **安全**：恶意网页夹具 instruction_detected→rejected→不进学习流（测试断言零执行/零改配置）；dedup 生效；sealed 零泄漏；offline 硬断。configs/online_learning.yaml 三开关默认关（network/auto_weight_update/promotion=manual）。
- 测试 **82 项全绿**（+tests/test_online_learning.py 10 项）。报告：experiments/m4_online/M4_REPORT.md。

## M4.1 稳定性预注册复验：遗忘门槛修复、外推瓶颈定位（2026-11-22 第三轮，AutoClaw）

- 预登记（小袁指令）：λ_anchor=0、λ_replay=0.15、48 步、1 seed（m4.1-stability-001）。结果 **REJECTED（4/5 门槛过）**：A 遗忘 −3.75→**−2.66pp ✅**（回放防遗忘起效）；B_OOD(T=32) **+1.55pp ❌**。参数证据：9/19 变化、max 权重 delta 0.1116、冻结逐位 0 违规（训练时+保存权重双重核验）；6830 是 block_load_count 计数器 buffer 累加，已剔除重算（口径更正）。
- **诊断修正（诚实）**：M4 报告“λ 过约束”不是 OOD 失败主因——同数据集上锚定配置 B_test 反而更好（+10.4pp vs +5.8pp）；B_OOD 在所有已测配置下都不达标（+2.9/+1.55pp）。真正瓶颈 = **T=32 长度外推**（训练时域外的递归状态分布）；run 间方差大，单 seed λ 归因不可靠。
- 报告：experiments/m4_online/M4_1_REPORT.md；参数证据：m41_param_evidence.json。
- 路线（小袁 2026-11-22）：M4.2=A→B→C→D 连续技能学习+遗忘矩阵（规划）；M5=互联网即知识空间（任务书已下，P0/P1/P2/P3 优先）；M6=多模态仅接口规划。

## M5-Language 中文神经语言交互（2026-11-22 第四轮，AutoClaw）

- **目标与边界**：自有随机初始化神经网络的受限领域中文理解/多轮状态/自由生成；禁止预训练权重/教师/API；rule_baseline 单独命名不计入神经成绩。
- **已实现并测试**：src/nmind/language/（tokenizer 词表冻结+UNK 不再静默丢弃 / dialogue_data 七 split 预登记+独立 oracle / NMindLM=NCell+Router+Workspace 自回归+452k 参数 / decoder 自由解码无标签泄漏 / ChatSession 跨轮状态+工具路由+response_source / train-language・evaluate-language・chat 三条 CLI）；数据 manifest（seed/sha/逐意图多数基线）落盘。
- **实测**：单 seed 200/200 步 154.4s，TF 字符准确率 74.41%（sealed）但 free-gen EM 仅 3.61%（sealed）/0~0.98%（四 OOD）——**判定 PIPELINE_READY_LANGUAGE_NOT_VALIDATED**（IN_DOMAIN/OOD 门槛均未过）；规则基线在合成域接近满分，反证了 rule/神经分开计量的必要性；CLI 演示错误全部保留（chat_demo_transcript.txt）。
- **失败分析**：200 步只够学会对话结构（回声+模板片段），世界追踪/工具路由未涌现；修复了 load_lm 缺 eval() 导致重载即漂移 buffer 的真 bug（test_9 抓住）。
- 与 **M5-Network（互联网即知识空间，2026-11-22 第二三轮已建 online/ 数据通路与 M4 集成）** 的关系：两者共用不可信输入隔离/授权开关/来源审计约定；M5-Network 的 P1-P3（CapabilityTracker/ResourceManager/SkillRegistry）被本指令接替尚未开发。
- 测试 **92 项全绿**（+tests/test_language_m5.py 10 项）。报告：experiments/m5_language/M5_LANGUAGE_REPORT.md。

## M5.1 中文语言理解修复阶段：P0 诊断 + 阶段化微型验证（2026-11-22 第五轮，AutoClaw）

- **P0 诊断（无需重训，diagnose_m5.json）四假设裁定**：①训练截断率 0 → 排除；②过拟合 → 排除（train TF-span-EM 12.93% ≈ sealed 12.55%，连训练集都没背下来 ⇒ **欠训练**）；③exposure bias → **实锤（次因）**（sealed TF-span-EM 12.55% → free-gen 0%）；④路由死块/机制 bug → 排除（8 块负载 0.08–0.15 无死块，TF 条件下部分 span 全对 ⇒ 梯度/状态通路正常）。停止条件正常（eos 82 / newline 15 / max_tokens 4），回声率 6.9%。
- **v1 设计错误（诚实记录并修正）**：从零阶段误用 M4 的"head+block_to_ws 局部更新"口径（随机主干 + 只训输出头），4 选 1 查找在数学上不可学，stage0 TF=0.0。v2 改为**从零阶段全参训练**，M4 局部更新口径只用于持续学习（父=已训练模型）；修正后 stage0 TF 0.0→0.5、free-gen EM 0→0.5。
- **机制改进（诊断驱动）**：答案掩码损失（CE 只计答案 span 含 EOS，用户轮与"助手："前缀零梯度）；等长批量答案掩码前向，v1 逐样本 257s → v2 批量 42s（6×）。N-Cell/Router/Workspace 核心架构未改。
- **阶段结果（m51_stages.json，全部实测，样例逐条保留）**：

| 阶段 | 步数 | free-gen EM（sealed） | TF-span-EM | 门槛 |
|---|---:|---:|---:|---|
| Stage 0 单实体复制 | 200 | **0.50** | 0.50 | EM 过（多数基线 0.25+10pp）/ TF 未达 0.9 → 整门 FAIL |
| Stage 1 两实体绑定 | 200 | **0.15** | 0.15 | FAIL（低于 0.25 随机线） |
| Stage T 工具选择（未见问法） | 100 | — | — | **PASS（1.00 ≥ 0.85，45/45）** |
| 持续学习 v0→v1 | 48 | — | — | SKIPPED（stage0 整门未过，按预登记不继承） |

- **判定**：管线与机制无 bug；瓶颈是"内容绑定 vs 训练预算"的缩放——200 步学会格式与工具意图，学不会事实绑定。stage0 是首个自由生成正确答案（EM 0.50）的微型任务；工具选择头是**首个完整通过的神经子能力**。错误样例：结构完美、内容绑定失败（`小明在1号`→`2号`；`小红在4号`→`1号`）。
- **资源**：各阶段独立预算均 <300s，总墙钟 157.2s，参数量 452,116，CPU 8 线程。
- **下一步（需批准，未自行启动）**：预登记 stage0/1 各 600–1000 步（约 2–5 分钟/阶段）验绑定能否跨过 35%；stage0 错误偏向"2号位置"，下一轮训练集需位置均衡采样校验；exposure bias 待绑定学会后复测。报告：experiments/m5_language/M5_1_REPORT.md。测试 **92 项全绿**。

## M5.2 中文事实绑定与泛化（2026-11-22 第五轮，AutoClaw）

- **P0 审计**："2号位置"先验非数据偏置（46/51/54/49 基本均衡）；冻结 M5.1 checkpoint 反事实探针位置跟随率 0%（全部生成 ack 模板）——**"陈述→好的。"确认语捷径是全对话 EM 3.61% 的机制性原因**。
- **数据修正**：位置精确均衡轮转 + ack 三选一化+30% 无确认 + 12 组合轮转绑定 + 反事实对照对 + S2 更新探针；预登记训练前落盘（m52_pre_registration.json）。
- **实测（预算 N-MIND 谱系 1250/1250 严格用满）**：过拟合自检 PASS（校准历史三次全披露：EM 判据两次误标定→改相对判据 8× 未训练基线）；**S0 单实体绑定 PASS（sealed free-gen EM 1.0 / TF 1.0）**；**S1 两实体绑定 FAIL——sealed EM 0.8375 过 EM 门槛，但位置交换跟随率 0.0833**（24 对仅 2 对）——**近因启发式实锤：模型答"最近陈述的位置"而非按查询实体绑定**（错误样例：事实在第一句时答第二句的位置）；S2 移动更新零样本 8.33%（未训练任务，真实缺口）；GRU 对照同预算 EM 0.0%（NMindLM 架构优势证据，GRU 600 步为诊断性追加计算已披露）。
- **指标勘误**：entity_selection_acc（答案含实体名恒 1.0）与 last_statement_copy_acc（or True 笔误）作废，以交换跟随率实测取代（m52_follow_eval.json 逐条记录）。
- 测试 92 全绿；提交 <见 log>。报告：experiments/m5_language/M5_2_REPORT.md；数据：m52_diagnostics/m52_pre_registration/m52_eval/m52_follow_eval.json。
- 待批准：① 近因启发式机制修复（查询重加权/步数加倍/structured_assist 对照臂三选一或组合）；② S2 事实更新训练；③ M4.2 遗忘矩阵；④ M4.1 混合时域。

### 1. M1 过拟合 + 训练预算扫描（experiments/2026-11-21_m1_overfit.json，seed 20261121）

| 实验 | 数据 | 梯度步 | loss 末值 | 测试遮挡段 acc | 探针 acc（随机≈0.028） |
|---|---|---|---|---|---|
| A 过拟合 | 32 条 | 800 | 0.415 | **0.941** | 0.79（train） |
| B 默认数据 | 96 条 | 240 | 1.221 | 0.603 | 0.285 |
| C 加长 | 96 条 | 720 | 0.670 | 0.609（平台 0.55–0.61） | **0.751** |
| 多数动作基线 | — | — | — | 0.444 | — |

结论：梯度通路健全，架构能学（A 证明）；B/C 显示**状态里记住了目标（探针 0.75）但决策读出跟不上（0.61）**——瓶颈在读出/优化，不是记忆容量。

### 2. 基线对照（results/baselines.json，72 步同预算同数据）

| 主干 | 参数 | acc | 遮挡段 acc | 单序列 ms |
|---|---|---|---|---|
| dense_n_cell（NModel 关路由，纯 CE） | 282,589 | 0.857 | **0.933** | 152.0 |
| gru_2layer | 54,248 | 0.530 | 0.556 | 2.2 |
| shared private_only | 204,008 | 0.633 | 0.663 | 11.1 |
| shared mixed | 129,128 | 0.624 | 0.650 | 10.0 |
| shared shared_only | 29,288 | 0.651 | 0.681 | 5.5 |

诚实结论：
- 同架构**稠密**版 72 步就到 0.933，而稀疏路由版 720 步只有 0.61 ⇒ **当前稀疏路由/联合损失训练配置拖累了学习**，N-MIND 尚未证明优于自己的稠密消融，更没证明优于 GRU 的性价比。这是 M2 最优先待解问题（公平口径：基线只用 CE 损失，需在 5 项损失下重跑稠密版对照）。
- 共享主干省参数不降准确率（29k 参数 > 204k 参数的遮挡段 acc）。

### 3. CPU 基准（results/benchmark_cpu.json，T=48，60 次，8 线程）

dense 35.4ms/序列（P95 51.2）；sparse k=1 **21.1ms**（1.68×）；k=2 23.6ms；k=4 28.5ms。
稀疏路由在 CPU 上有真实延迟收益，但准确率收益未证实（见上）。

## Agent Runtime R1 独立 Bot 自主学习（2026-11-22 第六轮，Qoder）

- **交付**：`src/nmind/bot/` 12 个模块（状态机/调度/目标/能力台账/练习环境/经验桥/学习环/晋升管理/策略/存储/状态快照/CLI）+ `configs/bot_runtime.yaml`、`bot_demo.yaml`、`bot_safe-local.yaml`、`bot_candidate-demo.yaml` + `tests/test_bot_runtime_r1.py`（16 项）+ `experiments/bot_runtime_r1/`（九份证据 + `reproduce.ps1` + `BOT_R1_REPORT.md`）。stable 源权重 `experiments/m5_language/m52_s0_checkpoint.pt` 只被复制，绝不被写。
- **无人值守闭环 PASS**：20 轮 ×3 实例（arithmetic/grid_rules/json_extract）全程零用户输入；累计 81 实例、81 唯一指纹、0 条封存样本进学习流；用户消息在实例边界真抢占（不丢队列、按序回答）；`pause` 冻住周期 22→22、`resume` 先答消息再继续→23、`stop` rc=0 且锁清理；重启后 23→26 轮、新增 18 个唯一实例、0 重复、SQLite 完好。
- **资源实测**：CPU 占整机 **42.5 %**（政策 50 %；torch 线程由代码 20→13 夹紧），RSS 峰值 **216.2 MB**（上限 2048 MB），Phase A 墙钟 7.44 s/20 轮，候选训练轮 37.9 s（上限 60 s 跨臂）。
- **神经增长判定 REJECTED**：显式授权 48 步全部用满（neural 24 + negative 对照 24），封存集自由生成 EM **0.0 → 0.0**，预登记门槛 +0.10 未达；17/23 张可训练张量真变化（max_abs_delta 0.0735）、buffer 复位取证 `reset_applied=true`、候选被隔离、stable sha 逐位不变、无 `--i-approve` 的晋升被拒。安全探针 **11/11 PASS**。结论：**闭环与防自欺机制可靠，能力没有增长**——三类任务的神经基线本身就是 0（M5.2 权重对这些题面输出无关句子，如 `请计算：11加18等于多少？`→`小明在2号位置。`），54/81 的 verified 全部来自确定性工具并单列计量，不计入神经分数。
- **端到端实验抓出并修掉的真实缺陷（8 项）**：① 共享预算把 negative 对照臂饿死成 0 步（四臂对照名存实亡）；② 参数取证混入 `block_load_count` 计数（max_abs_delta 9472 假象，M4 虚假增益同款口径漏洞）；③ `PAUSED → IDLE_EXPLORE` 非法转移使常驻进程 resume 后崩溃；④ §9.2 用户抢占此前并未真的实装；⑤ json_extract 题面与封存集同题造成矛盾监督（§9.6 FAIL）；⑥ 日志/事件无容量上限（§9.12）；⑦ `READY → CANDIDATE_TRAINING` 非法转移使 `bot candidate` 直接报错；⑧ 帮助文本承诺的 `--profile safe-local`/`candidate-demo` 缺配置文件（照文档敲就 SystemExit）。
- 测试 **108 项全绿**（修完 CLI 全局开关位置 bug 后重跑 53.29 s，含 M4/M5 回归未破坏；本任务 +16：§9.1–§9.15 + §9.16 实验回归 + profile/开关可执行性断言）。`reproduce.ps1` 修复后端到端重跑退出码 0、安全探针 11/11 PASS。
- **未做**：P4 真实联网（默认关闭，需单独授权）、P5 开机自启/GUI 面板/exe 打包；grid_rules 仍没有程序 oracle 工具路径（该族 27/27 rejected，tool 侧记 null）。
- **环境实测提醒**：本会话解释器 `torch 2.10.0+cpu`（本文上方 11-21 记录的 cu128 + RTX 5070 与本轮实测不一致）；R1 全程 CPU、零网络、零外部模型 API。

## Agent Runtime R2：任务-能力对齐审计与技能级练习改造（2026-11-22 第七轮）

- **P0 审计先行**：`docs/bot_r2_code_map.md` 在任何 R2 代码改动前落盘（基线 commit `291a263`，工作树干净），确认 R1 神经零增长的机制是**任务分布与词表不匹配**——stable checkpoint（m52_s0）词表仅 31 token（中文绑定域），R1 三族题面（算术/JSON/网格规则）字符集与之几乎不相交，每字符退化为 UNK；验证器把"工具算对"当"神经学会"的监督目标；台账把工具/规则/神经成功混记同一族。结论：不是训练不足，是"喂错了"。
- **新增四模块**：`language/binding_data.py`（A/B/C 绑定任务数据层，A 组模板与 M5.2 `run_m52` 同源逐字节一致）；`bot/task_capability_bridge.py`（TaskSpec 统一视图 + `run_gate_a`）；`bot/skill_registry.py`（SkillSpec + 与桥接/选择器共享归因词表）；`bot/curriculum.py`（SkillSelector 技能级可学资格评分，与 GoalManager 族级选择分工，各自写 selection_source）。
- **Gate-A（只测零训练）判定 `TASK_CAPABILITY_ALIGNED`**（`experiments/bot_runtime_r2/` 三工件 + `results/bot_r2/` 运行时镜像）：A 组复现 sealed free-gen EM=**1.0**（神经基线可用，Gate-B 抗遗忘锚点成立）；`unseen_entity_qa` 缺口 100pp（EM=0.0）且有独立 oracle、学习流/未见测试集输入指纹互斥 → 可进 Gate-B；`latest_statement_update` 被选择器拦截（模板空间仅 4 个不同输入，无法区分泛化与背诵）；R1 三族改按 TOOL 记账；C 组仅诊断、未经单独许可不训练。
- **旧实现收口（R2 必须改，非重写）**：`capability_tracker` 升六级口径 + response_source/feedback_source/learning_target/capability_state 四列归因（词表校验，缺一按最保守记账）；`experience_bridge.to_learning_rows` 代码级拦截（只收 `learning_target=neural_generation`，返回 skipped 明细）；`learning_loop` 空学习流 `NO_NEURAL_ELIGIBLE_EXPERIENCE` 如实报因；`practice.py` 新增 `grid_answer()` 独立可复算（verify 重算比对，不再信自报答案）+ 反馈四态词表；`runtime.py` 技能练习循环（Gate-A 产物只读加载、缺口驱动、轮换游标持久化）；`store.py` capabilities 表五列 ALTER 迁移（旧行可读）；`decoder.py` 新增 `free_generation()` 统一自由生成口径（收敛 neural_attempt/free_gen_em/eval_free 三处重复实现）。
- **测试 108 项全绿**（R1 回归 16 项适配新签名，显式指向不存在的矩阵路径做隔离；本机实测 2026-11-22）。
- **清理**：删除并行会话遗留 `chat_session.py`（幻觉产物：引用 4 个不存在的类，import 必崩，与 `language/session.py` 正式实现重复）、`cli.py.bak-20261122-parallelstub`、`__tmp/`、根目录 `_smoke_r2.py`。
- **未做**：`online/resource_manager.py`（§7 五档链路）未建；`bot skills/capability-report/learn-status/curriculum/eval` 五个子命令未注册；Gate-B（对 unseen_entity 受控候选训练）需预登记预算待批准；`fetcher.py` 传输硬化未动。

## Gate-B 001：首个受控候选训练 REJECTED + 遗忘指标真 bug 修复（2026-11-22 第八轮）

- **预登记先行**（experiments/bot_runtime_r2/gateb_pre_registration.json，用户批准后执行）：目标 skill.binding.unseen_entity（Gate-A 唯一全资格技能，EM 0.0/缺口 100pp/位置轴对半切指纹互斥），预算 96 梯度步跨臂硬总闸（neural 48 + negative 48）、lr=3e-3、trainable="all"、replay_ratio=0.15、A 组回放 48 行 + A 组封存 80 行做抗遗忘锚点、seed 20261601/20261602。
- **实测（22.5s，budget 96/96 用满，stable 逐位未动）**：训练行 loss 2.05→0.19（背题成功）但 **sealed（位置 3/4 号，未见）EM 0.0→0.0，gain=0.0 gate_1 FAIL**；负对照臂 0.0；memory 臂 abstain 40/40（符合"背题≠能力"口径）；per-position 分解 3/4 号位均为 0.0——模型过拟合到训练位置组合（8 个唯一输入），绑定未跨位置泛化。
- **抓出并修复 learning_loop 遗忘指标真 bug（R2 §4 修复项的完成）**：旧口径 base_ret 用 retention_pairs 但 old_after 仍测训练行前 8 条——本轮记忆化 +50pp 给出假阳性 PASS。从 rejected 候选补测真实 A 组保持率：**EM 1.0→0.6875 = −31.25pp，灾难性遗忘实锤**（远超 −3pp 门槛；M4.1 网格任务 −2.66pp 的回放保护在语言任务全参 48 步下失效）。修复：old_task_forgetting_pp 改为 retention 同口径，训练行前 8 条改列 train_rows_memorization 诊断不参与闸判定；修复后测试 **108 项全绿**。
- **判定 REJECTED（两项实fail）**：泛化 0 + 真遗忘 −31.25pp。候选隔离 candidates/rejected/，stable 逐位不变。按预登记 no_retry 条款不追加预算重跑；下一轮（若做）必须新预登记，候选方向：trainable 收窄（head/局部更新，R1 smoke 成功口径）、回放加强、lr 降档、步数结构调整——需用户批准。
- 结果：experiments/bot_runtime_r2/gateb_result_001.json（含 metric_caveat 勘误附注）+ gateb001_retention_diagnosis.json；results/bot_r2/ 镜像。

## Gate-B 002 单变量消融 REJECTED：遗忘定位到共享输出头（2026-11-22 第九轮）

- **审计先行**（用户要求，五项程序化审计全 PASS 后才执行，gateb002_consistency_audit.json）：父 checkpoint/训练集/封存集（实跑指纹逐字节）/seed/lr/回放/总步数/评测协议与 001 一致；唯一变量确认 trainable all→adapter（数值字段白名单 diff 仅此一项）；adapter 范围=head+block_to_ws 4 张量（19 张量冻结）；buffer 复位协议 TRANSIENT_BUFFERS=[block_load_count, router.load_bias] 干跑取证；001 防重放闸 exit 4。002 预登记补齐对齐字段（entity/train_sealed_overlap_rate/wall_limit_s_per_arm）。
- **实测（96/96 步用满，22.95s，stable 逐位未动）**：sealed（未见位置）EM 0.0→0.0 gain=0 gate_1 FAIL；**A 组遗忘 −100pp（1.0→0.0，retention 修复后原生口径）**gate_2 FAIL——比 001 全参（−31.25pp）更糟；训练行记忆化 0.0（连题都没背下，loss 2.05→0.70）；负对照 0.0；**冻结违规 0**（19/19 范围外张量逐位不变，审计③取证落实）。
- **配对对照报告**（gateb_paired_report.json）：反事实绑定跟随率 stable 0.25 → 001 0.2917 → 002 0.0833（近因捷径率全程 0——不是走捷径而是输出分布被改写）。
- **失败归因**（gateb002_failure_attribution.json）：假设"收窄可训练范围防遗忘"被推翻——遗忘位置在**共享输出头**：新实体分布与旧实体分布在同一 head.weight 直接竞争，R1 网格任务 Neural-Head 成功先例（动作空间小、任务同构）不适用于语言任务全词表输出。两轮共同结论：问题不在可训练范围选择，在架构层。
- **实现缺口如实登记**：预登记 6 闸中 gate_3（负对照不反超）未代码化进 gates 判定；反事实跟随率评测未进主流程（本轮为报告阶段独立评测）。
- **候选方向（需用户批准，未启动 003）**：a) A+B 混合课程（改变增量语义需任务书确认）；b) replay 加权 0.15→0.5+；c) 架构方案 per-task 输出头/输出残差（M6 级）；d) 接受证据暂停 003。禁止自动晋升：双双 REJECTED 隔离于 candidates/rejected/。

## Gate-B 003 冻结输出头实验 REJECTED：遗忘是分布式的（2026-11-22 第十轮）

- **定位性中间实验**（用户批准，002 归因的收窄探针）：单变量消融链第三格 001=all(23张量) → 002=adapter(head+block_to_ws,4张量) → 003=adapter(block_to_ws only,2张量,head冻结)。审计 PASS（gateb003_consistency_audit.json：唯一变量 adapter_prefixes，数据指纹逐字节一致，门槛数值一致，gates 文案差异均为设计内升级）。
- **实现改动**：set_trainable/masked_stream_update/run_candidate_update 暴露 adapter_prefixes 可选参数（默认行为不变），run_gate_b.py 从预登记读取；测试 108 项全绿。
- **实测（96/96 步，21.83s，stable 逐位未动，冻结违规 0——21/21 范围外张量逐位不变）**：sealed EM 0.0→0.0（泛化连续三轮为 0）；**A 组遗忘 −82.5pp（1.0→0.175）**——head 冻结只挽回 17.5pp（002 −100 → 003 −82.5）；训练行记忆化 0.0（loss 2.05→1.37，2 张量装不下新绑定）；负对照 0.0。
- **三轮消融链结论（gateb_paired_report.json 已更新为三轮版）**：遗忘是**分布式**的——head 贡献 ~17.5pp、block_to_ws 承载 ~82.5pp，任何单组件的序列更新都破坏 A 组绑定表示；容量递减方向（23→4→2 张量）只会"学得更少忘得更少但不学"（记忆化 0.5→0→0）。**结构性结论：当前 NMindLM（共享 workspace 写入 + 共享 head）下，序列式单任务增量更新必然破坏旧表示，隔离式持续学习的三条红线挡得住静默重建、挡不住表示冲突本身——增量学习三轮 REJECTED 需架构级方案。**
- **003 之后的方向（需用户批准，未启动）**：a) A+B 混合课程（验证冲突是否来自序列范式本身）；b) replay 加权 0.5+（序列语义内最后一搏）；c) 架构级 per-task 组件/输出残差（M6 级）；d) 暂停归档转投 M5 基线。反事实跟随率三轮配对报告单列（stable 0.25 / 001 0.2917 / 002 0.0833 / 003 0.0833）。

## 本次会话修复的 bug（评测层此前从未成功运行过）

1. `evaluation/{baselines,benchmarks,tasks}.py` 循环导入（`from . import datasets` → `..learning`）——`baseline`/`benchmark`/`evaluate` 三条命令修复前必崩。
2. `benchmarks.cpu_benchmark` repeats 超过测试集数量时索引越界 → 改为按 `i % len(rows)` 循环。
3. `_submodule_ms` 编码器计时喂错张量（z 64 维当 26 维观测）。
4. `Timing.as_dict` 缺 `per_event_ms`；CLI 引用不存在的 `peak_rss_mb` 键。
5. `baselines.fit/score`：NModel 返回 `action_logits`、GRU 返回 `logits`，键名自适应。
6. `SharedTrunkCell` 共享槽位输入未复制导致 GRUCell 批大小不匹配（16 vs 64）。
7. `EventEncoder` 在 text_feat/numeric 缺省时造隐式 CPU 零张量 → GPU 训练崩；已改为跟随输入设备（GPU 冒烟实验暴露）。

## GPU 状态（2026-11-21 启用，小袁确认安装）

- torch 2.10.0+cu128 落位（RTX 5070 需 cu128+；download.pytorch.org 直连 ~400KB/s 卡慢，改用阿里云镜像 `--find-links` 完成安装）
- GPU 冒烟实验（experiments/2026-11-21_gpu_smoke.json）：同 seed 6 epoch demo，GPU 43.4s vs CPU 17.1s，**小模型下 GPU 无优势**（kernel 启动开销主导，显存峰值仅 103MB）；GPU 训练→CPU 推理最大概率差 5.96e-08（等效逐位一致）
- 测试 32 项全绿（含 test_datasets.py 5 项——并行会话新增）
- 结论：GPU 先保留用于更大模型/更大批次，当前 M2 对比实验继续以 CPU 口径为准（两边数字都来自实测，不混用口径）

## 阻塞

- ~~GPU：torch CPU 版无法用 RTX 5070~~ → 已解决（cu128 已装，见上方 GPU 状态）
- 未实现：Transformer 基线、多种子重复、持续学习实验运行记录、门控收益验证
- 并行会话正在做 M2（m2_* 实验文件与 test_datasets.py 出现在仓库）；跨会话提交前先 `git status` 分清归属

## 下次会话第一件事

1. M5.1 绑定验证预算申请：预登记 stage0/1 各 600-1000 步（约 2-5 分钟/阶段）验两实体绑定能否过 35%——**需小袁批准**。
2. M4.1 的混合时域经验流新预注册（T=24+32 攻长度外推）——**需小袁批准**。
3. H3b/H4b v2 10-seed 全量重训——DEFERRED_NOT_CANCELLED，**需小袁批准**；M5-Network P1-P3（CapabilityTracker/ResourceManager/SkillRegistry）待排；F10 配置封装、M3 Step5/6 排队。

## M2.2 机制验证（2026-11-21 第四轮，AutoClaw）——含 10-seed 定案补充

- **pred 分支缺陷实锤并修复**：legacy 模式预测器把目标当输入（cos(z_hat,z)=0.998、动作敏感性 0.025、读出不超过拷贝基线），三臂 world_model_validated=false；新增 `pred_mode="next"`（(s_t,a_t)→s_{t+1}，与 error 门控语义一致）后 F10_next 两 seed **validated=true**（敏感性 1.05-1.15、z_hat 单独读出下一位置 0.92-0.93 **超过真值等价读出 R2**、主任务不退化 0.91-0.97）。
- **三个真 bug 连锁修复**：① train() 的 pred_mode kwarg 不同步进 cfg → 存档记录错误语义；② _model_kwargs 漏记 router_ste/pred_mode → 权重加载成功但语义静默回退；③ run_next_mode 未设 cfg。均已修复并重训验证（同 seed 逐位复现）。
- **配对 seed 定案（10 seed，ddof=1）**：C_sparse_ce 0.877±0.087（旧"0.941±0.002 稳定"是 seed 族假象，最差 0.659）；**F10_ste 0.931±0.050，配对 C−F10=−5.41pp（F10 反超，7/10 seed 优），bootstrap 95% CI 上界 +1.28pp < 3pp 门槛——配对口径下稳健满足**；F10 方差为 C 的 57%。M2 冻结基线更新为配对口径（experiments/m2_2_validation/f10_paired_combined.json）。
- **轨迹干预**（5 臂×3 seed，同一 fork 点）：detach/STE/关 pred 任意组合 240 步内无法把 D 从坏盆地拉出——D 失败是早期锁定，修复必须从初始化生效；"早期锁定"假设获证据支持。
- **决策 D6**：pred_mode 默认翻转为 next（legacy 是已实锤的缺陷目标）；router_ste 保持 opt-in（C 臂对照依赖）。多步闭环 rollout 完成：next 模式误差饱和（h1 0.59→h10 0.66 不爆炸），解码位置超 copy/majority 基线，h10 的 copy 高值是钉住态退化区（与 H4 同源）。
- **M2.2 判定**：F10 ≤3pp 门槛**配对口径稳健满足**；CONDITIONAL 保持（差 H3b/H4b 两项，M2.2 COMPLETE 前不升 ACCEPTED）。报告：experiments/m2_2_validation/report.md。
- 测试 52 项全绿；环境审计 environment.json；代码地图 docs/code_map.md。

## H3b/H4b 口径复核与 v2（2026-11-22）

- H3b v1 作废为验收证据：标签 oracle 读取完整障碍图，但 26 维输入没有障碍地图且轨迹零碰撞；“BFS 可解”也与实际曼哈顿贪心标签不一致。旧 H3b 数字仅保留为历史诊断。
- H3b v2（`h3b-v2-observable-bfs`）已完成数据层修正：观测追加 6×6 障碍图（26→62 维）、标签改为确定性 BFS 最短路、每条轨迹必须实际到达目标、dev/holdout 逐拓扑实例隔离、manifest 指纹覆盖实际 obs/actions/start/goal。旧 checkpoint 输入维度不兼容，**尚未重训、尚无模型成绩**。
- H4b 重新定义为“反转规则分布偏移”，不再声称规则切换识别。评分拆为冲突步/巧合步/总分：F10 10-seed 为 **0.589±0.145 / 0.862±0.068 / 0.720±0.084**；C 冲突步 0.471±0.125，配对 C−F10=-0.118，bootstrap 95% CI [-0.214,-0.018]。由于没有规则提示与真实闭环，不能外推为自主规则切换能力。
- H4b cued v2 另建正常/反转成对样本，同一 start/goal、每步显式规则提示（27 维）；旧 26 维 checkpoint 同样不兼容，等待配对课程重训。
- 证据：`experiments/m2_2_validation/ood_h3b_h4b_v2.json`；测试 **64 项全绿**。M2.2 继续保持 `CONDITIONAL`，下一关是 H3b/H4b cued v2 同口径重训。

## M2.2 机制验证（2026-11-21 第四轮，AutoClaw）

- **pred 分支缺陷实锤并修复**：legacy 模式预测器把目标当输入（cos(z_hat,z)=0.998、动作敏感性 0.025、读出不超过拷贝基线），三臂 world_model_validated=false；新增 `pred_mode="next"`（(s_t,a_t)→s_{t+1}，与 error 门控语义一致，默认 current 不动历史口径）后 F10_next 两 seed **validated=True**（敏感性 1.05-1.15、z_hat 单独读出下一位置超拷贝基线 +30pp、主任务不退化 0.91-0.97）。
- **配对 seed 反转旧叙事**（预登记 20261131-35，experiments/m2_2_validation/f10_paired_seeds.json）：C_sparse_ce 0.853±0.113（旧 seed 族 0.941±0.002 是 seed 运气，最差 0.659）；**F10_ste 0.934±0.071，配对均值 C−F10=−8.0pp（F10 反超），4/5 seed 优于 C**；bootstrap 95% CI [−21.6, +5.6] 跨零，n=5 证据方向明确但未定案。机制自洽：C 的路由器从未被任何损失训练（CE 模式连均衡/熵都没有），F10 的任务梯度路由跨数据 seed 稳健。
- 轨迹干预实验（同一 fork 点 5 臂分叉）结果见 experiments/m2_2_validation/pred_interventions.json。
- 环境审计：experiments/m2_2_validation/environment.json（运行时检测：torch 2.10.0+cu128、cuda 运行时可用但本轮实验全部 device=cpu）；代码地图 docs/code_map.md。
- 测试 52 项全绿（+pred_mode 回归 3）。

## M2 收尾（2026-11-21 第二轮，AutoClaw）

- **全量重载验证**：22 个 checkpoint 全部重载评分，Δ=0.0 逐位复现（registry.json）；sha256 已登记。
- **冻结 holdout**：H1 新对/ H2 长48 基本不掉点；**H3 障碍布局 OOD 掉 13-18pp（全部臂）**；H4 规则反转为弱对照（角点退化区，非泄漏信号）。
- **机制诊断**（gradient_diagnostics.json）：稀疏模式下 action/pred/probe 对 router 梯度全为 0（任务盲路由），仅 balance/entropy 训练它；稠密 B 的 router 有任务梯度且 action×pred cosine -0.453 但仍收敛；D 是"从未收敛"不是"先崩后逃"。
- **多 seed（20261122-24）**：C 0.941±0.002（架构稳定）；D 0.627±0.055（复现失败）；F1 0.895±0.016；**F7 0.872±0.038，其单 seed 0.923 确认为最好种子噪声，被 F1 反超**。
- **新增消融**：F9（只软化、不 detach）0.593±0.039 ≈ D ⇒ detach 必要；**F10（F7+STE 路由）0.914±0.031（n=5），距 C 2.7pp——均值首次进入 ≤3pp 目标线**，方差大待定案。
- **CPU 基准（独占）**：稀疏 25.6ms vs 稠密 38.7ms p50 @1 线程 = **1.51×**（旧 1.68× 撤回）；8 线程小算子受竞争（p95 升至 50ms）；RSS ~503MB。
- **判定**：M2 **CONDITIONAL**（未闭环，差 F10 定案 + pred 显式有效性指标）。详见 experiments/m2_closeout/m2_closeout_report.md。
- 测试套件 49 项全绿（+router_ste 3、+closeout 回归 3、+micro_lang 11）。
