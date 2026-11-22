# N-MIND Agent Runtime R2 代码地图（P0 审计产物）

生成时间：2026-11-22 · 基线 commit `291a263`（工作树干净）· Python 3.12.10 / torch 2.10.0+cpu / CUDA False / 20 线程。
本文件先于任何 R2 代码改动落盘，作用是"先确认现有实现再决定新增位置"，避免重复实现。

## 1. 现有模块与调用边

### 1.1 Bot Runtime（`src/nmind/bot/`，R1 交付）

| 模块 | 关键对象 | 职责边界（现有代码实际做的） |
|---|---|---|
| `runtime.py` | `BotRuntime` | 自主循环总装配：`run_idle_cycle()` = 状态机迁移 → `goals.next_goal()` → 生成任务 → 收件箱抢占检查 → `neural_attempt()` → 可选工具 → `verify()` → `ExperienceBridge.build_experience()` → `tracker.record_attempt()` → `_maybe_candidate()` → 写 `autonomous_cycles.jsonl` → COOLDOWN。语言侧在 `__init__` 里 `CharTokenizer.load` + `load_lm`，缓存 `stable_sha=state_hash()` / `stable_file_sha=file_sha256()` |
| `runtime.py` | `BotRuntime.neural_attempt(question)` | 输入桥接：`f"用户：{question}\n助手："` → 一次 forward 拿 `LMState(h, ws)` → `generate(..., prompt=tok.encode("助手："), state=state, greedy=True, max_new_tokens=24)` |
| `runtime.py` | `sealed_set/sealed_questions/sealed_as_rows/measure_heldout` | 封存集按题面去重、只读，评测复用同一桥接约定 |
| `practice.py` | `make_task / execute_tool / verify / task_fingerprint / make_sealed_set` | R1 三个练习族（arithmetic/json_extract/grid_rules）、计算器与 JSON 工具、规则校验器 |
| `goal_manager.py` | `GoalManager.candidates/next_goal`、`LearningGoal` | 白名单内的确定性规则选择器（缺口/失败率排序 + 轮转排除），`selection_source=rule` |
| `capability_tracker.py` | `CapabilityTracker`（`LEVELS`/`NEURAL_HELDOUT_THRESHOLD`/`record_attempt`/`record_heldout`/`family_summary`/`summary`/`consecutive_failures`） | 能力台账，主键 `capability_id`（形如 `arithmetic:neural`），`family_summary()` 才是按族聚合视图（历史 bug 注释在此） |
| `experience_bridge.py` | `VerifiedExperience`、`to_learning_rows` | 唯一"经验→学习行"入口；`is_learning_eligible` 只放行 `verification_status=="verified"` |
| `learning_loop.py` | `run_candidate_update`、`free_gen_em`、`NEW_SKILL_GAIN_THRESHOLD`、`FORGETTING_LIMIT_PP` | 候选学习：逐位继承断言 → 有界掩码更新 → 四臂对照 → gates → PROPOSED/REJECTED → `save_lm` |
| `evolution_manager.py` | `evaluate_candidate`、`promote`、`rollback`、`verify_lineage`、`list_candidates`、`quarantine_candidate` | 候选生命周期；晋升必须 `--i-approve`，否则 `NEED_I_APPROVE` |
| `policy.py` | `BotPolicy`、`PRACTICE_FAMILIES` | 硬预算/白名单/开关（代码强制，不受提示词与网页文本影响） |
| `store.py` | `BotStore`（SQLite） | 表：`events` / `capabilities` / `completed_tasks` / `kv` /（inbox、transitions 等）；`upsert_capability(rec)` 走的是 dict 字段，新增列需保持旧行可读 |
| `status.py` | 状态机 + `ControlAudit` | `BotPhase` 迁移、pause/resume/stop 检查点、控制面审计 |
| `controller.py` / `ipc.py` / `watchdog.py` | 单实例、控制指令、资源采样 | `get_proc_info()` 给出 RSS/CPU 秒，R1 资源台账由此来 |
| `cli.py` | R1 的 15 个 `bot` 子命令 | 全部在 `src/nmind/cli.py` 注册（`--offline/--json/--root` 为全局开关，叶子命令需 `_add_common`） |

### 1.2 语言/神经侧（`src/nmind/language/`、`core/`）

- `neural_dialogue.py`：`NMindLM`（`tok_emb` → `NCell.forward_sparse` + `BlockRouter.route` + `GlobalWorkspace.read/write` → `head`），`LMConfig(d_model=96, n_blocks=8, block_hidden=96, workspace_dim=64, workspace_slots=4, top_k=2)`；`save_lm/load_lm` 格式 `nmind-lm/1`，**词表哈希不匹配直接拒绝加载**（R2 不能换词表）。`block_load_count` 与 `router.load_bias` 是训练态 buffer。
- `tokenizer.py`：`CharTokenizer` 字符级；词表由训练语料闭集决定（M5.2 生产词表 31 个 token，见 §2）。
- `decoder.py`：`generate(model, tok, prompt_ids, *, state=None, max_new_tokens, greedy)`；换行/EOS/预算三种停止；`GenerationRecord(text, ...stop_reason)`。
- `lm_update.py`：`encode_rows`、`answer_masked_loss`（仅答案位计损，**prompt 位不训**）、`masked_stream_update(..., trainable=, budget=, replay_rows=, label_permutation_seed=)`、`parameter_audit`。`trainable` 的合法值已支持 `"all"` 与子串集合式范围（Head/Workspace/NCell 局部冻结用同一个入口）。
- `session.py`：`ChatSession` 跨轮携带 `(h, ws)` —— R2 的 A 组复现必须与该约定一致，不能另造会话实现。
- `learning/online_adaptation.py`：`AdaptationBudget`（硬闸，超限抛 `BudgetExceeded`）、`clone_snapshot`、`verify_bitwise_inheritance`、`seal_eval_state`、`state_hash`。
- `learning/online/scheduler.py`：`SchedulerConfig/TriggerResult/Scheduler`；`learning/online/verifier.py` 的 `outcome_source` 只有 `externally_verified` / `deterministic_oracle` 两值，`needs_user_confirm` 恒为"未验证"。
- `skills/toolbox.py`：`Toolbox`（`calculator` / `json_extractor`）。

### 1.3 联网（`src/nmind/online/`）

`fetcher.py`（allowlist 前缀 + 离线硬断 + 注入 transport + 内容 sha256 去重）、`explorer.py`、`retrieval.py`（`ExternalRetrieval`：本地 BM25/精确命中优先，**联网结果只写外部记忆、从不进学习流**）、`memory.py`、`promotion.py`。
R1 实测结论：36 次检索 35 次 `local`、联网 `not_authorized` 0 真实请求。

### 1.4 既有实验（可复用的"锚点"）

- `experiments/m5_language/run_m52.py`：A 组的权威定义 —— `build_s0(n, seed)` 模板
  `f"用户：小明在{p}。\n{ack}用户：小明在哪里？\n助手："` → `f"小明在{p}。"`，`ack ∈ {"好的。","明白了。","收到。",""}`，`p ∈ 1..4号位置`；
  `s0_sealed = build_s0(80, 20261403)` 记录 free-gen EM **1.0**（`m52_eval.json`）。R2 的 A 组必须与该生成器逐字节一致，否则"复现"没有意义。
  同文件给出 B/C 的构造先例：`build_s1(n, seed, update=, para=)`（两实体绑定 / 位置交换 / 更新 / 换说法）与 `eval_free()`（先 forward 全 prompt、再从 `"助手："` 续写的官方约定）。
- `experiments/bot_runtime_r1/`：R1 全量工件与 `reproduce.ps1`（PowerShell 5.1 需 UTF-8 **BOM**）。

## 2. 审计结论：R1 神经能力零增长的机制

1. **词表边界**：stable checkpoint（`m52_full/lm_best.pt`）绑定的 `tokenizer_m52.json` 只有 31 个 token，全部来自中文绑定域（`小在。明：号位置\n用户哪里？助手红1324收到好的白了`）。
2. **任务分布不匹配**：R1 三个族的问题文本是 `请计算：23加78等于多少？` / `JSON字段提取：{"name": "kxum", ...} 中字段 name 的值是多少？`，字符集与词表几乎不相交 → 每个 OOV 字符退化为 UNK，模型在"从未见过的输入流"上贪心解码，输出与答案不可能匹配。
3. **验证器口径错配**：`verify()` 要求模型自己算出 `1558-759=799`、从 JSON 里"读出"`qfvpd`。这类任务在有可信外部工具时本该走 `TOOL_USE` + 记忆；把工具答案直接当神经监督目标，等于要求语言头重新学会它没有词表的领域。
4. **台账混用**：`capability_id` 用 `{family}:neural` / `{family}:tool` 区分，但"神经选对工具"与"工具算对结果"仍记在同一族账上，容易把工具能力当神经能力（任务书 §3 明确禁止）。
5. **方法学缺陷（代码级，两处）**：
   - `learning_loop.run_candidate_update` 的旧技能遗忘样本取自 `prompts[:8]`（**新任务的训练行**），不是旧技能 A 的封存集 → 遗忘矩阵测的是错的分布（任务书 §6.4）。
   - `_memory_only_answer(question, answer)` 直接 `return answer`，memory 臂按构造恒为 1.0 → 该臂无法区分"检索命中"和"把标签抄进答案"。
   - 附带：四臂没有 A 组保持率、没有 (entity, position) 组合覆盖与 unseen-combo 报告，没有反事实跟随率（M5.2 已证明 EM 会掩盖近因启发式）。

结论：任务书判定成立 —— 问题在"模型与任务族分布不匹配"，不是训练次数不足。R2 因此先用**同一 stable checkpoint、同一词表**（否则 `load_lm` 拒绝加载）设计 A/B/C，A 复现已有技能、B 做可识别的相邻泛化、C 只做诊断。

## 3. R2 新增件与落点（最小接口）

| 新增文件 | 接口 | 为什么放这里 |
|---|---|---|
| `language/binding_data.py` | `BindingTask` / `build_split(family, n, seed)` / `oracle(task)` / `vocab_report(tok, tasks)` / `sealed_*` | 任务数学与词表属于语言侧（与 `dialogue_data.py` 同层），A 组模板须与 M5.2 同源；Bot 层不重复实现 |
| `bot/task_capability_bridge.py` | `TaskSpec`（A/B/C + 工具族统一视图）、`GateAResult`、`run_gate_a()` | 任务书 §1.3 指定的桥接位置；负责"任务↔模型输入↔oracle↔技能台账"的对齐审计 |
| `bot/skill_registry.py` | `SkillSpec`（§2 全部字段）+ `SkillRegistry`（`register/get/spec_for/for_family`）+ 归因枚举 | §1.4 技能注册表；声明"用什么验证器/是否允许联网/评测哪些指标/是否可作神经目标"，供 runtime/curriculum/CLI 共享 |
| `bot/curriculum.py` | `ScoredSkill`、`CurriculumConfig`、`SkillSelector.pick(assessments)` | §4 独立评分器：与 `GoalManager`（R1 族级规则选择器）分工——GoalManager 管练习族，Selector 管"技能级可学资格"，两者都写 `selection_source` |
| `online/resource_manager.py` | `CapabilityTier`、`ResolutionDecision`、`ResourceManager.resolve(...)` | §7 五档链路 + 真实工具注册表 + allowlist 校验 + mock transport 注入点 |
| `bot/cli.py` | `bot skills` / `bot capability-report` / `bot learn-status` / `bot curriculum` / `bot eval --family binding` | §8 只增加确实测过的子命令 |

## 4. 旧实现缺口（R2 必须改，不是重写）

- `learning_loop.py`：遗忘基准改用 A 的 sealed 集；memory 臂改为真实外部记忆查询（不读封存标签）；候选必须同时报 A 保持率（含样本量/CI）、B unseen 自由生成 EM、反事实跟随率；`trainable` 范围在实验前预登记并写入工件；四臂同源断言已有，保留。
- `capability_tracker.py`：`record_attempt` 增加 `response_source/skill_id` 与"工具执行 vs 神经决策"分账，旧行可读（SQLite 迁移，不重建表）。
- `practice.py`：R1 族保留为**工具任务**，标注 `domain_compatibility`；`grid_rules` 要么给确定性只读程序 oracle 并判 `TOOL_ONLY`，要么如实记「已验证工具任务、神经任务未支持」。
- `runtime.py`：反馈从"verified/rejected/uncertain"升级为四态；工具任务的 verified 经验不得进入神经学习流（`learning_target` 决定）；不可学任务 → `NO_ELIGIBLE_TASK` 休眠而不是继续盲练。
- `fetcher.py`：真实传输路径要硬化（HTTPS-only、精确 host+path、DNS 重绑定/私网/元数据 IP 阻断、逐跳转复校、流式字节上限、MIME、配额）。本轮不启用真实联网，但代码与探针必须在。
- `status.py`/控制面：每轮写任务族、尝试数、神经/工具/记忆路径、拒绝原因、模型 SHA、RSS/CPU/耗时、下轮是否训练（R1 已有部分，补齐字段与导出）。
