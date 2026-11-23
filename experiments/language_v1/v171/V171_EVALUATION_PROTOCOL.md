# V1.7.1 评测协议（修复 V1.7 口径缺陷）

## 一、两套必须区分的评分指标

### A. 快速判别 Token 评分 —— `DISCRIMINATIVE_TOKEN_PROXY`

- 定义：一次 prompt 前向，仅取候选答案**判别 token**的 logprob（颜色候选首个分歧位 fd=0）。
- 用途：仅用于训练过程轻量监控。
- **不得称为完整答案条件概率。**
- 输出字段名前缀：`DISCRIMINATIVE_TOKEN_PROXY_*`。
- 实现：`v171_protocol.proxy_first_token_lp()`。

### B. 完整答案条件概率 —— `FULL_ANSWER_CONDITIONAL`

- 定义：`log P(tok(answer) + EOS | prompt)`，按自回归顺序，**每个答案 token 与 EOS 全部参与**，求和（不平均）。
- 实现：`v171_protocol.full_answer_logprob()`。
- 加速：因果 LM 中对 `[prompt+answer+EOS]` 做**一次 teacher-forced 前向**，读取答案各位置的 logprob 即为联合条件概率；共享 prompt 计算。
- 数学正确性验证：`full_answer_logprob()`（单次前向）与 `full_answer_incremental_logprob()`（逐 token state passing 增量）在实测样本上 diff = 0.0；回归测试 `test_v171_protocol.py::test_full_logprob_teacher_forced_equals_incremental` 固化。
- 不同候选长度不同：求和而非取首 token，禁止长度混淆。
- 输出字段名前缀：`FULL_ANSWER_CONDITIONAL_*`。

### 配套记录字段（每条样本）

- `full_gold_logprob`：完整答案总 logprob（含 EOS）
- `proxy_ce`：首判别 token CE
- `full_tf_ce`：完整答案 Teacher-forced CE = `-总logprob / token数`
- `free`：自由生成原文（不替换、不修正）
- `finish`：EOS 结束原因（`eos`/`max_tokens`/`utf8_incomplete`/`utf8_filtered`/`byte_decode_error`）
- `utf8`：UTF-8 前缀状态（`complete`/`incomplete`/`invalid`）

## 二、公平比较统一配置（P3）

以下全部模型使用**完全相同**的配置：

| 项目 | 值 |
|---|---|
| 数据 | 同一份 600 条封存集 `v17_sealed.jsonl` |
| Tokenizer | V1.5 ByteBPE `hash=26af116ebd13ff00`（不变） |
| Decoder | `generate_bpe(utf8_safe=True)`（V1.7 修复版） |
| 候选集 | 每任务固定候选（见 `CANDIDATES_BY_TASK`） |
| 最大生成 Token | 16 |
| 评价程序 | `v171_protocol.evaluate_model()` 单一实现 |
| 测试顺序 | 同一 rows 顺序遍历 |

**独立性声明**：该封存集已用于 V1.7 开发期分析（decoder 对照、生成检查），其盲测独立性**可能已受影响**。本报告中的提升不得视为全新盲测成绩。

模型列表：
1. V1.5 Parent（有效）
2. V1.7 Best（权重 = 父模型，有效加载但无新信息）
3. V1.7 Final（**权重 = 父模型，按 FINAL_WEIGHTS_LOST 处理，仍可加载评测以作对照**）
4. 中间 checkpoint：不存在（见 Checkpoint 审计）

## 三、EOS / 生成诊断（P5）

| 指标 | 定义 | 分母 |
|---|---|---|
| `immediate_eos_rate` | 首步即 EOS（生成 token 数=0 且 finish=eos） | 样本数 N |
| `non_empty_answer_rate` | 生成 token>0 且文本非空 | N |
| `valid_utf8_non_empty_rate` | UTF-8 complete 且 token>0 且文本非空 | N |
| `correct_eos_rate` | finish=eos 且 token>0 且 UTF-8 complete 且文本非空 | N |
| `utf8_complete_rate` | UTF-8 字节缓冲 complete（空串也算 complete，故单列） | N |

- 立即 EOS **不得**解释为“正确完成回答”。
- 若需最小输出 Token 数，只能作为**单独对照配置**，成绩不得伪装为模型能力；本轮主评测**不启用**任何最小 Token 规则。
- 禁止用规则直接输出正确中文答案。

## 四、数据集隔离

- 训练集 / 验证集 / 封存集：按 fact key 与模板族去重隔离。
- 反事实成对：同一 key 的两题（原事实 / 交换属性）必须**同时**用 B 评分正确才计一对通过。
- 所有指标均报告分子与分母（详见 JSON 的 `n` 与 counts 字段）。
