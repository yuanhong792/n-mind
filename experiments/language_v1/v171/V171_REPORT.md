# N-MIND V1.7.1 训练成果抢救、Checkpoint 修复与语言评测纠错 — 总报告

分支：`experiment/language-v17-cn-training` ｜ 本轮 `optimizer.step = 0`（无新训练）

## 执行摘要

本轮按 P0–P6 顺序完成全部工程修复与核验。最重要的结论：**V1.7 声称的 1536 步训练权重已经永久丢失；V1.7 Best 与 Final 在逐张量层面都与 V1.5 父模型完全相同。** 修复了 Checkpoint 保存顺序缺陷、评测口径缺陷、EOS 退化统计缺陷、数据集结构缺陷，并在统一协议下完成公平评测与全部回归测试。

---

## P0：Checkpoint 抢救（`V171_CHECKPOINT_AUDIT.json` / `V171_RECOVERY_REPORT.md`）

逐张量严格加载并比较（同 tokenizer `26af116ebd13ff00` / `strict=True`）：

| 比较 | 结论 |
|---|---|
| V1.7 Best vs V1.5 Parent | 完全相同，L1=0.0，0/21 张量变化 |
| V1.7 Final vs V1.5 Parent | 完全相同，L1=0.0，0/21 张量变化 |
| V1.7 Final vs V1.7 Best | 完全相同，L1=0.0，Router 无变化 |

- Best 元数据 `steps=0`，Final 元数据 `steps=1536`，但**文件 SHA 的差异只来自 `steps`/`saved_at` 元数据**，权重逐元素一致。
- 穷尽搜索项目目录与 Windows 临时目录：**无优化器状态、无自动保存、无中间 Checkpoint**（`FINAL_WEIGHTS_LOST = true`）。
- 根因：`train_v17.py` 在 `m.load_state_dict(best_state)` 之后才构建 `final_state`，且本次 `best_step=0`（父模型即最优），故 Final 被写成了父模型权重。
- 未覆盖任何权重，未伪造/反向重建丢失权重。

## P1：Checkpoint 保存顺序修复（`train_v17.py` + `checkpoint_v17.py`）

- 新增 `checkpoint_v17.py`：`capture_state()` 先抓最后步权重 → `save_final_and_best()` **先落盘 FINAL，再恢复 BEST 落盘**；`save_periodic()` 每个评测点原子落盘最近有效权重；全部 `atomic_save()`（tmp + os.replace）。
- `train_v17.py` 已改用新逻辑；`router_delta` 改在 FINAL 权重上统计；结果新增 `final_equals_best_weights`、`checkpoint_save_order` 字段。
- 5 项必需测试（Best/Final 不同步数真不同、best=step0 时 Final 仍为末尾步、重载逐张量一致、中断保留最近有效、Stable 不被覆盖）全部通过（`tests/test_v171_checkpoint.py`，6 passed）。

## P2：评测口径修复（`V171_EVALUATION_PROTOCOL.md` / `v171_protocol.py`）

- 建立两套明确区分的指标：`DISCRIMINATIVE_TOKEN_PROXY`（首判别 token，仅训练监控）与 `FULL_ANSWER_CONDITIONAL`（`log P(answer+EOS|prompt)`，所有答案 token 与 EOS 全部参与）。
- FULL 用单次 teacher-forced 前向（共享 prompt），已回归验证与逐 token 增量 state passing **diff=0.0**；批量版与逐个版 ≤2e-6。
- 每条样本记录：完整总 logprob、首判别 token CE、完整答案 TF CE、自由生成原文、EOS 原因、UTF-8 状态。训练/验证/封存严格分离。

## P3：统一协议公平评测（`V171_FAIR_COMPARISON.json`）

相同 600 条封存集、同 tokenizer/decoder/候选集/16 max tokens/评价程序/顺序。**独立性声明**：该封存集已用于 V1.7 开发分析，盲测独立性可能受影响，不得视为全新盲测成绩。

| 模型 | B_bind(300) | C_counter(135) | D_position(164) | E_context(1) | 反事实成对(1 对) | 自由生成(600) |
|---|---|---|---|---|---|---|
| V1.5 Parent | 0.12 | 0.1259 | 0.0549 | 0.0 | 0/1 | 0.0 |
| V1.7 Best | 0.12 | 0.1259 | 0.0549 | 0.0 | 0/1 | 0.0 |
| V1.7 Final | 0.12 | 0.1259 | 0.0549 | 0.0 | 0/1 | 0.0 |

（表中为 FULL 口径；PROXY 口径 B_bind=0.12、D_position=0.0305。三模型数值完全相同——因为权重相同。）

- 基线：固定答案 0.1033、最近属性 0.4833、位置捷径 0.5167。模型表现**未超过随机/捷径基线**。
- 样本数分母已逐项标注。E_context 仅 1 条、反事实仅 1 对——V1.7 旧封存集的结构性缺陷（见 P4）。

## P4：数据集结构修复（`V171_DATA_AUDIT.json` / `dataset_v171.py`）

- 保留 V1.7 全部数据与 SHA（实测 4 个文件 SHA 与冻结审计一致，未覆盖）。
- 新建 V1.7.1 数据（写入 `v171/data/`，不含旧数据）：任务配额可控；跨集 prompt 重合=0；事实跨集=0；人物首字互异。

| split | 总数 | A | B | C(反事实) | D | E(多轮) | 有效反事实对 |
|---|---|---|---|---|---|---|---|
| train | 5418 | 20 | 1800 | 1198 | 1500 | 900 | 599 |
| val | 901 | 3 | 300 | 198 | 250 | 150 | 99 |
| sealed | 902 | 2 | 300 | 200 | 250 | 150 | 100 |
| counter | 300 | 0 | 0 | 300 | 0 | 0 | 150 |

- 反事实为真实成对：同一事实 → 原问题/原答案，交换属性 → 同问法/改变答案；两题同时对才算一对通过。
- 冻结 SHA-256 写入 `V171_DATA_FROZEN.json` / `V171_DATA_AUDIT.json`。

## P5：EOS 退化审计（`V171_EOS_AUDIT.json`）

- V1.7 旧口径把 `finish=eos` 当作成功：实测 **592/600 样本是首步即 EOS 的空串**，却被记为 `utf8_legal=1.0`、`eos≈1.0`。
- 新指标（三模型一致，n=600）：`immediate_eos_rate=0.9867`、`non_empty_answer_rate=0.0133`、`valid_utf8_non_empty_rate=0.0133`、`correct_eos_rate=0.0017`、`free_acc=0.0`。
- EOS 已进入训练监督掩码（权重 0.25）；但因真实训练权重丢失，无法验证 1536 步是否学会正确 EOS 位置——当前立即 EOS 是**父模型本身行为**。
- 主评测未启用任何最小输出 Token 规则；未用规则输出任何正确中文答案。

## P6：回归测试（`V171_REGRESSION_RESULTS.json`）

- 定向套件：**35 passed, 0 failed**。
- 完整套件：**298 passed, 3 skipped, 0 failed**（EXIT=0）。
- 覆盖：Checkpoint 保存加载、Best/Final 区分、Router 梯度、BPE 词表、答案掩码、完整答案评分、自由生成/EOS、UTF-8、事实级去重、反事实成对。全部为真实 pytest 结果。

---

## 五个必答问题（最终回答）

**1. 1536 步真实训练后的模型权重还在不在？**
**不在。** `FINAL_WEIGHTS_LOST`。Best/Final 均等于父模型，穷尽搜索无 1536 步末尾副本，不伪造、不重建。

**2. V1.7 Final 究竟是不是父模型？**
**是。** 逐张量 L1=0.0、0/21 张量变化、Router 无变化；文件 SHA 差异仅由 `steps=1536` 元数据造成。

**3. 当前模型在统一评测下究竟具备哪些语言能力？**
仅具备 decoder 层面的 UTF-8 合法中文 token 拼接（能输出合法中文字符串），但内容语义与输入无关。实体绑定 0.12、位置 0.0549、反事实 0/1、自由生成 0.0，全部未超过固定答案(0.1033)/最近属性(0.4833)/位置(0.5167)基线。**无稳定有效的语言理解与问答能力。**

**4. 哪些失败来自工程缺陷，哪些来自真实神经学习能力不足？**
- 工程缺陷（已修复）：Checkpoint 保存顺序导致训练成果丢失；评测用首 token 冒充完整答案概率；立即 EOS 空串被计为合法/成功；旧数据多轮仅 1 条、反事实仅 1 对；训练超预算 1829.8s>1800s。
- 神经能力不足（未解决）：即便按旧口径，1536 步训练 Loss 大幅下降仍未带来 B_bind/C_counter/自由生成提升；父模型即最优（best_step=0）；模型把 EOS 当作任意 prompt 的默认响应。这部分与 V1.6 结论一致，是学习能力问题，非工程可修复。
- 注意：因真实训练权重丢失，**无法区分"训练本身无效"与"训练有效但权重丢失"**——现有证据只能证明父模型无能力，不能证明 1536 步训练后的真实能力。

**5. 下一轮训练之前，还剩下哪些必须解决的问题？**
① 用修复后的 `save_final_and_best` 重训并**逐张量验证 FINAL≠BEST、FINAL≠父模型**后才认可训练结果；② 以 V1.7.1 均衡数据（多轮 900、反事实成对 100+/150 对）为训练/测试基底并冻结 SHA；③ 全程用 `FULL_ANSWER_CONDITIONAL` 口径 + 统一协议评测；④ 把实体位置信息从 `decoded = h.mean(dim=1)` 的平均中保留下来（V1.6 最可疑假设，先做零训练冻结探针，不直接重训）；⑤ 解决父模型立即 EOS 退化（核查 EOS 监督权重/位置，必要时单独做最小 Token 对照）；⑥ 把训练预算与评测耗时纳入墙钟控制（本轮评测单次≈600s）。
