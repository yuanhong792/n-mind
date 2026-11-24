# N-MIND C1.1 基础计算能力突破：真实训练与统一评测报告

## 1. 结论摘要

- 训练真实执行：**2048 / 2048 optimizer.step**，墙钟 **478.7s**，停止原因 `step_budget`。
- 父模型：C1.0 `best`（1152 步）；C1.1 Best 选择步数 **2048**。
- 固定面板宏平均：父模型 **0.2852** → C1.1 Best **0.3963** → Final **0.3963**。
- 封存集宏平均：父模型 **0.2489** → C1.1 Best **0.3133** → Final **0.3133**。
- 立即 EOS：父模型 0.1633 → Best 0.0 → Final 0.0。
- 多字符复制与短序列反转仍为 **0.0**；**不具备进入基础编程训练的前置条件**。

## 2. 父模型、数据与预算

- 父模型审计：`C11_PARENT_AUDIT.json`；逐张量核验 `last == final`，选定 `best`。
- 数据：train 14366 / val 1837 / sealed 1948；反事实 400 对；长度泛化 900 条。
- 泄漏：跨语义组重合 0，与 C1.0 历史 prompt 重合 0。
- 预算：2048 步 / 478.7s，台账 `C11_BUDGET_LEDGER.jsonl`，跨重启共享。

## 3. 本轮真正接入的训练修复

- 任务采样：当前重点 60% / 未达标基础 20% / 已学回放 20%；稳定门槛连续两个评估点。
- 答案监督：首 Token 2.0、其他内容 1.0、EOS 0.25；按样本权重和归一化。
- Padding：`valid_mask=False` 的位置不写 Workspace、不计 Router 负载、loss 权重为 0。
- Best 选择：固定验证面板、任务等权宏平均；Final 先于 Best 保存。

## 4. 固定面板统一评测（自由生成，n/ok 已给出）

| 任务 | 父模型 ok/n | Best ok/n | Final ok/n |
|---|---:|---:|---:|
| A1_copy1 | 2/10 (0.2) | 9/10 (0.9) | 9/10 (0.9) |
| A2_copyN | 0/60 (0.0) | 0/60 (0.0) | 0/60 (0.0) |
| A3_index | 24/60 (0.4) | 33/60 (0.55) | 33/60 (0.55) |
| A4_reverse | 0/60 (0.0) | 0/60 (0.0) | 0/60 (0.0) |
| B2_bind | 34/60 (0.5667) | 35/60 (0.5833) | 35/60 (0.5833) |
| B4_bind | 22/60 (0.3667) | 14/60 (0.2333) | 14/60 (0.2333) |
| B2_counter | 31/60 (0.5167) | 28/60 (0.4667) | 28/60 (0.4667) |
| B5_reassign | 14/60 (0.2333) | 22/60 (0.3667) | 22/60 (0.3667) |
| B6_indirect | 17/60 (0.2833) | 28/60 (0.4667) | 28/60 (0.4667) |

## 5. 封存集与泛化

- 反事实配对：父 7/400 (0.0175)，Best 27/400 (0.0675)，Final 27/400 (0.0675)。
- 长度泛化（5–6字符）：Best 0/900 (0.0)。
- 变量重命名 B7_rename：Best 20/43 (0.4651)。
- 整体宏平均是否超过所有适用捷径：**是**；捷径基线：{"fixed_answer": 0.0633, "first_digit": 0.3533, "last_digit": 0.3533, "nearest_assign": 0.36, "random_guess": 0.0073}。

## 6. Router / Workspace 审计

- parent: top_k=2/4，稀疏空间=True，使用块数=4，最大/最小负载比=1.813，Workspace 平均占用=1.0222，STE 梯度范数=0.375919。
- best: top_k=2/4，稀疏空间=True，使用块数=4，最大/最小负载比=2.083，Workspace 平均占用=1.028，STE 梯度范数=1.142308。
- final: top_k=2/4，稀疏空间=True，使用块数=4，最大/最小负载比=2.083，Workspace 平均占用=1.028，STE 梯度范数=1.142308。
- Padding 隔离实测：{"padded_h_zero": true, "padded_workspace_zero": true, "pad_load_contribution": 6.0, "expected_load_contribution": 6.0, "verdict": "PASS"}。
- 诚实限制：`workspace.write` 的 overflow 仍未被情景记忆消费者使用；信息保持缺陷未完全消除。

## 7. 保留与遗忘

- parent_to_best: macro Δ=0.1111；提升 ['A1_copy1', 'A3_index', 'B2_bind', 'B5_reassign', 'B6_indirect']；下降 ['B4_bind', 'B2_counter']。
- parent_to_final: macro Δ=0.1111；提升 ['A1_copy1', 'A3_index', 'B2_bind', 'B5_reassign', 'B6_indirect']；下降 ['B4_bind', 'B2_counter']。
- best_to_final: macro Δ=0.0；提升 []；下降 []。
- final_to_best: macro Δ=0.0；提升 []；下降 []。

## 8. 45 组真实输入输出（含失败，未用解释器纠正）

| 模型 | 任务 | Prompt | 期望 | N-MIND 原始输出 | 正确 | 结束 |
|---|---|---|---|---|---|---|
| parent | A1_copy1 | ECHO W | W | W | 是 | eos |
| parent | A1_copy1 | PRINT 1 | 1 | 1 | 是 | eos |
| parent | A3_index | INDEX 22 2 | 2 | 2 | 是 | eos |
| parent | A3_index | INDEX 73 1 | 7 | 7 | 是 | eos |
| parent | A3_index | INDEX 257474 4 | 4 | 4 | 是 | eos |
| parent | A3_index | INDEX 0571 3 | 7 | 7 | 是 | eos |
| parent | A3_index | INDEX 33 2 | 3 | 3 | 是 | eos |
| parent | A3_index | INDEX 73 1 | 7 | 7 | 是 | eos |
| parent | A3_index | INDEX 12 2 | 2 | 2 | 是 | eos |
| parent | A1_copy1 | PRINT P | P | I | 否 | eos |
| parent | A1_copy1 | PRINT N | N | I | 否 | eos |
| parent | A1_copy1 | PRINT T | T | I | 否 | eos |
| parent | A1_copy1 | PRINT 6 | 6 | I | 否 | eos |
| parent | A1_copy1 | PRINT = | = | I | 否 | eos |
| parent | A1_copy1 | PRINT / | / | I | 否 | eos |
| best | A1_copy1 | PRINT P | P | P | 是 | eos |
| best | A1_copy1 | PRINT T | T | T | 是 | eos |
| best | A1_copy1 | ECHO W | W | W | 是 | eos |
| best | A1_copy1 | PRINT 6 | 6 | 6 | 是 | eos |
| best | A1_copy1 | PRINT = | = | = | 是 | eos |
| best | A1_copy1 | PRINT 1 | 1 | 1 | 是 | eos |
| best | A1_copy1 | PRINT / | / | / | 是 | eos |
| best | A1_copy1 | PRINT V | V | V | 是 | eos |
| best | A1_copy1 | ECHO G | G | G | 是 | eos |
| best | A1_copy1 | PRINT N | N | NN | 否 | eos |
| best | A2_copyN | COPY 2KRR | 2KRR | KKKKKKKKKKKK | 否 | max_tokens |
| best | A2_copyN | COPY 3D | 3D | 333333333333 | 否 | max_tokens |
| best | A2_copyN | COPY JJ | JJ | JJJJJJJJJJJJ | 否 | max_tokens |
| best | A2_copyN | COPY VHCJ | VHCJ | VVVVVVVVVVVV | 否 | max_tokens |
| best | A2_copyN | COPY 6V | 6V | 666666666666 | 否 | max_tokens |
| final | A1_copy1 | PRINT P | P | P | 是 | eos |
| final | A1_copy1 | PRINT T | T | T | 是 | eos |
| final | A1_copy1 | ECHO W | W | W | 是 | eos |
| final | A1_copy1 | PRINT 6 | 6 | 6 | 是 | eos |
| final | A1_copy1 | PRINT = | = | = | 是 | eos |
| final | A1_copy1 | PRINT 1 | 1 | 1 | 是 | eos |
| final | A1_copy1 | PRINT / | / | / | 是 | eos |
| final | A1_copy1 | PRINT V | V | V | 是 | eos |
| final | A1_copy1 | ECHO G | G | G | 是 | eos |
| final | A1_copy1 | PRINT N | N | NN | 否 | eos |
| final | A2_copyN | COPY 2KRR | 2KRR | KKKKKKKKKKKK | 否 | max_tokens |
| final | A2_copyN | COPY 3D | 3D | 333333333333 | 否 | max_tokens |
| final | A2_copyN | COPY JJ | JJ | JJJJJJJJJJJJ | 否 | max_tokens |
| final | A2_copyN | COPY VHCJ | VHCJ | VVVVVVVVVVVV | 否 | max_tokens |
| final | A2_copyN | COPY 6V | 6V | 666666666666 | 否 | max_tokens |

## 9. Checkpoint 审计

- `best`: step=2048, role=best, finite=True, torch_rng=True, sha=250a16d1f3645bba…
- `final`: step=2048, role=final, finite=True, torch_rng=True, sha=e88c5843ff98d2db…
- `last`: step=2048, role=last, finite=True, torch_rng=True, sha=0dca8071dd0f4cf4…
- `best_copy`: step=768, role=best_copy, finite=True, torch_rng=True, sha=07e3570c5b5a15fd…
- `best_binding`: step=768, role=best_binding, finite=True, torch_rng=True, sha=0f8877fa4c702eb7…

## 10. 最终问题的逐条回答

1. **不能稳定复制两个或更多字符。单字符复制 A1=0.9，但 A2_copyN=0.0；连续两次评估没有任何任务达到稳定门槛。**

2. **只具备部分能力。A3_index 固定面板 Best=0.55（父模型 0.4），高于随机但低于 90% 晋级线，不能称为稳定读取。**

3. **不能。A4_reverse Best=0.0，短序列反转仍失败。**

4. **能区分但不可靠。B2_bind=0.5833、B4_bind=0.2333；查询对象不同带来的输出差异存在，但未达到 90%/80% 门槛。**

5. **不能可靠区分。B5_reassign=0.3667，仍显著低于 85% 门槛。**

6. **有弱证据但不足。B6_indirect=0.4667，低于 80% 门槛。**

7. **整体宏平均超过所有适用简单捷径：是；但分任务看，复制与反转仍被固定输出/位置捷径问题主导，不能宣称所有能力都超过捷径。**

8. **未见组合泛化能力有限：反事实配对=0.0675，长度泛化=0.0，变量重命名与未见格式结果见 C11_GENERALIZATION.json；没有达到可靠泛化证据。**

9. **Router 有真实稀疏空间（top_k=2 < n_blocks=4），但负载不均；Workspace 的 padding 污染已修复，overflow 仍未被消费，因此信息保持缺陷只部分解决。**

10. **仍有潜力但不能证明充分。Loss 从约 2.25 降到约 1.05、面板宏平均提升 0.1111，但基础复制/反转门槛未过，继续加步数不是当前最优策略。**

11. **受控支持：父模型统一面板审计、无泄漏数据审计、padding 隔离单测、固定面板三模型同协议评测、捷径基线；仍属假设：复制失败的具体归因、Router 负载不均与容量瓶颈的因果关系。GRU 对照 NOT_RUN。**

12. **不满足。单字符复制、多字符复制、反转与变量绑定均未形成连续两评估点稳定证据，禁止进入 C4–C9 正式训练。**

## 11. 下一步（未执行）

- 复制/反转失败的根因仍需受控消融：query readout、位置编码、长度泛化读写分离。
- 不得直接扩大步数掩盖架构缺陷；需在 V1.8 预登记方案中先做单变量改动。
- GRU 短时对照本轮 **NOT_RUN**，不得据此宣称架构优越性。
