# N-MIND M4 阶段报告｜最小充分信息交互

> 状态：**ZERO_GRADIENT_STAGE_COMPLETE**  
> 分支：`experiment/minimum-sufficient-interaction-m4`  
> 正式训练授权：**未授予**。当前所有 M4 主实验、正控制训练、ISO/Pareto 能力结论均为 `NOT_RUN`。  
> 本阶段只完成第一至第八阶段，以及第九阶段的授权确认准备。

---

## 一、本阶段真实完成的工作

1. **M3.1 历史审计**：监督协议、反事实捷径、数据语义泄漏、距离配对、Checkpoint 证据链、成本记账均已复核，详见 `M31_REPAIR_REPORT.md`。
   - M3 prompt 监督占比 train=0.8964；M3 EOS 权重 0.25，不是 1.0。
   - M3 反事实 400/400 查询第一位，“输出第一个数字”捷径配对成功率 1.0。
   - M3 成本按全局步数乘算，单模型虚增 1×–8×。
   - 字符串 prompt 零重合不等于语义隔离；canonical train/val 重合 47，train/sealed 重合 54。
   - 16 个 M3 checkpoint 均可加载，Best/Final 标记未发现错配。

2. **数学研究**：`M4_RESEARCH_SOURCES.md` 归档原始论文及核实状态，二手报道标 `NOT_VERIFIED`。

3. **Transformer 成本解剖**：`M4_TRANSFORMER_COST_ANALYSIS.json` 完成 16/32/64/128/256/512 长度、Batch 1/8/16、解析 MAC/Profiler/墙钟三口径分离。

4. **数学算子**：低秩信息交互、选择性交互、增量核、正交压缩、数学等价 head 优化均完成零梯度实验。

5. **公式发现实验室**：29 个候选，17 个存活，算子阶段 Pareto 前沿为 **F01, F17**。

6. **新数据与框架**：M4 数据 train=1392、val=464、sealed=464，canonical 重合全为 0；四类反事实各 100 对，查询位置 first/middle/last=136/132/132；严格距离配对 60 组 × 7 个距离。

7. **P14 数学验证**：`M4_MATH_TESTS.json` = **24/24 PASS**。

8. **训练器**：`train_m4.py --authorize` 已就绪，但当前只跑过 dry-run；`M4_TRAINING_RESULTS.json` 明确为 `DRY_RUN_COMPLETE`，实际 gradient step=0。

---

## 二、最终报告 26 问

### 1. M3 历史实验修复了哪些问题？

确认并修复了六类问题：监督协议错配、反事实位置捷径、A5_repeat 训练缺失、语义族泄漏、距离数据非严格配对、成本统计虚增。Checkpoint 标记错配担忧未被证实。历史原始报告未修改。

### 2. 现有参考 Transformer 真正的计算瓶颈在哪里？

随长度变化。短序列 attention share=0.5525，FFN=0.442；长序列 attention share=0.7814，FFN=0.1563。Profiler 还显示 LayerNorm、addmm/mm/bmm、arange、copy_、softmax 等工程热点。

### 3. 注意力计算占整个模型计算量多少？

按解析口径：短序列 55.25%，长序列 78.14%。这不是墙钟占比；墙钟还受内核实现、内存访问和 CPU 调度影响。

### 4. 不同输入长度下瓶颈是否变化？

是。有效三角区相对稠密矩阵的节省从 T=16 的 46.88% 增至 T=512 的 49.90%，但参考实现仍执行稠密矩阵，未用融合精确因果内核兑现该理论节省。

### 5. 哪些数学运算可以严格消除？

Answer-only 下 Prompt 位置无需计算词表 Head。数值测试 logits 最大差 4.77e-07、loss 差 0.0、梯度差 3e-08、参数更新差 7e-09，贪心下一 token 一致。但 CPU 实测时间变化 -3.42%（负收益），因此 **未获得可确认的实际节省**。

### 6. 哪些计算只能近似替代？

降低 QK 交互维度、分组摘要 + top-k 精确交互、线性核增量状态、正交低维压缩、核映射/归一化/门控公式变体，都是近似或带条件的机制，不能宣称与 softmax attention 等价。

### 7. 低秩信息交互是否保持任务能力？

**NOT_RUN**。算子层 output relative_l2 在 r=4/8/16/32/64 均较大，且全图收益从 r=4 的 60.42% 降到 r=64 的 -33.33%。没有训练后能力证据，不能判断是否保持。

### 8. 最有效的交互维度是多少？

按算子 MAC：r=4 全图节省最高（60.42%）；按预登记主实验：冻结 r=16，以误差和表达维度折中。按任务能力：**NOT_RUN**。

### 9. 选择性交互是否真正跳过矩阵计算？

是，算子层跳过精确注意力比例 75.00%；精确注意力 MAC 为 65,536，稠密为 262,144。因果扰动差为 0.0，STE 梯度范数 0.02329。

### 10. 稀疏选择自身需要多少计算成本？

摘要构造 16,384，摘要更新 2,048，粗检索 32,768，top-k 2,048，gather 32,768，精确注意力 65,536，合计 **151,552 MAC**；相对稠密 262,144 的名义节省为 42.19%。不能只报精确注意力节省。

### 11. 训练与实际部署是否一致？

部分一致但不完全等价。M4-B 训练用软门、推理用硬门；P14 测试只验证形状与有限性，不验证 logits 数值相等。必须在授权训练后单独报告软/硬门差距，禁止用软门能力搭配硬门成本。

### 12. 新公式是否减少实际 CPU 时间？

算子层 F17 中位耗时 52.9μs，低于参考 F01 的 84.55μs，但相对误差 0.660377。只有算子初筛证据，训练后能力与同等能力效率均为 **NOT_RUN**。

### 13. 反事实能力是否超过位置捷径？

数据层已超过：M4 查询位置均衡且四类反事实各 100 对；M3 的 100% 第一位捷径已被拆掉。模型能力层 **NOT_RUN**。

### 14. 查询切换能力是否真正建立？

数据层覆盖了 query_switch 100 对；模型层 **NOT_RUN**，不能宣称建立。

### 15. 多字符连续生成是否获得突破？

**NOT_RUN**。M4_GENERATION.jsonl 为 40 条明确 `NOT_RUN` 记录，不是生成结果。M3 的真实 48 组生成样例保留在 `../math_m3/M3_GENERATION.jsonl`。

### 16. 小规模正控制实验是否通过？

训练未运行，标记 `NOT_RUN_TRAINING`。当前只验证监督对齐；正控制数据已扩充为双字符复制、多字符复制和双变量绑定。

### 17. 是否具备真正的长度泛化能力？

数据层具备严格配对：距离 0/4/8/16/32/64/128 各 60 条，共享 60 个 pair_id。模型评测 **NOT_RUN**。

### 18. 是否减少达到相同能力所需的样本数量？

**NOT_COMPARABLE**。未训练，无学习曲线。

### 19. 是否减少达到相同能力所需的训练计算量？

**NOT_COMPARABLE**。训练 step=0。

### 20. 是否减少达到相同能力所需的推理计算量？

**NOT_COMPARABLE**。只有固定输入性能与算子成本，无同准确率生成成本曲线。

### 21. 是否达到 E1 或更高效率等级？

没有。所有 ISO 目标为 `TARGET_NOT_REACHED` / `NOT_COMPARABLE`，不授予 E1–E5。

### 22. 哪些收益属于数学优化？

因果有效三角区理论节省约 46.88%–49.90%；Answer-only Head 数值等价。但前者尚未由融合内核兑现，后者在本机词表 102 下实测负收益。

### 23. 哪些收益属于工程实现优化？

R2 已有融合 QKV、相对 RoPE 与 KV Cache；逐 batch 成本记账替代固定长度公式。当前没有新增并验证为正收益的工程优化。

### 24. 哪些新公式存在原创研究价值？

当前公式均为已有算子的受控组合与搜索，不宣称原创数学发现。有价值的是“静态维度检查 + 因果性 + 梯度 + 数值稳定 + 成本 Pareto”的筛选流程，但它属于实验方法，不是新定理。

### 25. 哪些候选应继续验证或者淘汰？

授权后继续：R1、R2、M4-A(r=16)、M4-B(group_size=4, top_k=2)。  
淘汰：Answer-only Head 单独优化（本机负收益）；F02–F09 等违反因果/梯度检查的 score/value 低秩公式；identity kernel（分母不稳定）。  
保留但 NOT_RUN：增量核、正交压缩、M4-C/D/E/F、固定局部窗口基线。

### 26. 是否具备进一步研究低成本自主学习的条件？

接口与纪律已具备：状态、会话记忆、长期权重、持久技能已区分；P18 只做数学建模。没有持久技能学习证据，禁止启动无限自主训练。

---

## 三、证据索引

| 主题 | 文件 | 结论 |
|---|---|---|
| 历史修复 | `M31_REPAIR_REPORT.md` | 五类历史问题被证实，checkpoint 错配未证实 |
| 数学验证 | `M4_MATH_TESTS.json` | 24/24 PASS |
| 成本解剖 | `M4_TRANSFORMER_COST_ANALYSIS.json` | 瓶颈由 FFN/Head 与 attention 随长度共同决定 |
| 低秩 | `M4_LOW_RANK_ATTENTION.json` | 只有 reduce_qk_dim 真降 score MAC |
| 选择性 | `M4_SELECTIVE_ATTENTION.json` | 跳过 75% 精确注意力，但路由成本必须计入 |
| 增量核 | `M4_INCREMENTAL_KERNEL.json` | 属已有线性注意力，不宣称等价 |
| 正交压缩 | `M4_ORTHOGONAL_COMPRESSION.json` | 小矩阵增量 QR 不快于全量 SVD |
| 等价优化 | `M4_EXACT_REDUCTION.json` | 数值等价，CPU 无收益，淘汰 |
| 公式搜索 | `M4_FORMULA_SEARCH_RESULTS.json` | 29 候选，17 存活，Pareto F01, F17 |
| 数据 | `M4_DATA_AUDIT.json` | train/val/sealed canonical 重合全为 0 |
| 预登记 | `M4_PREREGISTERED_PLAN.json` | 授权状态 NOT_YET_GRANTED |
| 训练 | `M4_TRAINING_RESULTS.json` | DRY_RUN_COMPLETE，0 step |
| 生成 | `M4_GENERATION.jsonl` | 40 条 NOT_RUN，非生成样例 |

---

## 四、训练授权请求（第九阶段）

M4 预登记的正式预算为：

- 正控制：R1 128 步 + R2 128 步 = 256 步；
- 正式对照：R1/R2/M4-A/M4-B × 2 seed × 480 步 = 3840 步；
- 合计最多 **4096 optimizer.step**，墙钟上限 **3600 秒**；
- 授权命令：`python positive_control.py --authorize` 与 `python train_m4.py --authorize`。

**当前状态：未授权，未执行任何梯度更新。** 用户明确授权前，不会启动训练，也不会把 dry-run 写成训练结果。
