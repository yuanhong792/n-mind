# N-MIND M3 数学规格（Minimum Sufficient Neural Computation）

## 0. 核心目标函数

```
minimize:  training_compute, inference_compute, state_memory, learning_cost
subject to: capability >= reference_capability
```

多目标约束优化。不将参数量直接等同计算成本，不将训练 Loss 直接等同智能能力，
不将门控值接近 0 直接等同实际节省计算。

## 1. 低秩增量状态更新（P3）

传统循环网络状态更新需高维矩阵乘法 O(d²)。M3 限制状态变换在低维子空间：

```
z      = V @ h          # V: [r, d],  h: [d]  →  z: [r]    成本 O(dr)
delta  = U @ z          # U: [d, r]           →  delta: [d]  成本 O(dr)
h_new  = h + delta                            成本 O(d)
```

当 r << d 时，状态变换理论成本从 O(d²) 降至 O(dr)。

加入可训练对角状态变换以保持部分长期信息：

```
candidate = activation(
    A @ x
    + diag_weight * h
    + U @ (V @ h)
    + bias
)
h_new = (1 - gate) * h + gate * candidate
```

- 对角变换 `diag_weight * h` 保持逐通道衰减，保留部分长期信息。
- 低秩更新 `U(Vh)` 表达子空间内的状态变换。
- 非线性 `activation` 补偿低秩限制。
- 门控 `gate` 控制新旧状态混合。

优先候选秩：r ∈ {4, 8, 16, 32}。

## 2. 自适应按需计算（P4）

三条计算路径，由可训练神经控制器从 (x, h) 决策（不含任务字符串规则）：

```
routing_logits = controller(x, h)        # 成本计入
routing_probs  = softmax(routing_logits)
```

- C0：保持/轻量更新（对角）—— 成本 O(d)
- C1：低秩更新 —— 成本 O(dr)
- C2：扩展神经计算（额外 MLP）—— 成本 O(d²·m)

### 2.1 低成本主路径

```
h_base = lightweight_update(x, h)
```

### 2.2 可选择的扩展路径

训练（软门，稠密）：

```
h_new = h_base + Σ_k probs_k * expensive_residual_k(x, h)
```

**关键纪律**：若先完整计算 expensive_residual 再乘 gate，并未实际节省昂贵计算。
正式高效推理（硬门，稀疏）采用：

```
h_new = h_base
active = learned_gate >= threshold
if any(active):
    h_new[active] += expensive_residual(x[active], h[active])
```

实际实现尽量减少 Python 调度和动态索引开销。

### 2.3 训练与推理一致性

- 允许：Straight-through 估计、可微门控、Gumbel-Softmax、受控策略梯度、计算成本正则化。
- 训练使用稠密计算估计门控梯度时，**如实统计稠密训练成本**，不将稀疏推理成本当作训练成本。
- 验证软门训练与硬门部署的准确率差异，硬门下降时不以软门成绩代替。

### 2.4 计算成本约束

```
L_total = L_task + λ × expected_compute
```

计算成本项与真实执行路径成本明确关联。记录各路径实际执行次数、每条路径乘加运算量、
门控自身成本、动态调度成本、路径利用率、准确率变化、实际 CPU 速度变化。

## 3. 正交保护记忆（P5）

研究具有约束的记忆更新方向（不直接进入 M3 主模型，仅独立算子验证）：

```
P = I - K(KᵀK)†Kᵀ          # 正交投影，消灭受保护键方向
u = Pk                      # 新键的正交分量
S_new = S + (v - Sk)uᵀ / (uᵀk)
```

满足数学条件时，该更新保持受保护键对应读取结果，同时修改新键关联。

必须验证：新关联写入、旧关联保持、相似键数值稳定性、键空间饱和行为、
分母趋近 0 的安全退化、旧关联主动修改的解除保护、正交投影计算成本。

## 4. 最小充分神经状态（P2）

```
S = f(X),  Y = g(S)
```

目标（信息瓶颈思想，但**互信息难以直接准确估计，不使用未经验证的代理冒充**）：

```
maximize:  information(S, Y)
minimize:  information(S, X), state_dimension, computation_cost
```

使用可观测代理指标：状态维度、有效秩、奇异值分布、压缩后能力保持率、
隐藏状态可解码性（线性探针 R²）、状态扰动敏感度、分量关闭后的能力变化。

候选状态维度：16 / 32 / 64 / 96 / 128。禁止使用人工选定的正确答案作为神经状态。

## 5. 数学公式自动发现（P6）

- 允许算子：+ - *、矩阵乘法、逐元素乘法、低秩投影、对角变换、外积、内积、
  归一化、sigmoid、tanh、softplus、ReLU、残差更新、可学习门控。
- 禁止：不受约束的任意 Python 代码执行。
- 候选生成：预定义数学语法、表达式树、受控变异、算子替换、表达式简化、常数搜索、候选组合。
- 淘汰：维度错误、非有限值、梯度无法传播、数值发散、因果性破坏、样本状态泄漏、
  计算复杂度超限、硬编码答案路径、与已有候选数学等价且无实际优势。

## 6. 效率等级（P11）

| 等级 | 同等能力下计算量减少 |
|---|---:|
| E1 | ≥20% |
| E2 | ≥50% |
| E3 | ≥90% |
| E4 | ≥99% |
| E5 | ≥99.9% |

仅适用于经过明确验证的任务范围。某项复制任务达到 E3，不意味着通用智能达到 E3。

## 7. 指标口径（P0.5）

严格区分：
- 内容准确率（content_accuracy）
- 完整答案且自然 EOS 准确率（full_answer_natural_eos）
- Macro accuracy
- Micro accuracy
- 反事实成对正确率（查询对象不变，只改变事实）
- 查询切换（独立测试）
- 重复输出率
- 超长输出率
- 立即 EOS 率

所有正确率提供 n 与 ok。固定验证面板成绩不当作独立封存集成绩。
