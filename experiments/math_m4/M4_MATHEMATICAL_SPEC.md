# M4 数学规格说明

本文件冻结 M4 零梯度阶段使用的数学定义、成本口径与判定边界。
它不是“已发现更高效机制”的声明；任何能力或效率结论必须以训练后评测为准。

---

## 1. 监督协议

统一序列：

```text
sequence = prompt_ids + [SEP] + answer_ids + [EOS]
```

**Protocol A：Answer-only**

- `w[j]=0`：`prompt` 与 `SEP` 作为目标的位置；
- `w[j]=1`：答案 token 与 `EOS` 作为目标的位置；
- PAD 权重为 0；
- `loss = mean_b(sum_j w[b,j] * CE(logits[b,j-1], seq[b,j]) / sum_j w[b,j])`；
- 禁止静默截断导致 SEP、答案或 EOS 缺失。

**Protocol B：Full-sequence** 仅作独立消融，不与 A 跨架构直接比较。

---

## 2. 参考注意力

标准多头注意力：

```text
Q = XW_Q, K = XW_K, V = XW_V
A = softmax(QK^T / sqrt(d_k) + M_causal) V
```

其中 `M_causal` 只允许当前位置访问不晚于自身的键。
参考实现仍执行稠密矩阵；有效三角区节省是解析量，不等于实际墙钟节省。

---

## 3. LRII：低秩信息交互

```text
Q_r = X U_Q,  K_r = X U_K
A_r = softmax(Q_r K_r^T / sqrt(r) + M_causal) V
```

严格区分：

1. 降低 Q/K 交互维度 `r < d`；
2. 分解投影矩阵 `W_Q ≈ AB`；
3. 对已算出的 score 矩阵做低秩近似。

只有第 1 种减少 QK 的 score MAC；第 2、3 种不得冒充省 QK 计算。
本轮主模型 M4-A 冻结 `r=16`，但任务能力未训练验证。

---

## 4. SII：选择性信息交互

历史表示分组，构造低成本摘要 `c_j = f_theta(K_j, V_j)`：

```text
s_j = q^T c_j
I = TopK(s_j)
A_I = softmax(Q K_I^T / sqrt(d) + M_causal(I)) V_I
```

约束：

- 摘要与粗检索必须带因果 mask；
- 当前未完成组只能看到已到达 token；
- Top-k 后不能回退构造完整 QK 再宣称省计算；
- 训练用软门、部署用硬门的差距必须单独报告；
- 能力以实际执行稀疏路径为准。

本轮算子层精确注意力跳过率 0.75；路由自身成本必须计入。

---

## 5. III：增量信息交互

```text
S_t = S_{t-1} + phi(k_t) v_t^T
z_t = z_{t-1} + phi(k_t)
y_t = phi(q_t)^T S_t / (phi(q_t)^T z_t + epsilon)
```

这是已有线性注意力方向，不宣称与 softmax attention 等价。
`phi` 候选：`relu`、`elu+1`、`raw`。`identity` 在本轮分母不稳定，淘汰。
固定状态成本随长度线性，但长序列信息干扰只是代理观察，不是能力证明。

---

## 6. 正交与信息压缩

```text
Z = X P
X_hat = Z P^T = X P P^T
```

若 `P` 为正交基，则 `P^T P = I`。注意：

- 低有效秩不等于无损压缩；
- 必须同时报告 Frobenius 误差、事实读取误差、位置读取误差、压缩/重构成本；
- 小矩阵下增量 QR 未必快于全量 SVD。

本轮 k=4 事实读取 MAE 0.6617，不能作为可用压缩；k=8/16/32 误差较低但仍未进入能力实验。

---

## 7. 数学等价优化

Answer-only 训练只需对监督位置计算输出 Head：

```text
hidden = neural_core(tokens)
supervised_hidden = hidden[supervised_positions]
logits = output_head(supervised_hidden)
```

等价性要求：logits、loss、梯度、参数更新、自回归生成一致。
本轮实测最大差异：logits `4.77e-07`，loss `0.0`，梯度 `3e-08`，更新 `7e-09`。
但 CPU 时间为负收益 `-3.42%`，因此淘汰为独立优化。

---

## 8. 成本口径

严格分离：

1. 解析 MAC：按矩阵乘法维度计算；
2. Profiler：实际算子 CPU 时间；
3. 墙钟：含调度、内存访问与 I/O 的真实耗时。

禁止使用：

```python
total_mac = steps * batch_size * 32 * per_token_mac * 3
```

冒充实际累计成本。必须逐 batch 记录实际 batch size、tensor length、valid tokens、
padding tokens、supervised tokens、模块执行次数与本模型累计成本。

---

## 9. 同等能力效率

```text
E_train = C_ref_train(Q) / C_candidate_train(Q)
E_infer = C_ref_infer(Q) / C_candidate_infer(Q)
```

前提：双方在同一任务集达到预登记能力门槛，且关键任务下限同时满足。
等级：E1 ≥20%，E2 ≥50%，E3 ≥90%，E4 ≥99%，E5 ≥99.9%。

未训练、未达门槛或不可比时标：

- `TARGET_NOT_REACHED`
- `NOT_COMPARABLE`

本轮所有 M4 候选均为 `NOT_RUN`，不授予任何等级。

---

## 10. 公式搜索约束

允许算子：Add、Subtract、MatMul、Dot、Outer、LowRank、Projection、Normalize、
Softmax、KernelMap、Gather、Reduce、Hadamard、Residual、Gate、Input。

候选必须通过：

- 静态维度检查；
- 因果性；
- 数值稳定；
- 梯度有效；
- 不访问标准答案；
- 不执行外部代码；
- 成本可估算；
- Pareto 初筛。

算子误差与 CPU 初筛不能替代训练后能力。本轮 29 个候选、17 个存活，
算子阶段 Pareto 为 F01 与 F17，均未经神经任务训练验证。
