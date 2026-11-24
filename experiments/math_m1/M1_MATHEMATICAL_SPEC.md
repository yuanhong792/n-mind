# M1 数学规格：固定状态神经记忆算子

记号：\(k_t\in\mathbb R^{d_k}\)、\(v_t\in\mathbb R^{d_v}\)、\(q_t\in\mathbb R^{d_k}\)；状态 \(S_t\in\mathbb R^{d_v\times d_k}\)；读出 \(o_t=S_tq_t\in\mathbb R^{d_v}\)。所有键做 L2 归一化，门由可训练投影产生并压到合法区间。

## 1. 基础 Delta 矩阵关联记忆（算子 A）

先衰减，再基于衰减后状态计算残差：

\[
\tilde S_t = \alpha_t S_{t-1},\qquad
\hat v_t = \tilde S_t k_t,\qquad r_t=v_t-\hat v_t
\]

共享门 Delta 更新：

\[
S_t = \tilde S_t + \beta_t\, r_t k_t^\top
     = \alpha_t(I-\beta_t k_tk_t^\top)S_{t-1}+\beta_t v_t k_t^\top
\]

本项目实现允许 \(\alpha_t=\mathrm{sigmoid}(a_t)\) 或逐通道 \(\alpha_t\in(0,1)^{d_k}\)。基础 M1-B 使用共享标量 \(\beta_t\)，便于和 M1-C 单变量比较。

预期数学性质：
- 相同键重复写入应更新旧值，而非简单累加；
- 正交键写入应尽量互不干扰；
- 非正交键干扰随余弦相似度上升；
- 键范数归一化后多次写入状态范数不爆炸；
- 批量更新与逐样本更新一致；
- 增量推理与整段 teacher forcing 一致。

## 2. 解耦擦除/写入门（算子 B，来源 Gated DeltaNet-2）

共享门版本（M1-B）：

\[
S_t = (I-k_t(\beta_t k_t)^\top)D_tS_{t-1}+(\beta_t v_t)k_t^\top
\]

独立门版本（M1-C）：

\[
S_t = (I-k_t(b_t\odot k_t)^\top)D_tS_{t-1}+k_t(w_t\odot v_t)^\top
\]

其中 \(b_t\in[0,1]^{d_k}\) 为 key 侧擦除门，\(w_t\in[0,1]^{d_v}\) 为 value 侧写入门，\(D_t=\mathrm{diag}(\alpha_t)\) 或 \(\alpha_tI\)。

**本轮单变量原则**：M1-B 与 M1-C 只改变门控结构；d_model、状态维度、控制器、数据、loss、课程、预算、评测全部一致。短卷积、不确定性门、不同读出机制本轮不叠加进主模型。

## 3. 不确定性关联记忆（算子 C，来源 KDN）

对角近似：

\[
\hat p_t=\alpha_t^2\odot p_{t-1}+\omega_t
\]

\[
\kappa_t=\frac{\hat p_t\odot k_t}{r_t+\sum_i \hat p_{t,i}k_{t,i}^2}
\]

\[
S_t = \hat S_t+\kappa_t(v_t-\hat S_t^\top k_t)^\top
\]

\[
p_t=\left(\hat p_t^{-1}+\frac{\mu}{r_t}k_t\odot k_t\right)^{-1}
\]

本轮只做算子级验证，记录 \(p_t,\kappa_t,r_t\)、旧记忆保持与新信息适应速度；未通过前置验证不得进入候选架构。

## 4. 局部因果卷积（算子 D）

核大小 \(K\in\{1,2,4,8\}\)，深度可分离卷积：

\[
x'_t = \sum_{j=0}^{K-1} h_j x_{t-j}
\]

只读取当前与过去 token，右填充仅在训练批次内用于对齐，未来位置权重为 0。验证必须覆盖：
- 因果性：改变未来输入不影响当前输出；
- 批量与逐样本一致；
- 无padding状态污染。

首轮 M1-B/M1-C 固定 \(K=1\)，卷积仅作为后续单变量实验候选。

## 5. 控制器与输出

候选模型共同骨架：

```text
token embedding
→ lightweight recurrent controller (GRU cell 或 TinyDelta cell)
→ matrix associative memory update
→ learnable fusion of controller state and memory read
→ output head
```

控制器状态与矩阵记忆状态都是固定大小。推理时状态不随序列长度增长；训练时反向传播仍按展开序列占用显存，不能宣称训练内存恒定。

## 6. 参数与状态核算

记录以下量，禁止用无作用参数填满区间：
- `trainable_params`：可训练参数数；
- `state_elements`：推理时循环状态元素数；
- `memory_state_elements=dv*dk`；
- `model_bytes`：checkpoint 文件大小；
- `cpu_token_latency_ms` 与 `generation_tokens_per_second`；
- `train_tokens_per_second`；
- `peak_rss_mb`；
- `peak_cuda_mb`：本轮 CUDA 不可用，记 NOT_AVAILABLE。

参数目标 80000–130000；若不落区间，必须如实报告，不得伪造参数。

## 7. 监督与指标

- 主指标：自然 EOS 的自由生成完整答案精确匹配。
- 辅助指标：完整答案条件概率 \( \log P(answer+EOS\mid prompt) \)，不含 prompt 自身概率。
- 立即 EOS、空输出单独统计；空输出不算成功。
- 验证集选 Best；封存集只在最终评测使用。
- 每项任务计算固定答案、首数字、末数字、最近赋值、随机猜测等捷径基线。
