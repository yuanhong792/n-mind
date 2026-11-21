# N-MIND R1 架构：实际实现与数学

> 本文档描述**代码里真实存在**的结构（不是愿望清单）。与最初提示词的差异在"与原方案的偏差"一节明说。

## 1. 一次 step 的数据流（core/architecture.py NModel.step）

```text
observe → Encoder → 误差门控(上一拍预测 vs 本拍真实) → Workspace.read
        → BlockRouter 选 Top-K 块 → NCell 更新被选块（其余保持）
        → 被激活块竞争写入 Workspace → Predictor(z_hat) → ActionHead(logits)
```

每一步都是可复现的纯函数（推理态无隐藏状态写进权重）；反塌缩负载偏置只在训练态更新。

## 2. N-Cell 数学（core/ncell.py）

GRU 风格门控，**每个块一份独立权重**（这是稀疏省算力的前提）：

```text
g  = σ(W_g·[z; h; w] + b_g)            # 门
c  = tanh(W_z·z + W_h·h + W_w·w + b_c) # 候选状态
h' = (1-g)⊙h + g⊙c                     # 门控残差
h  = h' 若该块本步被激活，否则 h 保持
```

- 稠密路径（`forward`）：全部 N 块 einsum 批算，`active` 掩码选择保留谁。
- 稀疏路径（`forward_sparse`）：只对选中的 (样本, 块) 对做 bmm，未选块原样保留。两路径数学等价（tests/test_ncell.py 钉死）。
- Top-K 的离散选择用直通估计：前向硬掩码、反向软掩码，不假装有完整梯度。
- 与 GRU 的相似性是**刻意声明**的：不宣称新的循环单元数学；创新主张在"块稀疏复用 + 预测误差门控 + 共享工作空间"的组合上，由消融实验裁决。

## 3. BlockRouter（core/router.py）

输入 `z`（事件潜状态）、块状态 `h`、工作空间读出 `w`、预测误差 `err`、不确定度 `unc`，输出每块分数 → Top-K 索引 + 软权重。训练时按路由概率做负载均衡（`aux_balance_weight`），负载统计走 `block_load_count` 缓冲。

**反塌缩偏置**：训练态按批内负载更新路由偏置，推理态冻结——保证"同权重+同输入→同输出"的复现验收不被偷偷破坏。

## 4. 全局工作空间（core/workspace.py）

固定 8 槽 × 64 维。`read` 用注意力掩码聚合（空槽被掩掉），`write` 是**竞争写入**：被激活块的载荷按效用评分抢槽位，挤出的旧内容记入 `overflow`（消融实验可直接统计）。压力指标 = 占用率。

## 5. 双头输出与预测误差

- `Predictor`：`z_hat = f(z, action_emb(a_prev), w)`；下一拍用 `normalized_error(z, z_hat)` 生成误差信号，供路由与治理器使用。第一步无历史，误差取 0.5 基准。
- `ActionHead.decide(ctx, w, mem)`：块状态池化 `ctx` + 工作空间读出 + 记忆向量（16 维，缺省零 = "无长期记忆"消融条件）→ 4 动作 logits。

## 6. 训练目标（learning/losses.py，5+1 项）

```text
total = CE(动作) + pred·MSE(z_hat, z) + var·方差正则
      + balance·负载均衡 + entropy·路由熵 + probe·记忆探针CE
```

探针回答"目标最初在哪一格"（6×6=36 类），只从循环状态 `ctx` 读，不读 SQLite、不看 oracle 变量——探针准确率是"网络自己记住了"的直接证据。遮挡期的输入里没有目标，所以遮挡段决策准确率 = 必须靠内部记忆才能拿到的分数。

## 7. 记忆与运行时（memory/、runtime/）

- SQLite 情景记忆（FTS5 全文检索 + 向量上下文）、事实表、巩固器（只写事实表不动神经参数）、墓碑删除。
- 治理器（runtime/governor.py）：单步耗时/RSS/递归深度预算，超限降级并计数。
- 云端策略（bridge/policy.py）：模式→敏感→白名单→预算→批准五道门，任一不过即拒绝；审计写 `results/cloud_audit.jsonl`。

## 8. 与最初提示词（FastCell/SlowCell 方案）的偏差

| 提示词提案 | 实际实现 | 原因 |
|---|---|---|
| FastCell + SlowCell 双时间尺度 | 单时间尺度 N-Cell 块集成 + Top-K 路由 | 先证明最小可训练原型；双时间尺度列为后续消融（M3 后） |
| RecursiveCore 固定 2/4/8/16 轮迭代 | depth=1 起步，governor 有 max_recurrence_depth | 深度递归与稀疏路由的组合需要先在 depth=1 建立干净基线，避免分不清收益来源 |
| HaltingGate 自适应停止 | evaluate 的 budget-难度对照已有，门控未训练 | 同上：先固定预算，再引入门控（提示词自身要求此顺序） |
| EpisodicMemory 第二阶段 | 已实现且可开关（use_memory / use_memory_input） | SQLite 路线成熟，提前就位便于消融 |

## 9. 参数量（doctor 实测，总 282,589）

blocks 197,632（8 块 × 门控循环权重）；其余为 encoder / router / workspace / predictor / head / 嵌入。打印各模块占比用 `NModel.n_params()`。
