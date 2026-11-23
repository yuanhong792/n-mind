# Minimal Learnability and Neural Path Optimization

状态：零训练分析完成；optimizer.step=0；Gate-2.4预算不复用。

## 已验证问题

在16条严格重复配对样本上，Tokenizer roundtrip 通过。对随机初始化 N-Cell 做 backward-only 梯度审计：

- Router score MLP grad_norm：0
- Head grad_norm：0.364785
- block_to_ws grad_norm：0.006516
- Workspace grad_norm：0.003770
- N-Cell block gradients 非零
- parameter_changed：false，optimizer.step：0

Router 梯度为零是确定性路径问题：当前 forward 使用 route.indices 的 hard Top-K 索引，损失只经过被选 block 的计算路径，不经过 Router score 概率；Router load bias 也不是可微语言损失路径。该结论来自实际 grad_norm，不是推断。

## Head / Workspace

当前 head(h.mean(dim=1)) 将所有 block 状态平均，可能稀释实体绑定。已实现但未训练的隔离候选：

- SelectiveBlockRead：选定 block + Workspace summary
- FullBlockRead：保留全部 block identity + Workspace summary
- TinyGRUBaseline：随机初始化小型 GRU 对照

结构 shape 测试通过；尚未声称任何候选改善能力。

## 评测协议

主指标改为：

- 完整答案拟合
- 未见组合
- 反事实跟随率
- 逐中文字符准确率
- EOS 准确率

UTF-8 首字节命中被标记为无效语义指标，因为中文答案共享公共字节前缀。

## 标签置乱

100% 标签变更率本身不足以证明有效。下一轮需冻结并审计：

- P(label | true position)
- P(label | queried entity)
- P(label | entity order)
- P(label | last-mentioned position)
- 反向映射可预测性与 mutual information

在完成该审计前，B 对照只标记为“样本级打断通过，条件独立性待确认”。

## 建议候选

每个方案只改变一个主要变量：

1. Router soft mixture：保留所有 block 的可微概率加权，先验证 Router grad_norm > 0。
2. Selective block + Workspace readout：只替换输出读取路径。
3. Binding slot：为实体写入 key/value 绑定槽，查询实体时读对应槽。

拟议新训练预算：每臂128步，四臂总计512步；需独立用户授权，不能沿用 Gate-2.4预算。

当前结论：最小可学习性尚未证明；首先修复/验证 Router 可微路径和输出读取，再考虑新训练。

