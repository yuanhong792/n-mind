# MINLEARN-2 Router可微学习与16题拟合报告

状态：PENDING_USER_APPROVAL。仅完成源码、零训练计算图诊断、数据预登记和测试；optimizer.step=0。

## 结论

原始 Hard Top-K 路径的 Router score MLP 语言梯度为0，原因是 route.indices 经 hard top-k 后作为离散索引进入 forward_sparse，答案 Loss 没有经过 Router 概率。

新增 Soft Mixture 路径保留 N-Cell、Workspace 和原始 Head，使用 router softmax 概率加权全体 block 状态与 Workspace 写入；实际 backward 测试确认 Router score MLP 获得非零梯度。将 soft 权重 detach 后，Router 梯度归零，而 Head/N-Cell 等路径仍可产生梯度，证明梯度来自答案 Loss 的真实计算图，不是人工附加项。

这只是梯度可达性证据，不是训练后能力证据。

## 数据

- train16：16条，含实体、位置、问询实体、陈述顺序和反事实配对
- dev：独立语义组
- sealed：独立语义组
- Tokenizer roundtrip：通过
- optimizer.step：0
- 训练/封存答案隔离：通过预登记约束

标签独立对照不再以 changed_fraction 单独判定；已记录条件分布、独立随机种子和反向映射审计要求。

## 四臂预登记

- A HARD_TOPK_ORIGINAL：128步
- B SOFT_MIXTURE_TRAINABLE：128步
- C SOFT_MIXTURE_ROUTER_FROZEN：128步
- D SOFT_MIXTURE_LABEL_INDEPENDENT：128步
- 总预算：512步
- 墙钟建议上限：60分钟
- 当前训练授权：0，状态 PENDING_USER_APPROVAL

## 测试

全仓 pytest：276 passed，3 skipped，0 failed，0 errors，1 warning。3个 skipped 仍为 tkinter GUI 显示环境问题。

## 未完成项目

完整答案拟合、未见组合泛化、反事实跟随率、逐中文字符准确率、EOS和自由生成只能在获得新的独立训练授权后测量。本轮不得把梯度非零称为神经能力通过。

Stable checkpoint、Gate-2.4候选和历史 sealed 数据均未修改。

