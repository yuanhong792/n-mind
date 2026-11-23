# V1.5 零步诊断

见 JSON 原始证据。Baseline 使用 V1.4 Candidate；Variant 仅新增随机初始化的 query_readout_proj，未执行 optimizer.step。

判定规则：只有候选答案条件对数概率在实体、颜色交换和人物顺序变化下正确反转，才可称为实体绑定；中间状态变化本身不是充分证据。
