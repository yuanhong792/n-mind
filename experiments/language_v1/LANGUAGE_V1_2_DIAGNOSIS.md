# Language V1.2 诊断

已执行零步 Candidate 复评。Gate-1 历史训练 loss 无法恢复，标记 HISTORICAL_TRAIN_LOSS_UNAVAILABLE。BPE 往返正常；问题是 LM Head 会选择未定义 merge 的词表槽位，已新增主动词表约束生成器。即使修复后文本合法，也不能据此宣称神经语义能力。Router 语言梯度仍为0；旧技能保持 RETENTION_INCONCLUSIVE。optimizer.step=0，Stable未修改，网络请求=0。
