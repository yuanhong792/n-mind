# V1.5 Training Report

状态：COMPLETED_AUTHORIZED_TRAINING。两组各执行 256 步，总计 512 步；总耗时 116.10s。

Baseline loss: 1.704358 -> 0.244844

Query-Attention loss: 4.691447 -> 0.232786

自由生成原文见 `V15_FREE_GENERATION.jsonl`。代表性输出包括：

- Parent：`小丽的书包是什么颜色？` → `红色。`
- Baseline：该封存题整体准确率 25%，仍有固定颜色偏置。
- Query-Attention：该封存题整体准确率 25%，反事实题未达到预登记门槛。

最终判定：`BINDING_FAIL`。Query-Attention 未优于 Baseline，也未突破固定回答“红色”问题。两组均未达到实体属性绑定标准。旧技能保持：`RETENTION_INCONCLUSIVE`，因为没有可用的同口径历史封存集；不得据此宣称无遗忘。Stable 未修改。

量化摘要：Parent accuracy=25%，Baseline=25%，Variant=25%；固定输出“红色”比例分别为 100%、62.5%、75%；EOS 完成率分别为 100%、87.5%、75%；UTF-8 解码异常分别为 0、1、2。Loss 下降不能替代自由生成泛化证据。
