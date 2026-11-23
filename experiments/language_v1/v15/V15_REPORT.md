# V1.5 报告

状态：ZERO_STEP / NOT_TRAINED。新训练预算未获批准，optimizer.step=0。

## 已完成

- 使用真实 `candidate_v14_main.pt`，父 checkpoint SHA-256 已登记。
- 使用与 V1.4 一致的 Byte-Level BPE 训练词表；有效 token 数和词表哈希已登记。
- 对四个封存实体绑定题计算了“红色/绿色”的完整候选条件对数概率，记录了原始 token ID、逐 token logprob 和总分。
- 逐 token 保存 N-Cell 状态、Workspace 状态、LM Head 输入范数和输出 top-5 分布。
- Baseline 保留 `h.mean(dim=1)`；Variant 仅增加 `query_readout_proj`，随机初始化，未执行训练。
- 旧 checkpoint 严格加载通过；Variant 仅缺少预期的两个新增层参数。
- 回归测试：29 passed；1 个既有 Workspace warning，不影响通过。
- Stable 未修改；Stable SHA-256：`B5B2BE048AD203F4784BE84D9B9F9496B4015A3C7DEB61E5A6DD3B7464405D11`。

## 诊断结论

Baseline 四题准确率为 0.50，但四题候选排序均偏向同一个颜色，且人物/属性交换没有使条件概率排序反转；这不构成实体关系理解。中间状态确实变化，但变化未有效传递到答案读出。该证据支持“平均读出是可疑瓶颈”，但不能排除训练监督和数据设计因素，因此 Variant 必须单独训练后再作因果判定。

Variant 零步准确率同为 0.50，且绝对 logprob 受随机新增层影响；这是预期的未训练状态，不是改进结果。

## 未运行

L1 单句生成完整基准、L3 反事实率、L4 多轮上下文、EOS 完成率、旧技能保持率的正式封存评估：`NOT_RUN`。它们已列入后续训练协议，不能用当前零步结果替代。

## 训练授权边界

预登记为两组各 256 步、总计 512 个 optimizer.step；当前累计 0。未获得新预算批准前不得执行训练脚本。
