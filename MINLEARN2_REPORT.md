# MINLEARN-2 受控训练报告

状态：已执行授权的512步训练；不自动追加训练。Stable checkpoint未修改。

## 训练

A/B/C/D各128步，实际 optimizer.step=512，随机种子20261130，网络请求0，sealed未进入训练。四个候选checkpoint均已隔离保存。

训练进程在四臂checkpoint全部写入后，于严格UTF-8评测阶段遇到非法预测byte并退出；因此本轮训练loss曲线没有持久化，未进行重复训练。该记录缺口已明确保留。

## 评测

修复后的评测器显式记录非法UTF-8，不再让解码异常中断：

- A/B/C/D train/dev/sealed完整答案exact：均为0
- EOS accuracy：均为1.0
- Unicode character accuracy：0
- invalid_utf8：1.0

这表明当前128步下模型主要输出不可逆的非法UTF-8 byte序列，不能宣称16题拟合、泛化或Router带来能力提升。

## 解释

Soft Mixture已证实Router梯度可达，但“有梯度”没有转化为可解码答案。当前主要瓶颈位于输出协议/byte序列训练与EOS前答案生成路径；需要后续独立诊断，不能追加本轮预算掩盖失败。

Gate-B 005继续暂停。任何下一轮训练需新的独立授权。

