# Gate-2.3 零训练因果诊断报告

## TL;DR

Gate-2.2 结果已从本地 checkpoint 和逐题 JSON 复现，未执行新的 optimizer.step。main 与标签置乱模型在 240 条 sealed 上逐题预测完全相同率为 100%，都输出同一个固定答案；两者 exact-match 均为 16.67%，反事实均为 0/80。lr=0 的 eval 重复前向逐位稳定，train 模式只改变 `block_load_count` buffer。反事实状态轨迹的 h/Workspace 表征几乎完全相同，支持“绑定信息没有进入有效状态/读出”的诊断，但尚不足以区分 Router、Workspace 写入和 Head 的唯一因果责任。

状态：`REPRODUCED_DIAGNOSED`；神经训练步数：0；Gate-2.4：`PENDING_USER_APPROVAL`。

## 复现与逐题审计

输入 checkpoint、评测 JSON、预算、预登记和 Gate-2 报告均已核对并记录 SHA。旧报告实际位于仓库根目录 `GATE22_REPORT.md`，输入审计已按真实路径修正。

main 与乱标逐题预测相同率：100%（240/240）。main 与乱标的预测分布均为单一固定答案，说明 16.67% 主要是固定答案/类别先验命中，而不是条件实体理解。逐题错误类型、prompt hash、pair ID 和两模型原始输出见 `GATE23_REPRO_PER_ITEM.jsonl`。

## Loss 分解

答案 span+EOS mask 的 32 条抽样断言通过，prompt token 不进入 mask。主模型 sealed teacher-forced 答案分解：颜色类 24 条平均 NLL 1.1162；位置类 24 条平均 NLL 2.4798。位置答案明显更难，且自由生成仍固定输出，说明 teacher forcing 的局部 token 预测不能替代条件生成能力。完整逐样本结果见 `GATE23_LOSS_DECOMPOSITION.json`。

## lr=0 漂移

同一输入、eval 模式重复前向最大差异 0；eval buffer 不变。train 模式只观察到 `block_load_count` 变化，参数未变化。Gate-2 记录中的 lr0 loss 变化不能归因于参数学习；其主要解释是训练过程中使用的样本/状态统计路径不同。固定 batch 的复现证据见 `GATE23_LR0_DRIFT.json`。

## 内部状态

对 10 对事实位置交换题采集了 prompt 末状态。h cosine 约等于 1，Workspace cosine 约等于 1，配对 logits 差异约为 1e-5 量级；这与模型对两种事实顺序给出同一固定答案一致。该结果支持 H4 “当前状态不保留可用绑定关系”，但不能单独证明是 Router、Workspace 写入或 Head 的唯一责任。直接 Router top-k 时间轨迹和形状保持的 Head/block_to_ws 替换本轮未执行，原因见 `GATE23_REPRESENTATION_DIAGNOSTICS.json`；不把缺失实验包装成已证实。

## 假设判定

| 假设 | 判定 | 证据 |
|---|---|---|
| H1 mask/shift 错误 | DISFAVORED | Gate-2.2 mask 审计和 Gate-2.3 逐 token 断言通过 |
| H2 数据/类别捷径 | SUPPORTED | main/乱标逐题相同 100%，固定单答案 |
| H3 评价协议错误 | DISFAVORED | checkpoint 重载后同协议复现 |
| H4 无可读绑定表征 | SUPPORTED | pair 状态 cosine≈1、logit 差异≈0 |
| H5 有表征但读写/Head失败 | UNRESOLVED | 尚未完成部件替换/探针因果实验 |
| H6 负对照无效 | SUPPORTED | 主/乱标完全相同，控制未分离；下一轮需更强对照 |
| H7 预算不足 | UNRESOLVED | 训练行 surface loss 下降，但被 H2/H4 解释，不能单独归因预算 |

## 软件缺陷与修复

本轮没有发现需要修改 N-Cell、Workspace 或评测协议的软件缺陷；零训练诊断脚本只读 checkpoint。此前已修复的 L0/L1 采样和答案 span mask 保持不变。没有修改历史 sealed 数据或 stable checkpoint。

## Gate-2.4 候选（仅预登记）

1. A：增加独立的答案条件/slot 读出头，验证“状态有信息但共享 Head 无法读出”。
2. B：结构化位置答案目标与自然语言答案双头，验证“字节自回归目标压制关系学习”。
3. C：成对反事实一致性目标，验证“显式关系区分是否能打破固定输出”。

三者均需新的训练预算、独立未来 sealed 集和有效乱标对照；状态均为 `PENDING_USER_APPROVAL`，本轮未执行。

## 复现命令

```powershell
$env:PYTHONPATH = "src"
& "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe" experiments\language_v2\gate23\run_gate23.py
& "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe" -m pytest -q
```
