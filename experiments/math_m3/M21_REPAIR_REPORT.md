# M2.1：M2 历史实验缺陷修复报告

本轮只做历史协议复现与审计，**不修改任何 M2 原始结果**。所有新结论写入 `experiments/math_m3/`，
原始 `experiments/math_m2/` 报告保持不动。缺权重标 NOT_AVAILABLE，不伪造。

## P0.1 SEP 训练/推理协议交叉评测

M2 缺陷：Legacy 模型用无 SEP 训练，但评测函数默认 SEP 输入。交叉评测结果（R1 GRU，
各 256 step，同数据同 seed；4 个 checkpoint 均本地存在并加载成功）：

| 训练协议 | 推理协议 | val macro | sealed macro |
|---|---|---:|---:|
| Legacy | **Legacy**（公平） | 0.0836 | 0.0567 |
| Legacy | SEP（M2 误用） | 0.1108 | 0.0967 |
| SEP | Legacy | 0.0937 | 0.0660 |
| SEP | **SEP**（公平） | 0.4511 | 0.3540 |

**结论 1**：正确协议下 SEP 提升依然成立，甚至更强：Legacy→Legacy 0.0836 vs SEP→SEP 0.4511，
优势 +0.3675（封存 +0.2973）。M2 报告的 0.1108 实为 Legacy 训练→SEP 评测的错配值。

**结论 2**：历史 Legacy/SEP 的 Best 均由 SEP 推理选出，存在选模污染。本轮同时报告 Final；
本例 Best 与 Final 恰好同权重（macro 相同），但不能据此认为选模公平。

细节见 `M21_CROSS_PROTOCOL_EVAL.json`（含 n/ok/per_task/立即EOS/样例）。

## P0.2 真正受监督位置的冲突重统计

M2 缺陷：原扫描把 prompt 内部未受监督的延续字符也计入冲突。本轮只统计 loss 权重>0 的
预测位置（answer token 与 EOS），在 49,035 条历史样本上重统计：

| 指标 | Legacy | SEP |
|---|---:|---:|
| 受监督位置总数 | 约 13 万 | 约 13 万 |
| 冲突前缀数量 | **128** | **0** |
| 受影响监督位置 | **299** | 0 |
| 受影响位置比例 | **0.23%** | 0 |
| 条件熵（经验估计） | 0.9728 bits | 0 |
| 理论最低交叉熵 | 0.9728 bits | 0 |
| 真 prompt 多答案歧义 | 见 JSON | 见 JSON |

**结论 3**：M2 原报 1663 个冲突被夸大（把未监督位置算进去），真实受监督冲突只有
128 个前缀 / 299 个位置（0.23%）。但 SEP 后真实冲突**结构性归零**，方向性结论不变。
冲突样本条件熵约 0.97 bits 说明即使有冲突，理论上的不可约错误上限也很低——
**监督冲突不是本任务学习困难的主要限制因素**，主要瓶颈在别处（见 M3 报告问题 3）。

细节见 `M3_SUPERVISED_CONFLICT_AUDIT.json`（含任务分布、冲突 token 频率、SEP 前后变化）。

## P0.3 增量推理审计

- M2 Transformer 完整前向 vs 逐 token 推理：最大差 **0.0**（一致），但实现是
  「保存历史 token 并重算」，复杂度 O(L²)，**不是标准 KV Cache**。
- 位置编码：位置 ≥ max_len 被 clamp 到 max_len-1，**确认碰撞**（实测位置 15/16/21 编码相同）。
- Padding：M2 模型有 valid_mask 保护，无 NaN/Inf 传播。
- 因果卷积：M2 主模型未使用卷积（仅算子验证），无缓存问题遗留。

**结论 4**：M2 的增量推理功能正确但低效；M3 参考 Transformer 必须实现真 KV Cache
并通过完整前向等价测试（见 math_verifier 第 15 项）。

详见 `M21_INCREMENTAL_AUDIT.json`。

## P0.4 反事实与距离测试口径修复

- **反事实**：M2 原实现的 B_swap 同时改变了赋值与查询值，属于「属性交换 + 查询切换」混合。
  M3 严格定义：反事实 = 保持查询对象不变、只交换事实；查询切换作为独立测试。
- **距离**：M2 距离样本用随机干扰内容，未保证「相同事实/相同查询/不同距离」严格配对。
  M3 数据集按 `(fact_group, distance)` 生成严格配对，且干扰行不重新赋值目标变量，
  避免语义错误标签（见 dataset_m3.py 与 M3_DATA_AUDIT.json）。
- **历史值/当前值**：M3 C 组任务在输入中用不同符号显式区分历史快照与当前值。

## P0.5 指标口径统一

M3 起所有正确率统一提供 `n` 与 `ok`，并严格区分：
内容准确率、完整答案+自然 EOS 准确率、macro、micro、反事实成对正确率、
重复输出率、超长输出率、立即 EOS 率。固定验证面板成绩**不得**当作独立封存集成绩
（分别报告 val/sealed）。

## 修复状态汇总

| 项 | 状态 | 证据文件 |
|---|---|---|
| SEP 交叉评测 | DONE | M21_CROSS_PROTOCOL_EVAL.json |
| 受监督冲突重统计 | DONE | M3_SUPERVISED_CONFLICT_AUDIT.json |
| 增量推理审计 | DONE | M21_INCREMENTAL_AUDIT.json |
| 反事实/距离口径 | 修复进 M3 dataset | M3_DATA_AUDIT.json（待生成） |
| 指标口径 | 修复进 M3 evaluate | M3_EVALUATION.json（待生成） |
