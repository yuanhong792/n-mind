# M3.1 历史实验修复与零梯度复验报告

分支：`experiment/minimum-sufficient-interaction-m4`
执行日期：2026-11-24
性质：**零 optimizer.step** 审计。不修改 M3 原始报告、数据、权重；所有结论可追溯到新生成的 JSON。

---

## 1. 执行摘要

任务书列出的 M3 九项问题，本轮对其中可静态/零梯度验证的六项做了实测审计。
**三项被证实为严重问题，一项被证伪，两项需训练后验证。**

| 问题 | 结论 | 证据 |
|---|---|---|
| 训练监督协议 | **已证实严重**：89.64% 监督位置是 prompt | `M31_PROTOCOL_AUDIT.json` |
| 反事实数据捷径 | **已证实严重**：400/400 查首变量，配对捷径 100% | `M31_COUNTERFACTUAL_AUDIT.json` |
| 数据分布 A5_repeat=0 | **已证实**：训练集 0 条，val 240 / sealed 60 | `M31_LEAKAGE_AUDIT.json` |
| fact_group 语义隔离 | **已证实不充分**：规范化语义 train/val 重合 47、train/sealed 54 | `M31_LEAKAGE_AUDIT.json` |
| 计算成本固定长度估算 | **已证实**：实际 T=42.88、padding 42%、成本虚增 1×–8× | `M31_COMPUTE_ACCOUNTING_AUDIT.json` |
| Checkpoint 标记错配 | **未证实**：16 个权重全部可加载，8 组 Best/Final 角色与步数一致 | `M31_CHECKPOINT_AUDIT.json` |
| 软门训练/硬门推理不一致 | 需 M4 训练后验证（M3 已测 -11.76pp） | M3 报告第 7 节 |
| 连续符号生成困难 | 需 M4 正控制验证 | 未授权 |
| Transformer 能力上限未证明 | 需 M4 成本解剖与公平实验 | 未授权 |

---

## 2. 监督协议（3.1）

审计 8161 条 M3 样本（train 5575 / val 1469 / sealed 1117）：

- M3 原协议中 **prompt 监督位置占 89.64%**，答案与 EOS 只占约 10%。
  这解释了 M3 面板高分任务多为需少量答案 token 的绑定类，而复制/反转类失败——
  模型大部分梯度用于语言建模而非任务输出。
- M3 的 `w[sep_idx+1]=2.0` 索引**本身正确**（w[j] 对应目标 seq[j]，logits 在 j-1 预测），
  错误不在 off-by-one，而在 prompt 赋权与 EOS 权重 0.25。
- M3 用 batch 全局归一化 `(loss*w).sum()/w.sum()`，任务书要求按样本归一化后对 batch 平均。
- 全部样本无 SEP 缺失、无答案缺失、无 EOS 缺失；max_len=64 未触发截断。

**M4 实现**：`audit/m3_protocol_repair.py` 定义 Protocol A（答案+EOS，权重 1.0，
prompt/SEP/PAD 为 0，按样本归一化）与 Protocol B（全序列，独立消融候选）。
24 项验证中的第 18/19 项将检查监督位置严格对齐与自然 EOS。

**历史权重不能追溯修复**：M3 权重是在旧协议下训练的，只能用新协议重新评测现有能力，
报告已明确标注，不把重评测当作"用 Answer-only 重新训练"。

---

## 3. 反事实捷径（3.2）

M3 的 400 对反事实：

- 查询位置分布：`first=400, middle=0, last=0`，首变量占比 **100%**。
- **配对首变量捷径命中率 100%**：不看 `GET` 指令、直接输出第一行赋值数字，base 与 cf 两条同时正确。
- `query_switch` 覆盖率为 0；无关事实单变量修改覆盖为 0；M3 的 cf 是成对交换赋值。
- 变量数分布集中在 2 变量，缺少 4/8 变量的中间/末位查询。

**结论**：M3 报告中的反事实成对 0.75（R2）与 0.5525（M3-B）**不能作为事实绑定证据**，
因为存在 100% 的平凡解。

**M4 必须覆盖**：查询首/中/末位均衡（每类 ≥25%）、同查询目标单值修改、
同查询无关事实修改、事实不变查询切换、赋值顺序交换保持语义；
配对两条都正确才计通过，四类指标分别统计。

---

## 4. 数据分布与语义隔离（3.3）

- `A5_repeat`：train **0**，val 240，sealed 60，generalization 900 → 训练前置知识完全缺失。
- 规范化语义键（排序变量赋值 + 查询对象）重合：train/val **47**，train/sealed **54**；
  字符串 prompt 重合为 0，说明 M3 的"零重合"只成立于字符串层。
- 反事实事实族中，存在同一 canonical 语义对应多个 `fact_group` 字符串的情况，
  证明 `fact_group` 不同不能作为语义隔离依据。
- 查询位置分布在 bind 类任务中偏向首/末位，中间位不足。

**M4 处理**：数据生成器改为按规范化语义族做全局去重与分割；
所有核心任务保证训练样本≥32；未见任务与未见组合分别标记；
A5_repeat 纳入训练；查询位置按变量数均衡轮转。

---

## 5. 成本统计（3.6）

- 原始 curve 用全局累计步数乘固定 seq=32，造成 **1×（第1个 run）到 8×（第8个 run）** 的系统性虚增。
- 抽样 32 个真实 batch：平均张量长度 **T=42.88**（非 32），有效 Token 占比 **0.58**，
  即约 42% padding 仍参与张量运算；每 batch 有效监督目标 **758** 个。
- 解析 MAC、Profiler 热点、真实墙钟三种口径必须分列：
  M3 中 R1 的 Python 逐 token 循环占 44.77%，说明 CPU 上实际瓶颈未必是 MAC。

**M4 处理**：`compute_accounting_m4.py` 逐 batch 记录实际 Batch Size、有效 Token、
Padding Token、实际张量长度、有效监督 Token、各模块执行次数、累计成本，
并把数学估算 / Profiler / 墙钟三类结果分开输出。禁止用固定长度公式冒充实际成本。

---

## 6. Checkpoint 证据链（3.5）

- 16 个 checkpoint 全部加载成功，无错误；`errors=0`。
- 8 组 Best/Final：角色标签与步数关系全部一致（`role_labels_match_steps=True`），
  **未发现"加载 Final 标记 Best"**。M3-B 的 Best/Final 张量完全相同是因末点即最优点，已如实记录。
- 小面板（每任务 5 条，共 85 条）重评测 macro 与 meta 声明 macro 的平均绝对差 0.0341、
  最大 0.0839。差异来自**面板样本量不同**（meta 用 314 条），不是证据链错配；
  此项在报告中澄清，避免误判。
- 每个权重记录实际路径、文件 SHA256、张量 SHA256、meta 角色、种子、本模型步数。

---

## 7. 未修复与缺失证据

- 软门/硬门不一致、连续符号生成、Transformer 上限：需 M4 授权训练后验证，本轮 `NOT_RUN`。
- 历史权重无法追溯修复协议错误：标 `NOT_RETRAINED_UNDER_PROTOCOL_A`。
- 能耗：`NOT_AVAILABLE`（无硬件功耗测量）。
- 本报告不修改 M3 原始报告与任何 checkpoint。

---

## 8. 交付物

`audit/` 下五个脚本：`m3_protocol_repair.py`、`counterfactual_audit.py`、
`semantic_leakage_audit.py`、`checkpoint_audit.py`、`compute_accounting_audit.py`；
证据 JSON：`M31_PROTOCOL_AUDIT.json`、`M31_COUNTERFACTUAL_AUDIT.json`、
`M31_LEAKAGE_AUDIT.json`、`M31_CHECKPOINT_AUDIT.json`、`M31_COMPUTE_ACCOUNTING_AUDIT.json`。
