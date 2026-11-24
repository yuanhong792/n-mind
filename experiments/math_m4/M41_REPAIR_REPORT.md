# N-MIND M4.1 正式训练前深度修复报告

## 0. 当前状态

- 分支：`experiment/minimum-sufficient-interaction-m4`
- 基准提交：`caf7d48`
- 预检结论：`PREFLIGHT_PASS_AWAITING_AUTHORIZATION`
- `optimizer_step = 0`
- `training_status = AWAITING_AUTHORIZATION`

预检十项条件全部来自实际审计文件，不是硬编码 `true`。预检通过只表示**允许进入请求授权阶段**，不等于训练已授权，也不等于任何能力或效率结论成立。

## 1. 历史证据保护

`M4_MATH_TESTS.json` 的 24/24 PASS 未被覆盖。其 SHA256 与 `M41_BASELINE_SNAPSHOT.json` 一致：

```text
ff59cdbb68b8530a85bae4e0b9e184eedc694164a15a8a43cae6cbd87fe5e39a
```

`M41_PREFLIGHT_REPORT.json` 中重新核验了该哈希。旧 24/24 仍是历史记录；M41 的正式模型因果性审计另存于 `M41_CAUSALITY_AUDIT.json`。

## 2. 已完成的源码与科学验证修复

### 2.1 M4-A 低秩 KV Cache

修改 `models/low_rank_interaction.py`：

- transpose 后 K/V 形状为 `[B, heads, sequence, head_dim]`，cache 改沿 `dim=2` 拼接；
- QK score 缩放改为 `sqrt(reduced_head_dim)`；
- Value 聚合仍按真实 `value_head_dim` 计成本；
- padding key 通过 `valid_mask` / `cache_valid` 屏蔽；
- 统计 cache 元素数与 FP32 字节数。

`M41_KV_CACHE_AUDIT.json` 对 `(B,T)=(1,16),(2,17),(3,32)` 实测：

| B | T | 增量最大误差 | 分块 Prefill 最大误差 | 每层 cache 字节 |
|---|---:|---:|---:|---:|
| 1 | 16 | 8.34e-07 | 7.15e-07 | 5120 |
| 2 | 17 | 8.34e-07 | 9.54e-07 | 10880 |
| 3 | 32 | 8.94e-07 | 8.34e-07 | 30720 |

cache 形状均为 `ck=[B,4,T,4]`、`cv=[B,4,T,16]`，没有把序列维误当头维。

### 2.2 M4-B 选择性交互未来信息泄漏

重写 `models/selective_interaction.py`：

- 只有 `group_end < query_pos` 的已完成历史组可用完整摘要；
- 当前未完成组只用 `pos <= query_pos` 的已到达 token；
- 未来组完全不可见；
- padding 不进入摘要或 softmax；
- 选中位置使用联合 softmax，而不是分组 softmax 后相加；
- 训练与部署都执行硬路径，STE 只改变梯度；
- stats 分离理论选择率、实际矩阵执行率、摘要、粗检索、top-k 比较、gather 元素数与精确注意力成本。

### 2.3 正式模型强制因果验证

新增 `m41_causality_verifier.py`，只测试 R1、R2、M4-A、M4-B 正式模型实例。覆盖长度：

```text
1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 32, 64, 128
```

每个未来位置扰动、完整/逐 token、分块 prefill、prefill 后生成、batch 1/2/4、padding、训练/部署、硬门与 STE 路由梯度均已覆盖。比较前显式检查 shape，禁止广播。

最大逐位置误差：

| 模型 | 最大误差 | 失败断言 |
|---|---:|---:|
| R1_gru_reference | 8.94e-08 | 0 |
| R2_tiny_transformer | 7.15e-07 | 0 |
| M4-A_low_rank_interaction | 1.19e-06 | 0 |
| M4-B_selective_interaction | 9.54e-07 | 0 |

### 2.4 逐架构计算成本

重建 `compute_accounting_m4.py`：

- GRU、标准注意力、低秩 QK、选择性真实路径、输出 Head、优化器分别建模；
- 标准注意力包含真实序列平方项；
- 低秩 score 用 `reduced_head_dim`，Value 聚合用 `value_head_dim`；
- 训练成本按 `x[:, :-1]` 实际输入长度；
- 反向传播标记为 `ESTIMATED`，中心倍率 2.0，范围 1.5x–2.5x；
- 优化器按 `8 × params` 估算；
- 小张量手工核算和 Profiler 交叉检查均通过；
- 能耗仍为 `NOT_AVAILABLE`。

### 2.5 唯一训练入口与共享预算

- `positive_control.py` 已移除独立 optimizer 与 `--authorize`，不再能训练；
- `math_lab/exact_computation_reduction.py` 的两处 `optimizer.step()` 改为显式 SGD 公式数值比较，避免绕过预算；
- 静态 AST 审计确认：全目录只有 `train_m4.py` 一处 `opt.step()`；
- `BudgetManager` 新增 `reserve_step()`、`confirm_step()`、`mark_reserve_uncertain()`；
- 每次 step 先落盘 reserve，再执行 optimizer.step，成功后 confirm；
- 未确认 reserve 跨重启仍计入；
- 正控制 256、主训练 3840、总计 4096、墙钟 3600s 的硬上限均通过临时账本实测；
- 真实账本仍是 0 step，dry-run 实测 `optimizer_steps_executed=0`。

### 2.6 严格 ISO 指标

修改 `evaluate_m4.py`：

- 严格正确必须满足原始生成 token 序列等于 `tok.encode(answer)+[EOS]`；
- 必须自然 EOS；
- PAD/SEP/BOS/UNK 不能被 decode 过滤后伪装为正确；
- `iso_record()` 必须连续两个独立验证点达标；
- 成本只取第二个确认点；
- 关键任务下限从冻结预登记方案读取；
- 单 seed 独立判断；
- 软门、缺失 strict_macro、缺失关键指标都阻断达标。

`M41_METRIC_AUDIT.json` 用预设 logits 桩模型实测上述逻辑，10 项检查全部通过。

## 3. 数据独立审计与最小修复

`M41_DATA_AUDIT_BEFORE_REPAIR.json` 保留了首次审计失败证据。发现一个真实生成缺陷：

- 原 `same_query_fact_change` 使用 `(old + 1 + i) % 10`；
- 当 `i % 10 == 9` 时新值等于旧值；
- 10 个配对实际上没有发生目标事实变化。

修复范围仅 `data/m4_counterfactual.jsonl` 的这 10 条：

- 原 SHA256：`84b902ca4833748091a582a211c0b0a82a92647174fcc9534eed4add5627d251`
- 修复后 SHA256：`3bb867d55e02461fe68ec2824552175e395ca429a40ad907456365ed87962453`
- 明细：`M41_DATA_REPAIR_AUDIT.json`

未修改 `train/val/sealed/distance/generalization`。修复后独立审计通过：

- train=1392、val=464、sealed=464；
- canonical 跨集合重合为 0；
- 400 个反事实唯一 pair_id，四类各 100；
- 每类查询位置 first/middle/last 近似均衡；
- 距离测试 60 个 pair_id × 7 个距离；
- train/val/sealed 均未超过 max_len=96；
- 无相同 prompt 对应不同答案，无跨主分割任务语义泄漏。

捷径基线（val+sealed）也如实记录：

| 捷径 | 准确率 |
|---|---:|
| first_digit | 0.2646 |
| last_digit | 0.3385 |
| fixed_position_0 | 0.3455 |
| fixed_position_1 | 0.2257 |
| fixed_position_2 | 0.1914 |
| nearest_assignment | 0.9878 |
| constant_most_common | 0.0647 |
| random_digit | 0.0506 |

`nearest_assignment` 的高分说明绑定类任务存在强规则捷径。它只作为评估基线，不进入模型推理；正式能力结论仍必须使用真实 neural token 生成与严格 EOS 指标。

## 4. 额外工程结论

`M41_EXACT_REDUCTION_AUDIT.json` 重新验证了 Answer-only Head 优化：

- logits 最大差异 4.77e-07；
- 梯度最大差异 3e-08；
- 实测时间变化 `-0.3376`，即变慢约 33.8%；
- 结论：`REJECT_NO_MEASURED_BENEFIT`。

不能把数值等价宣传为 CPU 工程收益。

## 5. 强制准入结果

`M41_PREFLIGHT_REPORT.json`：

```text
CAUSALITY_PASS                 true
INCREMENTAL_PARITY_PASS        true
KV_CACHE_PASS                  true
PADDING_ISOLATION_PASS         true
ROUTING_GRADIENT_PASS          true
ACTUAL_COST_ACCOUNTING_PASS    true
STRICT_EOS_METRIC_PASS         true
DATA_LEAKAGE_AUDIT_PASS        true
SHARED_BUDGET_PASS             true
POSITIVE_CONTROL_GATE_READY    true
```

`POSITIVE_CONTROL_GATE_READY` 表示门禁与唯一入口已就绪，不表示正控制已经训练通过。授权后仍必须先完成 256 步以内正控制，实测 TF≥95%、自然 EOS 完整拟合≥90%、梯度有限非零、参数真实更新、无未来泄漏；参考模型不过门禁就必须暂停正式训练。

## 6. 对核心问题的当前回答

问题：

> 新数学机制是否在保持参考模型能力的前提下，真正减少了整个神经网络的计算成本？

当前答案：**尚不能判定**。

原因是所有正式训练、Best/Final 封存评测、严格 ISO 达标点与真实 Pareto 都未执行。当前只有模型因果性、成本函数、数据、预算和指标链通过预检。没有能力证据就不计算效率收益，不授予 E1–E5。

获得明确授权后的顺序：

1. 256 步以内正控制；
2. 正控制通过后，四架构双种子正式训练；
3. 全部种子 Best/Final 独立封存评测；
4. 同等能力成本、CPU 推理效率与 Pareto；
5. 完整报告、本地权重、非敏感代码与证据提交。

授权前保持：

```text
optimizer.step = 0
training_status = AWAITING_AUTHORIZATION
```
