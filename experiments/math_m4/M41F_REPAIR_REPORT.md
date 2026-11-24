# N-MIND M4.1F｜最终数据修复与正式训练执行准备报告

## 0. 当前状态

- 基于提交：`a55c42a`
- 本轮性质：**零 optimizer.step 修复与预检**
- `optimizer_step = 0`
- `training_status = AWAITING_AUTHORIZATION`
- 最终预检：`M41F_PREFLIGHT_REPORT.json`
- 预检状态：`PREFLIGHT_PASS_AWAITING_AUTHORIZATION`（10/10）
- 未增加架构、未启动 M5、未扩大 4096 步预算。

本轮没有执行正式训练，因此第 7 节六个训练结果问题仍不能给出能力结论；不以参数量、局部 MAC 或数学测试代替训练结果。

## 1. 原始数据快照与重新冻结

- 原始数据快照：`data_snapshot_a55c42a/`
- 快照清单：`M41F_DATA_SNAPSHOT.json`
- 重新冻结清单：`M41F_REFREEZE_MANIFEST.json`
- 历史 `M4_DATA_AUDIT.json`、`M41_*` 审计与报告未覆盖。
- 新数据审计：`M41F_DATA_AUDIT.json`、`M41F_DATA_AUDIT_FULL.json`

重新冻结后 train/val/sealed 仍为 1392/464/464，canonical 跨集合重合为 0。

## 2. 任务语义修复

### 2.1 C_indirect

改为真实变量引用与值复制，源变量后续被覆盖。样例：

```text
i=4
e=i
g=6
a=6
i=8
GET e
```

答案 `4`：`e=i` 在赋值时复制当时的 `4`，之后 `i=8` 不改变 `e`。

### 2.2 C_history_current

改为真实历史值与当前值查询：

```text
g=6
d=5
e=4
h=1
g=0
GET g~OLD
```

答案 `6`；`GET g~NEW` 则返回 `0`。查询模式进入 canonical，不会与当前值查询碰撞。

### 2.3 C_reassign

目标变量重赋值后追加 2 条随机无关赋值，最后一个数字不再等于答案：

```text
j=4
i=6
p=7
c=1
j=9
n=4
l=7
GET j
```

答案 `9`，末位数字是 `7`，破坏“最后一个数字”捷径。

### 2.4 B_rename

使用一对一变量重命名：从原始变量名之外的池中 `sample` 4 个唯一新名，原变量与新变量构成一一映射；禁止两个原变量映射到同一个新名。

### 2.5 独立参考解释器

新增 `reference_interpreter.py`：

- 普通赋值写入当前值；
- `b=a` 是值复制，不建立后续引用依赖；
- `GET v~OLD` 返回历史首个值，`GET v~NEW` 返回当前值；
- 只用于数据生成、数据审计与评测，不进入模型 forward / 训练推理路径。

生成器在 `row()` 与 `take()` 中对每个标签调用参考解释器核验；不匹配立即抛错。

## 3. B_rename 原始错误排查

对 `data_snapshot_a55c42a` 独立扫描：

- 一对一违规：**76 条**
- 标签错误：**18 条**
- 用户提到的 7 条已单独列入 `M41F_BASELINE_AUDIT.json` 的 `seven_requested_cases`。

主要错误原因：

1. 目标变量被重复赋值，旧生成器保存重命名前的旧标签，参考解释器按题面最后一次赋值取新值；
2. 两个原变量被映射到同名新变量，语义族被折叠，canonical 丢失其中一个赋值；
3. 查询变量本身被重复赋值时，旧答案与当前题面不一致。

独立扫描实际发现 18 条而非 7 条；以实际扫描结果为准，不把 7 条当作完整上限。

## 4. 基线重分类

### A. 无语义位置捷径

| 捷径 | val+sealed 准确率 |
|---|---:|
| first_digit | 0.2892 |
| last_digit | 0.2166 |
| fixed_position_0 | 0.3785 |
| fixed_position_1 | 0.2448 |
| fixed_position_2 | 0.1953 |
| constant_most_common | 0.0744 |
| random_digit | 0.0636 |

### B. 语义规则参考

| 规则 | val+sealed 准确率 |
|---|---:|
| query_specified_variable | 1.0000 |
| nearest_assignment | 0.2830 |
| indirect_reference_interpreter | 1.0000 |
| history_state_interpreter | 1.0000 |

`nearest_assignment` 原 M41 的 0.9878 不再描述为“位置捷径”，它是语义规则参考；新数据中因 C_reassign 追加无关赋值与 C_indirect/C_history 语义变化，降至 0.2830。所有规则只用于评估，不进入模型推理。

## 5. 验证/封存隔离

`M41F_EVAL_SPLIT_AUDIT.json`：

- 反事实验证集：320 对
- 反事实封存集：80 对
- 距离验证集：371 行
- 距离封存集：49 行

按 `sha256(family) % 5` 划分；CF 的 family 为 `pair_id`，距离的 family 为目标事实 `GET var|answer`。同一事实族及全部距离/配对变体只进入一个集合；跨集合 pair_id 与 canonical 重合均为 0。

训练期 `train_m4.py` 只加载 `m4_counterfactual_val.jsonl` 与 `m4_distance_val.jsonl`；封存集留到训练与选模完成后。

## 6. 选择性交互 Gather 内存与成本来源

`M41F_GATHER_AUDIT.json`：

- 旧实现中间元素：139,264
- 新直接索引元素：52,224
- 节省元素：87,040
- Profiler 分配：765,952 B/次 → 210,048 B/次，节省 555,904 B/次
- 新旧 Gather 输出 logits 最大差异：**0.0**
- 真实矩阵次数：新旧相同（107,520）
- Prefill T=64 中位数：4.5671 ms → 4.1959 ms（-8.13%）
- Prefill+16 增量生成：43.7304 ms → 44.5011 ms（+1.76%）

真实计算对比：

- T=17、B=2：选择性实际执行 107,520 MAC，dense attention 147,968 MAC，比例 0.7266
- T=64、B=1：选择性实际执行 303,104 MAC，dense attention 1,048,576 MAC，比例 0.2891

结论：直接索引显著降低 Gather 临时内存并改善 Prefill 延迟，但增量生成未稳定变快；内存访问仍有成本。是否带来整体效率收益，必须等训练后按真实能力与完整模型成本判断。

`M41F_COST_SOURCE_AUDIT.json` 证明训练成本统计复用真实训练前向的 `state["stats"]`，`batch_cost` 不再额外执行一次模型前向；审计中第二次 forward 调用数为 0。

## 7. 唯一入口守卫与状态分类

`preflight_guard.py` 在任何真实梯度更新前校验：

- 最新 `M41F_PREFLIGHT_REPORT.json` 状态；
- 当前代码哈希；
- 当前数据哈希；
- 预登记方案哈希；
- 4096 step / 3600s 预算；
- 正控制双参考模型完整记录。

`M41F_GUARD_AUDIT.json` 通过：

- 当前预检有效；
- 过期代码哈希被拒绝；
- 过期数据哈希被拒绝；
- 缺少 R2 正控制记录被拒绝；
- 正控制报告部分完成被拒绝；
- 未通过正控制时主训练门禁拒绝；
- 状态分类覆盖 `COMPLETED`、`PARTIALLY_COMPLETED`、`POSITIVE_CONTROL_FAILED`、`OVER_BUDGET`、`ABORTED_EXCEPTION`。

预检代码哈希：

```text
c09f79b9045eaff957ac95520ca9a8159024cdeb72b1e4238632f68615a0a823
```

数据哈希：

```text
7107aab75e1e72ef2236e8e767e818ab40294e4bb69d3d2e7d9fbe6d5df7206b
```

## 8. 最终预检

`M41F_PREFLIGHT_REPORT.json`：

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

历史 `M4_MATH_TESTS.json` 的 24/24 SHA256 仍为：

```text
ff59cdbb68b8530a85bae4e0b9e184eedc694164a15a8a43cae6cbd87fe5e39a
```

## 9. 正式训练结果问题

当前全部不能回答，原因是没有执行正式训练：

1. 低秩信息交互能否达到参考 Transformer 任务能力：`NOT_RUN`
2. 选择性交互能否达到参考 Transformer 任务能力：`NOT_RUN`
3. 哪种机制需要更少训练计算：`NOT_RUN`
4. 哪种机制需要更少推理计算：`NOT_RUN`
5. 哪种机制在普通 CPU 上实际更快：`NOT_RUN`（仅完成局部 Gather 延迟测试）
6. 是否取得 E1 或更高效率收益：`NOT_RUN` / 不授予

研究假设当前状态：**未获支持，也未被否定**。

## 10. 授权后执行

唯一入口：

```bash
python train_m4.py --authorize
```

预算：

- R1 正控制最多 128 步
- R2 正控制最多 128 步
- 四架构双种子主实验最多 3840 步
- 累计最多 4096 步
- 墙钟最多 3600 秒

正控制不达标立即停止主实验，不扩大预算，不降低标准。正控制失败时输出训练曲线、TF 准确率、逐位置准确率、自然 EOS、梯度与实际生成结果。
