# 阶段 1.5 + 阶段 2 预注册交付

日期：2026-11-24。没有运行任何训练；只读加载历史 checkpoint，并把新评测写入 `experiments/phase1_reeval/20261124/`。

## 先确认

1. 上一份报告中的 1 个 skipped 来自 GUI 测试的 `tkinter` 显示初始化保护（无可用显示环境时 `pytest.importorskip`/`pytest.skip`）。它是环境性跳过，不是功能通过；本次环境重新运行 `tests/test_bot_v1_gui.py` 为 5 passed。完整套件为 287 passed（本次没有跳过）。
2. `tests/test_bot_v1_web.py` diff 只有：轮询从 50 次扩为 150 次、每次 PENDING 后 sleep 20ms；三个断言完全未改变：必须收到记录、`response_source == tool`、文本含 `56`。

## M5.2 捷径基线

在历史 S0/S1 sealed 构造（各 n=80）上按同一 oracle 计算：

| 集合 | checkpoint EM | 最近实体 | 多数答案 | 随机 | 固定位置(1/2/3/4) |
|---|---:|---:|---:|---:|---|
| S0 | 1.000 | 1.000 | 0.250 | 0.338 | 0.250/0.250/0.250/0.250 |
| S1 | 0.838 | 0.500 | 0.175 | 0.213 | 0.175/0.088/0.163/0.075 |

验收口径已固定：checkpoint 的 EM 必须显著高于所有捷径基线且反事实跟随率达标，才可称为“学会”。S0 训练/ sealed 输入指纹重叠为 16，双方各只有 16 个唯一输入；模板抽象空间重叠为 4，S0 EM=1.0 存在记忆整个模板空间的可能，不能作为泛化证据。

## 跟随率统计

历史配对汇总的成功数为 stable=6/24、001=7/24、002=2/24、003=2/24、S1=2/24。随机两问都正确的 pair-level 水平为 1/16=0.0625。Wilson 95% 区间如下：

| 臂 | 跟随率 | Wilson 95% CI | 当前结论 |
|---|---:|---:|---|
| stable | .250 | [.120,.449] | 高于随机，但 n=24 |
| 001 | .292 | [.149,.492] | 与 stable 差异不显著 |
| 002 | .083 | [.023,.258] | 与 stable 的配对检验缺少逐对历史数据 |
| 003 | .083 | [.023,.258] | 同上 |
| S1 | .083 | [.023,.258] | 复现审计值，未达 .70 |

新组合探针生成器已能确定性生成 200 对（10 实体 × 10 位置组合）；但旧 checkpoint 的历史评测文件没有这些新对的逐样本 logits，故没有伪造“已评估 200 对”的结论。旧 `build_binding_swap_wide` 的唯一上下文空间仍为 96 对，已记录为缺口。

## 封存与零重叠审计

`phase1_archive` 已扩展 `position_qa` 为实体池×位置池×问法×确认语组合，生成 500 条去重有效样本；单元测试通过。完整历史学习流/封存集指纹清单并不存在于仓库：已发现 M5.2 `data/manifest.json`、sealed JSON 与 Gate-B 各自的 `data_fingerprints`，但缺少统一逐样本指纹导出。因此本次不宣称历史全集 overlap=0；缺口已明确保留。

## 只读 logits 重评与验收 B/C

只读结果目录包含 stable_s0、S1、Gate-B 001/002/003，每个 S0/S1 各 8 条样本，保存答案 token log-prob、整体 LL、NLL、候选分数、排名、free-generation EM，并写入严格元数据。由于 CPU 单条递归模型评测成本高，本次有效 n=8，只能作为管线冒烟，不能替代要求的完整 n。

在这 8 条上，连续指标能计算，但 Gate-B 负对照 checkpoint 文件不存在（仓库只有 rejected/001、002、003），且 n=8 不足以做可靠区分；验收 B 结论：**区分不出**。R1 arithmetic/json_extract/grid_rules 只有历史 capability JSON，没有对应可加载 checkpoint/logits；验收 C：**区分不出**，不能声称 NLL/排名优于随机。

结果写入器现在强制要求 torch 版本/build、Python、git revision、数据指纹、配置哈希、seed、样本数和有效样本数；缺字段抛错。新增缺字段拒绝测试已通过。

## 阶段 2：预注册与实现准备

已新增：

- `experiments/phase2_baselines/preregister.json`：四模型、四任务、固定 10 seeds、3 点 dev 学习率网格、公平性和指标优先级；状态为 `PREREGISTERED_NOT_RUN`。
- `models.py`：GRU、因果 Transformer、delta-rule 风格线性递归基线，统一 token 接口。
- `run.py --dry-run`：只打印预注册，不训练；非 dry-run 明确拒绝。
- 参数构造单元测试。

正式 20 步计时没有执行，因为这会进入训练/优化器运行，与本轮“不运行任何新的训练”冲突。正式预算和更小冒烟预算仍为 `PENDING_APPROVAL`，不能给出虚假的墙钟估计。

## 当前需要批准

1. 是否批准只读重评扩大到完整 80/200 对，并为缺失的 Gate-B 负对照 checkpoint 补充可读取工件。
2. 是否批准阶段 2 的 20 步计时与正式训练预算；在批准前不运行 optimizer step。

已知局限：历史 Gate-B 结果只有聚合跟随率，没有逐对标签；旧封存与学习流没有统一指纹清单；R1 checkpoint 工件缺失；当前 logits 重评 n=8 仅为管线验证。

假设更新：A 得到更强支持——S0 EM=1.0 与最近实体基线同为 1.0，且模板空间完全重复；C 仍无神经泛化证据，必须等完整反事实和连续指标重评后再判断。
