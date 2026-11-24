# 阶段 2.0：任务有效性与预注册修订

日期：2026-11-24。未运行正式训练；只运行批准的只读重评与 timing-only 20 步计时。stable、历史结果、历史封存集未改写。

## A. S1 重叠审计

S1 训练与 sealed 各 12 个唯一输入，逐输入指纹重叠为 **12**；模板抽象空间重叠为 **1**。S1 与 S0 一样是很小的输入空间。因此历史 S1 sealed EM=0.8375 **可能来自记忆**，不能作为绑定能力证据。

统一指纹清单已导出到 `experiments/phase2_baselines/fingerprints_20261124/MASTER.json`，当前收录 M5.2 S0/S1 train/sealed 和新 position archive，共 528 个唯一指纹；新 archive 自身与已纳入清单重叠为 0。Gate-B 历史文件只提供聚合 `data_fingerprints`，没有逐样本全集，因此仍不能宣称对所有历史 Gate-B 数据完成零重叠审计。

## B. 完整 S1 与 200 对探针只读重评

结果目录：`experiments/phase1_reeval/20261124/`。`stable_s1.json`、`gateb001.json`、`gateb002.json`、`gateb003.json` 均包含 S1 sealed 80 条和探针 200 对（400 条 A/B 样本），保存逐样本 token log-prob、答案 LL、NLL、候选排名、free-generation EM；四份结果 UNK 丢弃数均为 0。

| checkpoint | S1 EM | S1 NLL | S1 平均排名 | 探针跟随率 | 探针 Wilson 95% CI |
|---|---:|---:|---:|---:|---|
| stable_s1 | 0.8375 | 0.1834 | 1.425 | 0.130 (26/200) | [0.090, 0.184] |
| Gate-B 001 | 0.4250 | 0.7028 | 3.562 | 0.125 (25/200) | [0.086, 0.178] |
| Gate-B 002 | 0.0875 | 0.7244 | 3.987 | 0.000 | [0.000, 0.019] |
| Gate-B 003 | 0.0875 | 1.2548 | 4.250 | 0.000 | [0.000, 0.019] |

探针随机 pair-level 水平为 1/16=0.0625。stable 的跟随率虽高于随机点估计，但区间仍宽；001 与 stable 的区间高度重叠，不能称为差异。002/003 低于随机点估计，但不能据此宣称稳定的负向能力。

有逐样本 S1 EM 数据，因此做了配对符号置换检验（10000 次、seed=7）：001−stable=-33/80、002−stable=-60/80、003−stable=-60/80，双侧 p 均约 0.0001。该检验只说明这些候选在本次重评上显著低于 stable，不等于证明 stable 学会了绑定；S1 的组合泄漏仍然成立。

历史 S1 基线（n=80）为：checkpoint 0.8375，最近实体 0.500，多数答案 0.175，随机 0.213，固定位置最高 0.175。checkpoint 高于这些基线，但因 12 个输入组合完全重叠，结论仍是记忆/复现，不是泛化。完整 200 对探针更新后的 stable 跟随率为 0.130（26/200，Wilson [0.090,0.184]）；旧的 0.25 仅是 n=24 的小样本估计。

## C. 任务有效性门槛

门槛代码：`src/nmind/evaluation/phase2_validity.py`；每个任务结果在 `experiments/phase2_baselines/task_validity_20261124.json`。

| 任务 | 结果 | 失败项 |
|---|---|---|
| M5.2 S0 | FAIL | 最近实体捷径、有效样本<500、组合重叠、无反事实对 |
| M5.2 S1 | FAIL | 最近实体捷径、有效样本<500、组合重叠 |
| M1 grid | INSUFFICIENT_N | 缺统一组合级指纹/词表/反事实报告 |
| 新大组合绑定 | FAIL | 最近实体捷径超过随机+0.05 |

没有调整门槛来让任务通过。根据规则，阶段 2 当前 `tasks=[]`，上述任务全部移入 `smoke_only`。S0 明确降级为冒烟测试，不参与比较。

## D. 预注册修订

`experiments/phase2_baselines/preregister.json` 已修订：

- 正式比较任务只保留通过有效性门槛者；当前为空。
- 增加因果 Transformer 正控制：新大组合绑定任务需 EM≥0.80 且跟随率≥0.70，否则阶段 2 中止；由于任务当前未通过，正控制不进入正式比较。
- 分层规则固定为 smoke=1 seed、pilot=3 seeds、formal=10 seeds；进入和停止条件写入 JSON。
- 每个臂（含负对照）必须保存 checkpoint、逐样本 logits 与元数据；缺失即运行无效。代码级检查和测试已加入。

## E. 20 步 timing-only

每个模型恰好 20 optimizer steps，未保存 checkpoint、未评 sealed，结果明确标记 `timing_only=true`：

| 模型 | 20 步墙钟 |
|---|---:|
| GRU | 0.150s |
| causal Transformer | 0.2772s |
| delta-rule | 0.1106s |
| N-MIND | 4.361s |

代码没有允许超过 20 步。预注册已写入 smoke/pilot/formal 的步数估算（20/60/200 steps 每模型×任务×seed）和 PENDING_APPROVAL 状态。由于当前没有通过任务，正式总步数暂为 0；不能把 timing-only 墙钟当成训练结论或晋升依据。

## F. README/STATUS 注记

已在 README 与新建 STATUS.md 注明：M5.1 stage0、M5.2 S0/S1 可被最近实体捷径解出或反事实跟随率无法稳定区别随机，不构成绑定能力证据。

## 需要批准

1. 是否批准为 Gate-B 负对照补充可读取 checkpoint，以完成真正的连续指标对照。
2. 是否批准在任务有效性门槛通过后进入 smoke/pilot/formal；当前没有任务获准进入正式比较。

## 已知局限与假设更新

局限：S1 组合空间只有 12 个；Gate-B 完整逐样本历史指纹和负对照 checkpoint 缺失；M1 统一组合级审计尚未具备；timing-only batch 只是成本基准。

假设 A 得到进一步支持：S0/S1 的高 EM 可由捷径或重复组合记忆产生。假设 C 暂不支持神经泛化主张：stable 探针跟随率仅 0.13，虽然高于随机点估计，但组合泄漏与区间宽度使其不能作为学会绑定的证据。
