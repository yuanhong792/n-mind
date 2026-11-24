# 阶段 0：现状核查报告

> 只读阶段。未修改任何模型、训练、评测或数据处理代码；本文件是唯一新增交付物。
> 核查时间：2026-11-24；仓库：`D:\cpu本地模型开发\n-mind`。

## 1. 环境与执行口径

- Python `3.12.10`，torch `2.10.0+cpu`，CUDA 不可用，Windows 11。
- pytest `9.2.0`。
- 实际完整测试：`278 tests collected`；本次运行 **276 passed / 2 failed**，上一轮同命令为 **277 passed / 1 failed**。
- 当前工作树未提交文件：`experiments/math_m2/M2_REPORT_LOG.txt`、`experiments/math_m2/M2_TRAIN_LOG.txt`（与本阶段无关，未改动）。

## 2. 代码地图（只列实际存在的核心路径）

| 层 | 文件 | 作用 | 关键位置 |
|---|---|---|---|
| 核心 | `src/nmind/core/architecture.py` | `NModel.step`：encode → workspace read → router → NCell → workspace write → predictor → head | `:171-231`（read/write）、`:123`（`block_to_ws`）、`:255`（head） |
| 核心 | `src/nmind/core/ncell.py` | GRU 风格门控块；稠密 `forward` 与稀疏 `forward_sparse` | `:83-118`（稠密）、`:122-169`（稀疏） |
| 核心 | `src/nmind/core/router.py` | Top-K 路由 + `load_bias` 负载均衡缓冲 | `:29`（类）、`:73-84`（route/topk）、`:116-129`（load_bias） |
| 核心 | `src/nmind/core/workspace.py` | 8 槽固定工作空间；query 注意力读；效用竞争写；溢出记录 | `:42-68`（参数/槽键）、`:79-88`（read）、`:96-193`（write） |
| 核心 | `src/nmind/core/encoder.py` / `predictor.py` / `decision_head.py` | 事件编码、下一状态预测、动作头 | 由 `architecture.py` 调用 |
| 学习 | `src/nmind/learning/trainer.py` | `train()`、`train_continual()`、`evaluate()`、post-hoc probe | `:238`、`:453`、`:154`、`:209` |
| 学习 | `src/nmind/learning/losses.py` / `datasets.py` / `checkpoint.py` | 5+1 联合损失、网格数据/oracle、checkpoint | `docs/architecture.md:46-53` |
| 评测 | `src/nmind/evaluation/tasks.py` | 任务套件、记忆跨度、预算-难度、决策指标 | `:26`、`:66`、`:128`、`:179` |
| 评测 | `src/nmind/evaluation/baselines.py` | dense N-Cell / GRU / shared-trunk 基线 | `:1-14`（明说 Transformer 未实现）、`:37`、`:57`、`:106` |
| 评测 | `src/nmind/evaluation/benchmarks.py` | CPU 延迟/RSS 测量 | `:65-96`（measure）、`:146-149`（默认 `seq_len=48`） |
| 语言 | `src/nmind/language/neural_dialogue.py` | `NMindLM`：NCell + Router + Workspace 自回归 | `:55`（类）、`:84-105`（forward）、`:128`（n_params） |
| 语言 | `src/nmind/language/tokenizer.py` | 字符级 tokenizer；UNK 不静默丢弃 | `:18-20`（特殊 token）、`:54-65`（建词表）、`:71-82`（encode） |
| 语言 | `src/nmind/language/train_language.py` / `evaluate_language.py` | 200 步上限训练；自由生成为主指标、TF 为诊断 | `train_language.py:32`、`evaluate_language.py:1-6,307,350` |
| 语言 | `src/nmind/language/decoder.py` | v1 anchor 二次前向；v2 直续 | `:96-109`（v1）、`:126-140`（v2） |
| 语言 | `src/nmind/language/lm_update.py` | 有界流更新、AdamW、adapter 训练范围 | `:118`（默认 lr=3e-3）、`:159`（AdamW）、`:78-107`（trainable） |
| Bot | `src/nmind/bot/learning_loop.py` | 四臂候选更新、预算、replay、curriculum | `:122-135`（默认 lr=3e-3/48 步/replay 0.15）、`:224`（AdaptationBudget） |
| Bot | `src/nmind/bot/store.py` / `client.py` | SQLite 消息队列与客户端 | `store.py:83`（共享连接，`check_same_thread=False`）、`:271-284`（enqueue） |
| Bot | `src/nmind/bot/practice.py` / `capability_tracker.py` | 三族练习、独立 oracle、能力台账 | 由 `bot/runtime.py` 调用 |
| 其他 | `src/nmind/memory/`、`online/`、`bridge/`、`planning/`、`v3/`、`skills/` | 记忆、联网经验流、云端策略、规划、V3 接口、技能工具箱 | 本阶段只做目录级核对 |

规模：`src/nmind` 142 个 `.py`；`tests` 47 个 `test_*.py`；`experiments` 26 个一级目录、364 个 JSON、73 个 JSONL。

## 3. 测试结果

命令：

```bash
python -m pytest tests -q
```

结果：

- 收集：**278** 项。
- 最近一次：**276 passed / 2 failed**。
- 前一次同命令：**277 passed / 1 failed**。
- 失败项：
  1. `tests/test_bot_v1_chat.py::test_concurrent_submit_no_loss`（稳定复现 3/3 次失败）
  2. `tests/test_bot_v1_web.py::test_chat_message_full_protocol`（全量运行时失败 1 次；单独运行通过，表现为并发干扰下的 flaky）

`test_concurrent_submit_no_loss` 的代码路径：

- 20 个线程并发调用 `BotClient.submit`：`tests/test_bot_v1_chat.py:63-79`
- `BotClient.submit` 直接调用 `BotStore.enqueue_message`：`src/nmind/bot/client.py:36`
- `BotStore` 使用单个共享 `sqlite3.Connection(..., check_same_thread=False)`：`src/nmind/bot/store.py:83`
- `enqueue_message` 中 `SELECT MAX(seq)` 与 `INSERT`/`UPDATE` 未用同一把 Python 锁串行化：`src/nmind/bot/store.py:271-284`
- 实际异常：`sqlite3.InterfaceError: bad parameter or other API misuse` / `not an error`；最终 20 条只入库 10 条。

结论：当前测试套件**不是全绿**；失败集中在并发 SQLite 访问，不是模型数学测试失败。

## 4. README 数字与 JSON 核对

### 4.1 能对上的数字

| README 声明 | 证据 | 核对 |
|---|---|---|
| M1 测试遮挡段 `0.941` | `experiments/2026-11-21_m1_overfit.json` A 组 `test.accuracy_occluded=0.940625` | 一致 |
| M2 `22/22` checkpoint 重载一致 | `experiments/m2_closeout/registry.json` `n_checkpoints=22, n_verified=22, n_failed=0` | 一致 |
| M3 `279,220` 参数、holdout `32.7%`、dev `24.6%` | `experiments/m3_language/m3_v1_results.json` `params.total=279220`、`holdout.span_exact=0.3267`、`dev.span_exact=0.2463` | 一致 |
| M5 `452k` 参数、200 步、TF `74.41%`、free-gen `3.61%` | `experiments/m5_language/train_language.json` `params_total=452116`、`optimizer_steps_executed=200`；`eval_language.json` `char_accuracy_teacher_forced.sealed_test=0.7441`、`splits.sealed_test.free_generation_exact_match=0.0361` | 一致 |
| CPU T=24 1 线程 `38.659/25.626 ms`、p50 约 `1.51×` | `experiments/m2_closeout/cpu_benchmark.json` `A_dense_ce.threads_1.p50_ms=38.659`、`C_sparse_ce.threads_1.p50_ms=25.626` | 一致 |

### 4.2 README/文档不一致

1. **测试数**：README `:89` 写“截至该次报告 49 项通过”，`:137` 写“92 项通过”，`:148` 写“108 项通过”；STATUS `:182` 写“131 项全绿”；实际收集 **278 项**，最近运行 **276 passed / 2 failed**。这些历史数字没有统一更新，读者无法从 README 判断当前测试状态。
2. **CPU 基准口径**：README `:110-117` 使用 T=24、1 线程、`38.659/25.626 ms`、`1.51×`；`experiments/README.md:26` 仍写 T=48、`35.4/21.1 ms`、`1.68×`；STATUS `:248` 明确说旧 1.68× 已撤回。README 主表与 experiments 索引不一致。
3. **环境**：AGENTS `:19-20` 写 torch `2.10.0+cu128`、测试 72 项；STATUS `:7` 也写 cu128，但 STATUS `:122` 与 ROADMAP `:78` 都提醒实测为 `2.10.0+cpu`。本机实测确为 `2.10.0+cpu`、CUDA 不可用。

## 5. 假设 A–G 核对表

| 假设 | 判定 | 代码/证据 |
|---|---|---|
| **A. 神经基线为 0、EM 地板效应、封存集小** | **部分成立** | R1 三族神经台账均为 `TESTED_FAIL`，`sealed_em=0.0`：`experiments/bot_runtime_r1/capability_matrix.json`；Gate-B 001/002/003 的 3/4 号位置各 `n=20, em=0.0`：`experiments/bot_runtime_r2/gateb_result_00{1,2,3}.json`。EM 是否“太粗看不到进展”无法从现有证据判定，因为仓库同时有 TF 字符准确率、free-gen EM 等不同指标（`src/nmind/language/evaluate_language.py:1-6,307,350`），但没有把它们与 EM 做同一 checkpoint 的相关性分析。封存集规模确实偏小（Gate-B 每位置 20 条、M5.2 sealed 80 条），方差风险成立。 |
| **B. M5/M3 约 28–45 万参数、200–1000 步，可能严重欠训练** | **部分成立** | 参数量：M3 `279,220`（`experiments/m3_language/m3_v1_results.json`）；M5 `452,116`（`train_language.json`）。步数：M5 200（`train_language.py:32`、`train_language.json`）；M5.1 stage0/1 各 200（`run_m51_stages.py:12-14,48`）；M5.2 S0/S1 各 600（`run_m52.py:52`）。M5.1 诊断给出“欠训练”证据：train TF-span-EM 12.93% ≈ sealed 12.55%（`diagnose_m5.json`），但 M5.2 S0 600 步已达到 sealed free-gen EM 1.0（`m52_eval.json`），说明同一规模并非一律欠训练；“严重欠训练”只在特定任务/阶段成立。 |
| **C. 两实体绑定 0.15 低于随机线；Workspace 固定槽位缺按内容寻址 KV 读写** | **部分成立** | 0.15 是 M5.1 stage1：`m51_stages.json` stage1 eval EM=0.15、门槛/随机线 0.25（`run_m51_stages.py:12-14` 与报告）。M5.2 S1 sealed EM 0.8375，但位置交换跟随率仅 0.0833（`m52_follow_eval.json`），更精确地说明“近因/位置捷径”而非单纯容量不足。Workspace 读是 query 对槽内容的注意力（`core/workspace.py:79-88`），写是 payload 对固定 learned `slot_keys` 的 softmax 分配（`:120-140`）；它有“按内容读”的成分，但没有把 payload 自身投影成显式 key、再按 key 检索的 KV 接口。因此“完全缺少内容寻址”过强，判为部分成立，需阶段 5 的 MQAR/bAbI 对照验证。 |
| **D. Gate-B 001–003 的 lr/optimizer/回放未单独消融** | **部分成立** | 001 预登记：96 步、lr=3e-3、replay=0.15（`gateb_pre_registration.json` budget）；002 只改 trainable（`gateb_pre_registration_002.json` budget）；003 只改 adapter_prefixes（`gateb_pre_registration_003.json` budget）；005C 只改课程呈现（`STATUS.md:175-183`）。仓库里没有 lr 扫描、Adam/SGD 对比或 replay 0.15/0.3/0.5 的已执行结果。`masked_stream_update` 固定用 AdamW（`src/nmind/language/lm_update.py:159`）。但“新建 Adam 早期近似符号更新破坏读写接口”是机制假设，当前证据不能判定。 |
| **E. CPU 基准主要测 PyTorch 调度开销，而非稀疏收益** | **部分成立 / 因果无法判断** | 实测延迟与模型规模：T=24、1 线程 p50 25.6–38.7 ms（`experiments/m2_closeout/cpu_benchmark.json`），模型约 282k 参数（`docs/architecture.md:70`、`results/baselines.json`）。`benchmarks.py` 只测 wall-clock/RSS（`:65-96`），没有 profiler 区分算子执行与调度，也没有记录估算 FLOPs；“约 14 MFLOP”在仓库中没有对应 JSON。高延迟事实成立，但“主要测到调度开销”缺少直接测量证据。 |
| **F. 稠密/GRU/Transformer 公平基线缺失** | **成立（但需细分）** | `evaluation/baselines.py:1-14` 明说 Transformer 基线未实现；实际只有 dense N-Cell、GRU、shared-trunk（`:37`、`:57`、`:106`）。dense/GRU 已存在，但不是同损失、同预算的完整公平对照：baselines 的 `fit` 只用 CE，而 N-MIND 主训练是 5+1 项联合损失（`README.md:301`、`experiments/README.md:20`）。多 seed、同参数量 ±5% 的 Transformer/线性注意力基线仍未完成。 |
| **G. 词表 31 token 导致 OOV；TF 与自由生成差距大** | **部分成立** | `tokenizer_m52.json` 词表为 31；`baseline_capability_matrix.json` 记录 arithmetic 题面 OOV 率 0.5131、binding 词表 31 且 `unseen_position_values` 等族 OOV。TF 74.41% vs free-gen 3.61%：`eval_language.json`。但“词表只有 31”不适用于 M5 v1：`tokenizer.json` 是 123 词表；31 是 M5.2/bot 绑定域词表。`tokenizer.py:71-82` 对未见 token 返回 UNK，decode 时显示替换符（`:84-94`），不会静默丢弃。 |

## 6. 与阶段 0 目标相关的其他发现

- 测试失败不是模型数学失败：`tests/test_ncell.py`、`test_router.py`、`test_workspace.py` 等核心测试在本轮均通过；失败集中在 `bot` 的并发 SQLite 路径。
- 历史结果与代码存在“文档滞后”：README/STATUS/ROADMAP 记录了不同时间点的测试数、环境（cu128 vs cpu）和 CPU 口径；阶段 6 的文档整理应优先处理，但阶段 0 不改。
- `experiments/m2_closeout/cpu_benchmark.json` 的 `env.torch` 写 `2.10.0+cu128`，但 `results/benchmark_cpu.json` 的 `hardware.torch` 写 `2.10.0+cpu`；这是历史运行环境差异，不是同一份 JSON 的内部矛盾，但跨文件比较时必须注明。
- README `:314` 明确说根目录缺少独立 `LICENSE`，`pyproject.toml` 声明 MIT；阶段 6 再补，不在阶段 0 处理。

## 7. 需要批准的事项

1. **是否允许在阶段 1 之前修复并发测试基础设施**：`BotStore` 共享 SQLite 连接需要串行化/锁或每线程连接；这是非模型行为修复，但会改代码。建议先修，否则阶段 1 的“测试必须全绿”无法满足。
2. 阶段 1 的评测基础设施本身不需要训练预算；但“用历史候选 checkpoint 重评”只读，不新增训练。
3. 阶段 2/4 的任何训练仍需预注册和单独预算批准；本阶段没有运行任何训练。

## 8. 已知局限

- 本报告没有逐行阅读 142 个源文件和 364 个实验 JSON；代码地图按核心调用链和目录职责整理，未覆盖每个 V3/planning 辅助模块。
- 测试失败只做了复现与代码路径定位，没有在本阶段修复；这是阶段 0 的只读边界。
- A–G 的判定基于现有仓库证据；“EM 地板效应”“Adam 早期符号更新”等机制性表述需要阶段 1/4 的新实验才能定论。

## 9. 下一阶段建议

1. 先修 `tests/test_bot_v1_chat.py::test_concurrent_submit_no_loss` 的 SQLite 并发问题，并复查 web 协议 flaky；让阶段 1 从全绿基线开始。
2. 阶段 1 先做连续指标与统计工具，再用“标签置换负对照 vs 真实训练”验证新指标能在 EM=0 时区分好坏。
3. 阶段 2 再实现同参数量 Transformer/线性注意力基线；在预算批准前只写预注册与命令，不运行训练。
