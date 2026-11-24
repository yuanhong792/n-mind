# N-MIND 阶段 0.5 + 阶段 1 交付报告

日期：2026-11-24。全程未启动新训练，stable 与历史 JSON 只读。

## 阶段 0.5：BotStore 并发

复现证据保存在 `reports/phase05_repro_before.txt`：原实现的 chat 并发测试出现 SQLite `InterfaceError`/空值，单独 web 测试当次通过。

根因是共享 `check_same_thread=False` 连接缺少进程内串行化，且 `SELECT MAX(seq)+1` 与 INSERT 不在同一事务。采用方案 (a)，没有改表结构：所有共享连接访问通过 `threading.RLock`，入队使用 `BEGIN IMMEDIATE` 包住建会话、取 MAX(seq)+1、插入和提交；新增 `cancel_message` 也走同一锁。旧数据库无需迁移，原表结构保持不变。

新增 20/100 线程序号连续性测试，以及 submit/read/confirm 混合压力测试。客户端返回 `seq`，取消不再直接操作裸连接。

阶段 0.5 提交：`09b57ec fix(bot): serialize shared sqlite chat store`。随后网页测试增加了有等待的 3 秒轮询窗口，修复原测试自身的时序假失败；修复后另有 35 次通过（此前窗口过短的 2 次失败保留在旧日志中）。chat 文件循环日志无失败；完整测试最终 `287 passed, 1 skipped`。

## 阶段 1：评测基础设施

新增模块：

- `src/nmind/evaluation/phase1_metrics.py`：per-token NLL、答案对数似然、候选排名、字符编辑距离，EM 不被替换。
- `phase1_baselines.py`：最近实体、多数答案、随机、固定位置，以及绑定反事实跟随率。
- `phase1_archive.py`：带种子的封存生成、数据指纹和重叠审计。
- `phase1_stats.py`：均值/样本标准差、可复现 bootstrap 95% CI（默认 10000）、配对 bootstrap、置换检验；样本不足返回 `INSUFFICIENT_N`。
- `tests/test_phase1_eval_infra.py`：已知指标、统计可复现性、封存指纹、捷径和探针行为测试。

历史自由生成判定逻辑未改；新模块是旁路能力。结果元数据接口包含 torch/Python/git revision、数据指纹和配置哈希字段，供结果写入器调用。

## 只读验收

**A（旧指标复现）**：`experiments/m5_language/m52_follow_eval.json` 的历史 S1 反事实跟随率为 `0.0833`（24 对、2 对跟随），与审计记录一致；新代码未覆盖该文件。历史 JSON 没有逐 token logits，因此 NLL/候选排名无法从旧结果重算，标记 `INSUFFICIENT_N`，不补造数。

**B（Gate-B 001/002/003）**：历史 paired report 显示三候选与 stable 的负对照 EM 均为 0.0；001/002/003 跟随率分别 0.2917/0.0833/0.0833，stable 0.25。由于历史文件没有 logits，连续指标区分能力为 `INSUFFICIENT_N`；没有把 EM 0.0 误报成区分成功。

**C（R1 三族神经基线）**：历史 capability matrix 的 arithmetic/json_extract/grid_rules 神经 sealed EM 均为 0.0（各 n=8），但未保存 token log-prob，NLL/排名同样为 `INSUFFICIENT_N`。因此不能声称已显示出与随机的差距。

## 假设更新

- 假设 A：EM 可能被“最近实体/位置”捷径抬高；S1 的 0.0833 跟随率和历史 `rule_baseline_follow_rate=1.0` 支持该风险，后续绑定任务应以跟随率为主、EM 为辅。
- 假设 C：当前 R1 神经基线没有可证实的神经泛化证据；EM=0 与连续指标缺失不能推出更强结论。

## 封存集与限制

生成器是确定性的并记录指纹；但 `position_qa` 模板空间只有 16 个不同字符串，要求每族 500 条时允许重复样本，不能把它伪装成 500 个独立泛化点。与真实学习流/历史封存集的最终零重叠审计需使用完整历史指纹清单；当前单元测试验证了审计路径，但不宣称历史全集已零重叠。

需要批准的事项：若要完成 A/B/C 的连续指标验收，需批准对历史 checkpoint 做只读重评并保存逐 token logits/候选分数的新结果文件；仍不涉及训练或 stable 改写。
