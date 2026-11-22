# N-MIND · 脉脑｜Agent Runtime R1 独立 Bot 自主学习 —— 实测报告

生成时间：2026-11-22（本地实测，非计划稿）
任务书：`N-MIND_Agent_Runtime_R1_独立Bot自主学习_Codex开发提示词.md`（§9 验收 + §11 交付物）
分支：`master` @ `f1c7b1c`（M5.2 由并行会话提交，本任务全部为增量新增，未覆盖他人文件）
环境：Windows 10.0.26100 / Python 3.12.10 / torch 2.10.0+cpu / 28 逻辑核 / 无 GPU 依赖

---

## 0. 一句话结论

**P1–P3 的"无人发消息也能自主学习并留下可复现证据"闭环已经真实跑通（11/11 安全探针 PASS、16/16 R1 单元测试 PASS）；神经参数增长没有拿到（候选被判 REJECTED，封存集 EM 0.0 → 0.0）。** 本轮证明的是运行时、取证与防自欺机制可靠，**不是** "Bot 变聪明了"。

---

## 1. 交付物与证据对应

| 任务书要求 | 代码 | 证据文件 | 状态 |
| --- | --- | --- | --- |
| §2 状态机 + 生命周期 | `src/nmind/bot/runtime.py` | `runtime_transitions.jsonl`(20 KB) | 已写 + 已测试 + 已实测 |
| §3 模块与数据契约 | `src/nmind/bot/`（12 个模块） | `tests/test_bot_runtime_r1.py` | 已写 + 已测试 |
| §4 空闲自主循环 | `bot/practice.py` `bot/goal_manager.py` `bot/capability_tracker.py` `bot/experience_bridge.py` | `autonomous_cycles.jsonl`(55 KB / 81 次尝试) | 已写 + 已测试 + 已实测 |
| §5 分层量化（知识/工具/神经） | `capability_tracker` 的 `{family}:neural` / `{family}:tool` 双台账 | `capability_matrix.json` | 已写 + 已实测 |
| §6 Windows CLI 控制面 | `src/nmind/bot/cli.py`（由 `src/nmind/cli.py` 注册，`bot` 子命令组 15 个） | 本报告 §6 + `reproduce.ps1` | 已写 + 已实测 |
| §7 权限与硬预算 | `bot/policy.py` `configs/bot_runtime.yaml` | `security_tests.json`(11/11 PASS) | 已写 + 已测试 + 已实测 |
| §8 stable/candidate 隔离与原子晋升 | `bot/learning_loop.py` `bot/evolution_manager.py` | `lineage_and_buffer_audit.json` | 已写 + 已测试 + 已实测 |
| §9 十五条单元测试 | `tests/test_bot_runtime_r1.py` | pytest（见 §5） | 15 条 + 1 条实验回归，全绿 |
| §11 九件产物 | `experiments/bot_runtime_r1/` | 本目录 9 件 + `summary.json` | 全部真实产出 |

`_work/` 与 `results/bot_r1_e2e/`（Bot 本地 SQLite、权重副本、控制文件）不入库。

---

## 2. 能力状态（严格区分五种措辞）

| 能力 | 已写代码 | 已测试 | 已验证能力 | 研究失败 | 尚未开发 |
| --- | :-: | :-: | :-: | :-: | :-: |
| 独立启动/停止/暂停/恢复（跨进程） | ✅ | ✅ §9.11 | ✅ 实测 rc=0、锁清理 | — | — |
| 无人消息时自主练习（≥3 任务族） | ✅ | ✅ §9.1 | ✅ 20 轮 60 次，全程零 inbox | — | — |
| 用户消息抢占低优先级练习 | ✅ | ✅ §9.2 | ✅ 实测队列 0 丢失、按序回答 | — | — |
| 目标池白名单 / 阈值不可被外部指令改 | ✅ | ✅ §9.5 | ✅ 探针 PASS | — | — |
| 封存集与练习集互斥（题面级） | ✅ | ✅ §9.6 | ✅ 81 指纹 0 重复、0 封存行 | — | — |
| 验证拒绝的经验不得触发神经更新 | ✅ | ✅ §9.7 | ✅ 实测（rejected-only 时 executed=false） | — | — |
| 候选不得自动晋升 / 无 flag 拒绝 | ✅ | ✅ §9.8 | ✅ `approve` 无 `--i-approve` 被拒 | — | — |
| Router buffer 隔离（M4 虚假增益防线） | ✅ | ✅ §9.9 | ✅ `reset_applied=true`，漂移单列 | — | — |
| 断电式重启一致性 | ✅ | ✅ §9.10 | ✅ 23→26 轮，stable sha 不变 | — | — |
| 日志/事件容量上限 | ✅ | ✅ §9.12 | ✅ 轮转 + prune 生效 | — | — |
| 零外部模型依赖 | ✅ | ✅ §9.13 | ✅ socket 计数 0 + 源码扫描 0 处调用点 | — | — |
| 不会答时如实拒答（不拿规则冒充神经） | ✅ | ✅ §9.14 | ✅ 台账 neural 0/81，工具另计 | — | — |
| **神经参数增长（新技能）** | ✅ | ✅ | ❌ | ✅ **REJECTED** | — |
| 旧技能遗忘量化 | ✅ | ✅ | 0.00 pp（无遗忘，但也没有增益） | — | — |
| 只读联网探索（P4） | 部分（`online/fetcher` 复用） | 单测覆盖 offline 硬断 | ❌ 未做真实网络 | — | ✅ 待单独授权 |
| Windows 开机自启 / GUI 面板 / exe 打包（P5） | ❌ | — | — | — | ✅ 未开发 |
| 成熟中文对话 | 复用 M5 原型 | M5.2 测试 | ❌ M5.2 两实体绑定仍 BINDING_FAIL | — | 属 M5 线，不在 R1 范围 |

---

## 3. 自主练习实测（Phase A/C/E 累计，全程无人发消息）

台账累计 **81 次任务实例尝试**（4 个 `run_id`、27 轮自主周期，明细在 `autonomous_cycles.jsonl`；状态转移 126 条、观察到 10 个状态、非法转移 0 条，见 `runtime_transitions.jsonl`）：

| 任务族 | 尝试 | 神经侧 verified | 工具侧 verified | outcome_source | 封存集自由生成 EM（stable） |
| --- | --- | --- | --- | --- | --- |
| arithmetic | 27 | **0** | 27 | tool | 0.000（n=8） |
| json_extract | 27 | **0** | 27 | tool | 0.000（n=8） |
| grid_rules | 27 | **0** | 无工具路径 | none | 0.000（n=8） |
| 合计 | 81 | **0 / 81** | 54 / 81 | — | 0.000 |

要点（这是本轮最重要的诚实结论）：M5.2 stable 权重面对这三类任务模板时，自由生成输出的是与题目无关的句子（例：`请计算：11加18等于多少？` → `小明在2号位置。`），因此**神经基线为 0**；54 次 verified 全部来自确定性工具/oracle，工具贡献单独计量、绝不记入神经分数。`selection_source=rule`（沿用 `pick_task_family` 的"最少 verified 经验优先"确定性选择器，未宣称学习型探索策略）。

Phase A 单独口径：20 轮 ×3 实例 = 60 次尝试，39 verified / 21 rejected，候选决策全部 `MEMORY_ONLY`，`user_inbox_present=false`，墙钟 7.44 s。

---

## 4. 常驻运行、重启与候选更新

**Phase B（真实后台进程，`--profile demo`）**：3 轮自主周期推进；2 条用户消息（`bot ask`）均被优先处理并按序回答；`pause` 后周期停在 22→22（8 s 观察窗口，暂停期间投递的第 2 条消息进队列未丢），`resume` 先答消息再继续 → 23；`stop` → 进程退出 rc=0、`lock.json` 已清理。

资源（`resource_audit.json`）：CPU 占整机 **42.5 %**（政策上限 50 %），torch 线程 20 → **13**（由代码按 `28×50%−1` 强制夹紧），RSS 峰值 **216.2 MB**（上限 2048 MB），累计 `cpu_s` 75.61 s / `wall_s` 6.36 s，磁盘远低于 512 MB 上限。冷却期内为可中断等待，无忙等。

**Phase C（重启一致性）**：23 → 26 轮，新增 18 个唯一任务实例、**0 次重复**，`bot_state.db` 完好，stable 文件 sha256 逐位不变。

**Phase D（显式授权的候选更新，四臂对照）**：

```
授权：bot candidate --i-authorize --steps 48   （promotion_mode 仍为 manual）
预算：adapters 全部实验臂合计 48 optimizer.step（任务书 §9 上限 120 步 / 300 s 之内）
      执行 48 步：neural 24 + negative 24，预算剩余 0（硬闸用满，不是"没跑"）
墙钟：本轮 37.9 s（上限 60 s 跨臂合计，按臂各 30 s）
世系：parent 6404f453b0aa…  → child 393defc41444…，逐位继承断言通过
参数：named_parameters 23 张量中 17 张变化，max_abs_delta 0.073495
缓冲：2 张 buffer 变化（block_load_count 计数跳到 9472），评测前已复位 reset_applied=true
门槛：new_skill_gain 0.0000 < 预登记 0.10 ❌ | old_task_forgetting 0.00 pp ✅ | weight_changed ✅ | inheritance ✅
判定：REJECTED → 候选文件隔离到 results/bot_r1_e2e/rejected/candidate_*.pt
stable：sha256 4165ab3a11e2…（与源权重 experiments/m5_language/m52_s0_checkpoint.pt 逐位一致）全程未变
晋升：不带 --i-approve 的 `bot approve` 被拒绝；REJECTED 候选不自动回捞
```

门槛口径说明：本轮没有为了"能跑进候选训练"而调低预登记阈值——`--verified-threshold` 用默认 32，而 Phase A 已积累 39 条 verified，触发条件本来就满足；步数 48 = `configs/bot_*.yaml` 里预登记的 `candidate.max_steps`，不是临时改出来的数。

研究等级：**REJECTED**（够不上 `SIGNAL_SINGLE_SEED`——封存集没有任何正信号）。基线 0 + 单臂 24 步的条件下 +0.10 门槛在数学上不可能达到；这一轮的目的是验证"授权→训练→独立评测→拒绝→隔离→stable 无损"这条链在代码里是真的被强制，而不是靠提示词。**不能**据此宣称 N-MIND 具备参数级语言/推理能力。

---

## 5. 测试

```
tests/test_bot_runtime_r1.py            16 passed in 45.16s   （§9.1–§9.15 + §9.16 实验回归）
全仓测试套件                            108 passed in 53.29s（exit 0，修完 bot/cli.py 的 --offline 位置 bug 后重跑，M4/M5 回归未破坏）
```

单元测试用可插拔单调时钟（`FakeClock`/`StepClock`）与微型 `NMindLM`（`d_model=32, n_blocks=2`）跑真实前向/更新，不读写仓库里的任何权重；`§9.3/§9.11/§9.13/§9.15` 走真实子进程 CLI。

**端到端实验抓出并修掉的真实缺陷**（1–8 由断言发现，9–10 由"照着文档逐条敲一遍"发现——两类都算真实缺陷）：

1. **共享预算饿死负对照臂**：`masked_stream_update` 拿"全部臂累计已用步数"去比"本臂上限"，导致首轮 negative 对照 **0 步**、四臂对照名存实亡（Phase D 实测 24/48 步，探针 `budget_is_hard_cap_after_use` FAIL）。修复：臂内计数从 `used0` 起算 + 总闸单独判 + 步数与墙钟按训练臂均分。补 §9.16 回归测试锁定。
2. **参数取证混入计数器**：`parameter_audit` 跑在整个 `state_dict()` 上，`max_abs_delta` 被 `block_load_count`（跳到 9472）污染——正是 M4 虚假增益的同款取证口径漏洞。修复：拆成 `parameter_change`（仅 `named_parameters()`）与 `buffer_change_during_training`，门槛 `weight_changed` 只看真参数。
3. `PAUSED → IDLE_EXPLORE` 非法转移使常驻进程在 `bot resume` 后崩溃 → 练习循环开头加 PAUSED 守卫 + `run_forever` 自动恢复。
4. §9.2 抢占此前只在 pause/stop 上生效 → 补实例间 inbox 检查与 `_preempt_for_user`。
5. `json_extract` 题面不含载荷本身，导致练习集与封存集出现同题不同答案的矛盾监督（§9.6 FAIL）→ 题面内嵌 JSON + **题面级**封存互斥。
6. 日志/事件无容量上限（§9.12）→ 按字节/行数轮转 + `prune_events`。
7. `READY → CANDIDATE_TRAINING` 非法转移使 `bot candidate` 直接报错 → 补入合法表（仅人工授权入口）。
8. 驱动脚本对 `_work/` 子目录 `unlink()` 在 Windows 抛 PermissionError → 目录走 `rmtree`。
9. **帮助文本承诺了不存在的 profile**：`--profile safe-local` / `candidate-demo` 列在 argparse choices 里，但没有 `configs/bot_safe-local.yaml`、`configs/bot_candidate-demo.yaml`，照文档敲就 `SystemExit：找不到 profile 配置` → 补齐两份配置，并加断言"每个 advertised profile 必须能真的执行"。
10. **文档命令写法跑不通**：`bot once --offline` 报 `unrecognized arguments`——`--offline` 这类全局开关此前只挂在 `bot` 组上，没挂到叶子子命令（项目其余命令都是"前后都收"）。修复 `bot/cli.py` 的 `add()` 调 `_add_common(sp, suppress=True)`，并加回归断言。

提交前另做了一次死代码清理（`ruff --select F,E9` 对本任务文件现已 0 命中）：删掉 `bot/{experience_bridge,goal_manager,status,policy}.py` 里 4 处未使用的 import、驱动脚本里 3 处"调用后从未读取"的局部赋值与 1 处未用 import、测试文件里 2 处未用 import。被删的都是只调用不读值的语句，CLI 调用本身保留，**九份产物由清理前同一逻辑生成，数字未受影响**；仓库其余历史文件在本规则集下仍有既有告警，未一并改动。

---

## 6. 可直接复制执行的命令（Windows PowerShell）

```powershell
cd D:\cpu本地模型开发\n-mind
$env:PYTHONPATH = "src"; $env:PYTHONIOENCODING = "utf-8"

python -X utf8 -m nmind.cli bot doctor                       # 体检：依赖/线程/权重/策略/锁
python -X utf8 -m nmind.cli bot once --offline --cycles 20   # 无人值守自主练习（不更新权重）
python -X utf8 -m nmind.cli bot eval                          # 封存集自由生成 EM（不训练）
python -X utf8 -m nmind.cli bot run --offline --profile safe-local   # 常驻（Ctrl-C 或下面的 stop）
python -X utf8 -m nmind.cli bot status                        # 另一个终端：状态/心跳/预算余量
python -X utf8 -m nmind.cli bot ask "你好，你现在能做什么？"   # 用户消息，优先处理
python -X utf8 -m nmind.cli bot pause                          # 跨进程控制：控制文件，不是进程内布尔
python -X utf8 -m nmind.cli bot resume                         # 恢复（先补答攒下的消息）
python -X utf8 -m nmind.cli bot stop                               # 在安全边界退出并释放锁
python -X utf8 -m nmind.cli bot goals                          # 目标池与 selection_source=rule
python -X utf8 -m nmind.cli bot audit --limit 20               # 世系、隔离区、经验计数
python -X utf8 -m nmind.cli bot logs --cycles --limit 5         # 自主循环明细（含每次尝试与 oracle 判分）

# 候选更新与晋升（都是显式动作，默认配置下 auto_candidate_training=false）
python -X utf8 -m nmind.cli bot candidate --i-authorize --steps 48
python -X utf8 -m nmind.cli bot approve --i-approve            # 仅 PROPOSED 且人工确认才原子切换
python -X utf8 -m nmind.cli bot rollback                        # 回到上一次晋升前备份
```

一键复现全部实测（A–E 五阶段 + 11 探针 + 9 件产物）：`powershell -File experiments\bot_runtime_r1\reproduce.ps1`

本机完整重跑记录：修掉缺陷 10（`bot once --offline` 的 argparse 位置 bug）之后，脚本从头跑到尾打印 `全部步骤完成。`，退出码 0，安全探针 11/11 PASS。
修之前它在第 4 步 `bot once --offline --cycles 20` 就 `unrecognized arguments: --offline` 中断——也就是缺陷 9/10 这两条，只有真的按文档逐条敲才能抓到。
脚本需以 UTF-8 **带 BOM** 保存（`reproduce.ps1` 已如此）：Windows PowerShell 5.1 读无 BOM 的 UTF-8 中文脚本会按 GBK 解析而报语法错误。

---

## 7. 默认安全姿态（`configs/bot_runtime.yaml`，由代码强制）

`network_enabled=false`、`max_network_requests_per_cycle=0`、`auto_candidate_training=false`、`promotion_mode=manual`、`allow_external_code_execution=false`、`allow_remote_model_api=false`、`max_optimizer_steps_per_cycle=0`、`max_training_wall_seconds_per_cycle=0`、`stable_checkpoint_readonly=true`、`min_cooldown_seconds=60`、`max_idle_cpu_percent=50`、`max_rss_mb=2048`、`max_disk_mb=512`、`practice_task_allowlist=[arithmetic, grid_rules, json_extract]`。

安全探针 11 项全 PASS，含：白名单拒绝野目标、网页/消息内容无法改阈值或调用 shell、真实自主循环内 socket 出网计数 **0** 且 `src/nmind/bot/*.py` 扫描 0 处网络调用点、`FetchConfig().offline` 默认为 True、stable 文件 sha256 在 A–E 全部阶段逐位不变、预算用满后下一轮仍 `MEMORY_ONLY`、拒绝/不确定经验无法触发 `neural_adapt`、封存样本 0 泄漏、`stop.flag` 释放锁。

---

## 8. 已知限制与下一步（不粉饰）

1. **神经增长未证明**：需要更长训练与更大封存集；但按任务书要求，扩预算前必须先有独立于阈值的预登记判据，不能为了提高通过率调门槛。
2. **grid_rules 无工具路径**：该族 27/27 全 rejected，`{family}:tool` 记 null；要区分"没学会"与"没有对照工具"，需补一个可验证的程序 oracle。
3. **任务族仍偏玩具**：三类任务都是程序可判分的合成题，离自然语言应用还有距离；M5.2 的两实体绑定 `BINDING_FAIL` 是更靠前的瓶颈。
4. **P4 联网、P5 自启/面板/打包未开发**：联网需要单独的 allowlist + 私网/重定向/流量硬限审计，R1 不做（任务书 §7）。
5. **单 seed**：本轮所有结论都是单 seed 诊断性结果，未做多种子重复；`SIGNAL_SINGLE_SEED` 都未触发。
6. **CPU 占用 47 %**：靠夹紧 torch 线程达标；常驻 `safe-local` 档冷却 60 s 时实际占空比远低于此（demo 档冷却 3 s 是观察用快节奏，已在产物中注明来源）。

---

## 9. 阶段出口对照（任务书 §10）

| 阶段 | 出口 | 实测 |
| --- | --- | --- |
| P0 仓库审计 | API 地图 + 测试基线 | ✅（复用 `online/scheduler`、`online/promotion`、`learning/online_adaptation`、`learning/replay`、`ChatSession`，未另造第二套调度器） |
| P1 运行时 + CLI | 启动/停止/空闲/用户输入三路径可测 | ✅ 已交付并实测 |
| P2 无人值守安全练习循环 | ≥3 类任务、无消息自动执行 | ✅ 已交付并实测（3 族 81 实例） |
| P3 候选学习集成 | 一轮"可回滚的失败实验"即验收 | ✅ 已交付并实测（REJECTED + 隔离 + stable 无损）；真实能力增长单独判定 = 未通过 |
| P4 可选联网 | 默认关闭 | ⛔ 未开发（保持默认关闭） |
| P5 Windows 体验 | 可选面板/自启/打包 | ⛔ 未开发 |
