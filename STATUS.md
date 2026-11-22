# STATUS.md — 当前真实进度

更新：2026-11-22（AutoClaw 会话）。所有数字来自本机实际运行，证据在 `results/` 与 `experiments/`。

## 环境

- Windows 11，Python 3.12.10，torch **2.10.0+cu128**（2026-11-21 15:44 由 CPU 版升级，阿里云镜像），pytest 9.2.0，包已 `pip install -e .`
- i7-14700KF（28 逻辑核，AVX2，无 AVX-512）/ 32GB RAM / **RTX 5070 12GB 已启用**（sm_120，驱动 596.36，doctor 报设备 cuda）
- 云端 `mode: off`，开发全程零网络调用（pip 安装除外）
- Git：2026-11-21 初始化；**注意可能有并行会话（Codex）在同一仓库工作，提交时只 add 自己的文件**

## 可复现命令

```powershell
# M1 过拟合验收（约 2 分钟）
python experiments/2026-11-21_m1_overfit.py
# 基线与基准（证据：results/baselines.json、benchmark_cpu.json、evaluate.json）
python -m nmind.cli baseline --json --epochs 12 --seed 20261121
python -m nmind.cli benchmark --json --seed 20261121
python -m pytest tests -q          # 27 passed
```

## M4 增量学习：参数连续性修复与首次四臂 smoke（2026-11-22，AutoClaw）

- **P0 bug 修复（实锤并闭环）**：`train_continual()` 的 B 阶段此前调用 train() 重新 build_model()——同一 seed 只重建相同初始权重，A 阶段学到的参数从未进入 B，forgetting 指标无法解释为真实遗忘。修复：`train()` 新增 `model=`/`probe=` 显式入参（提供时跳过 build_model，语义冲突直接报错不静默重建）；`train_continual()` B 阶段改为从 A 后权重逐位克隆 + 内置继承断言；optimizer 增量阶段新建并记录；采样 seed 与初始化 seed 分开记录。
- **新增** `src/nmind/learning/online_adaptation.py`：`fit_existing_model`（neural_adapt 唯一合法入口）+ AdaptationBudget 硬预算守卫 + 逐位继承/冻结取证（torch.equal，不用容差）+ transient buffer（load_bias/block_load_count）单独记录与复位。
- **首次四臂增量学习 smoke**（experiments/m3_incremental/，1 预声明 seed，104/120 步，20.5s，RSS 645MB）：父 checkpoint = C_sparse_ce_s20261131（A 遮挡段 0.6594，与 M2.2 冻结记录一致）。新任务 B = reversed_control（训练 96 条 / 未见测试 32 条，seed 20261141/42）。结果：Neural-Head 未见 B 冲突步 0.6019→**0.8889（+28.7pp）**、A 0.6594→0.6609（零遗忘）；Neural-Core 0.7191（+11.7pp）但 A −4.2pp（门槛内）；memory-only +2.5pp 且参数逐位不变；lr=0 对照逐位不变。判定：PIPELINE_FIXED / NEURAL_UPDATE_VERIFIED / NEW_SKILL_SIGNAL_SINGLE_SEED (Neural-Head) / RETENTION_PASS_SINGLE_SEED；RESEARCH_VALIDATED 不由 1-seed 授予。报告：experiments/m3_incremental/m3_incremental_report.md。
- 测试 **72 项全绿**（64 项既有 + tests/test_true_continual.py 8 项新断言）。
- H3b/H4b v2 的 10-seed 全量重训 **DEFERRED_NOT_CANCELLED**（脚本 train_ood_v2_full.py 就绪未运行，约 2.5h CPU，待小袁批准）。

## M4 联网自主神经学习：全链路打通 + 首轮 REJECTED（2026-11-22 第二轮，AutoClaw）

- **链路 PASS**：src/nmind/online/（fetcher/fixtures/extractor/provenance/verifier/experiences/scheduler/promotion/explorer）+ learning/online_trainer.py（有界流更新：CE + λ_replay·CE_replay + λ_anchor·锚定）+ evaluation/lifelong.py。模拟网页夹具发布结构化规则 control-reversed-v1 → 解析/程序化验证 → 96 条 verified 经验流 → 调度触发候选更新（四臂+负对照，96/120 步，48.6s，RSS 647MB）→ 封存评测 → 晋升审计。
- **判定 candidate_status=REJECTED**：B_OOD 冲突步 gain +2.9pp < +10pp、A 遗忘 −3.75pp > 3pp；其余门槛过（继承逐位/冻结 0 违规/负对照 −45.6pp 无泄漏/neural>frozen）。候选隔离 ckpts/rejected/，stable 逐位未动。参数学习本身 PASS（17/19 非零变化）＝NEURAL_UPDATE_VERIFIED；可泛化技能 FAIL；真实联网 NOT_TESTED（门槛未过不触网）。
- **关键发现**：训练态 router.load_bias 漂移承载 ~26pp 的“表面适应”——不复位时 B_OOD 0.8889，复位后 0.6244。buffer 复位协议（任务书要求）防止把 buffer 漂移冒充参数学习，本轮给出定量证据。
- **失败诊断**：λ_anchor=0.1+λ_replay=0.3 在 40 步内过度约束 adapter（对照：上轮无锚定 48 步单 seed +28.7pp）；Hybrid 证明锚定+回放保 A（−0.31pp）但代价是 B 几乎不学。修复需新预注册并批准。
- **安全**：恶意网页夹具 instruction_detected→rejected→不进学习流（测试断言零执行/零改配置）；dedup 生效；sealed 零泄漏；offline 硬断。configs/online_learning.yaml 三开关默认关（network/auto_weight_update/promotion=manual）。
- 测试 **82 项全绿**（+tests/test_online_learning.py 10 项）。报告：experiments/m4_online/M4_REPORT.md。

## M4.1 稳定性预注册复验：遗忘门槛修复、外推瓶颈定位（2026-11-22 第三轮，AutoClaw）

- 预登记（小袁指令）：λ_anchor=0、λ_replay=0.15、48 步、1 seed（m4.1-stability-001）。结果 **REJECTED（4/5 门槛过）**：A 遗忘 −3.75→**−2.66pp ✅**（回放防遗忘起效）；B_OOD(T=32) **+1.55pp ❌**。参数证据：9/19 变化、max 权重 delta 0.1116、冻结逐位 0 违规（训练时+保存权重双重核验）；6830 是 block_load_count 计数器 buffer 累加，已剔除重算（口径更正）。
- **诊断修正（诚实）**：M4 报告“λ 过约束”不是 OOD 失败主因——同数据集上锚定配置 B_test 反而更好（+10.4pp vs +5.8pp）；B_OOD 在所有已测配置下都不达标（+2.9/+1.55pp）。真正瓶颈 = **T=32 长度外推**（训练时域外的递归状态分布）；run 间方差大，单 seed λ 归因不可靠。
- 报告：experiments/m4_online/M4_1_REPORT.md；参数证据：m41_param_evidence.json。
- 路线（小袁 2026-11-22）：M4.2=A→B→C→D 连续技能学习+遗忘矩阵（规划）；M5=互联网即知识空间（任务书已下，P0/P1/P2/P3 优先）；M6=多模态仅接口规划。

## 最新真实结果

### 1. M1 过拟合 + 训练预算扫描（experiments/2026-11-21_m1_overfit.json，seed 20261121）

| 实验 | 数据 | 梯度步 | loss 末值 | 测试遮挡段 acc | 探针 acc（随机≈0.028） |
|---|---|---|---|---|---|
| A 过拟合 | 32 条 | 800 | 0.415 | **0.941** | 0.79（train） |
| B 默认数据 | 96 条 | 240 | 1.221 | 0.603 | 0.285 |
| C 加长 | 96 条 | 720 | 0.670 | 0.609（平台 0.55–0.61） | **0.751** |
| 多数动作基线 | — | — | — | 0.444 | — |

结论：梯度通路健全，架构能学（A 证明）；B/C 显示**状态里记住了目标（探针 0.75）但决策读出跟不上（0.61）**——瓶颈在读出/优化，不是记忆容量。

### 2. 基线对照（results/baselines.json，72 步同预算同数据）

| 主干 | 参数 | acc | 遮挡段 acc | 单序列 ms |
|---|---|---|---|---|
| dense_n_cell（NModel 关路由，纯 CE） | 282,589 | 0.857 | **0.933** | 152.0 |
| gru_2layer | 54,248 | 0.530 | 0.556 | 2.2 |
| shared private_only | 204,008 | 0.633 | 0.663 | 11.1 |
| shared mixed | 129,128 | 0.624 | 0.650 | 10.0 |
| shared shared_only | 29,288 | 0.651 | 0.681 | 5.5 |

诚实结论：
- 同架构**稠密**版 72 步就到 0.933，而稀疏路由版 720 步只有 0.61 ⇒ **当前稀疏路由/联合损失训练配置拖累了学习**，N-MIND 尚未证明优于自己的稠密消融，更没证明优于 GRU 的性价比。这是 M2 最优先待解问题（公平口径：基线只用 CE 损失，需在 5 项损失下重跑稠密版对照）。
- 共享主干省参数不降准确率（29k 参数 > 204k 参数的遮挡段 acc）。

### 3. CPU 基准（results/benchmark_cpu.json，T=48，60 次，8 线程）

dense 35.4ms/序列（P95 51.2）；sparse k=1 **21.1ms**（1.68×）；k=2 23.6ms；k=4 28.5ms。
稀疏路由在 CPU 上有真实延迟收益，但准确率收益未证实（见上）。

## 本次会话修复的 bug（评测层此前从未成功运行过）

1. `evaluation/{baselines,benchmarks,tasks}.py` 循环导入（`from . import datasets` → `..learning`）——`baseline`/`benchmark`/`evaluate` 三条命令修复前必崩。
2. `benchmarks.cpu_benchmark` repeats 超过测试集数量时索引越界 → 改为按 `i % len(rows)` 循环。
3. `_submodule_ms` 编码器计时喂错张量（z 64 维当 26 维观测）。
4. `Timing.as_dict` 缺 `per_event_ms`；CLI 引用不存在的 `peak_rss_mb` 键。
5. `baselines.fit/score`：NModel 返回 `action_logits`、GRU 返回 `logits`，键名自适应。
6. `SharedTrunkCell` 共享槽位输入未复制导致 GRUCell 批大小不匹配（16 vs 64）。
7. `EventEncoder` 在 text_feat/numeric 缺省时造隐式 CPU 零张量 → GPU 训练崩；已改为跟随输入设备（GPU 冒烟实验暴露）。

## GPU 状态（2026-11-21 启用，小袁确认安装）

- torch 2.10.0+cu128 落位（RTX 5070 需 cu128+；download.pytorch.org 直连 ~400KB/s 卡慢，改用阿里云镜像 `--find-links` 完成安装）
- GPU 冒烟实验（experiments/2026-11-21_gpu_smoke.json）：同 seed 6 epoch demo，GPU 43.4s vs CPU 17.1s，**小模型下 GPU 无优势**（kernel 启动开销主导，显存峰值仅 103MB）；GPU 训练→CPU 推理最大概率差 5.96e-08（等效逐位一致）
- 测试 32 项全绿（含 test_datasets.py 5 项——并行会话新增）
- 结论：GPU 先保留用于更大模型/更大批次，当前 M2 对比实验继续以 CPU 口径为准（两边数字都来自实测，不混用口径）

## 阻塞

- ~~GPU：torch CPU 版无法用 RTX 5070~~ → 已解决（cu128 已装，见上方 GPU 状态）
- 未实现：Transformer 基线、多种子重复、持续学习实验运行记录、门控收益验证
- 并行会话正在做 M2（m2_* 实验文件与 test_datasets.py 出现在仓库）；跨会话提交前先 `git status` 分清归属

## 下次会话第一件事

1. M5 P1-P3（CapabilityTracker/ResourceManager/SkillRegistry）+ P4/P5 触发管线已建，集成实验见 experiments/m5_cognition/；神经更新继续 NOT_TESTED（m4.1 REJECTED，等混合时域新预注册批准）。
2. 混合时域经验流新预注册（B 训练流 T=24+32 混合，攻长度外推）——**需小袁批准**（约 1-2 分钟 CPU）。
3. H3b/H4b v2 10-seed 全量重训——DEFERRED_NOT_CANCELLED，**需小袁批准**；F10 配置封装、M3 Step5/6 排队。

## M2.2 机制验证（2026-11-21 第四轮，AutoClaw）——含 10-seed 定案补充

- **pred 分支缺陷实锤并修复**：legacy 模式预测器把目标当输入（cos(z_hat,z)=0.998、动作敏感性 0.025、读出不超过拷贝基线），三臂 world_model_validated=false；新增 `pred_mode="next"`（(s_t,a_t)→s_{t+1}，与 error 门控语义一致）后 F10_next 两 seed **validated=true**（敏感性 1.05-1.15、z_hat 单独读出下一位置 0.92-0.93 **超过真值等价读出 R2**、主任务不退化 0.91-0.97）。
- **三个真 bug 连锁修复**：① train() 的 pred_mode kwarg 不同步进 cfg → 存档记录错误语义；② _model_kwargs 漏记 router_ste/pred_mode → 权重加载成功但语义静默回退；③ run_next_mode 未设 cfg。均已修复并重训验证（同 seed 逐位复现）。
- **配对 seed 定案（10 seed，ddof=1）**：C_sparse_ce 0.877±0.087（旧"0.941±0.002 稳定"是 seed 族假象，最差 0.659）；**F10_ste 0.931±0.050，配对 C−F10=−5.41pp（F10 反超，7/10 seed 优），bootstrap 95% CI 上界 +1.28pp < 3pp 门槛——配对口径下稳健满足**；F10 方差为 C 的 57%。M2 冻结基线更新为配对口径（experiments/m2_2_validation/f10_paired_combined.json）。
- **轨迹干预**（5 臂×3 seed，同一 fork 点）：detach/STE/关 pred 任意组合 240 步内无法把 D 从坏盆地拉出——D 失败是早期锁定，修复必须从初始化生效；"早期锁定"假设获证据支持。
- **决策 D6**：pred_mode 默认翻转为 next（legacy 是已实锤的缺陷目标）；router_ste 保持 opt-in（C 臂对照依赖）。多步闭环 rollout 完成：next 模式误差饱和（h1 0.59→h10 0.66 不爆炸），解码位置超 copy/majority 基线，h10 的 copy 高值是钉住态退化区（与 H4 同源）。
- **M2.2 判定**：F10 ≤3pp 门槛**配对口径稳健满足**；CONDITIONAL 保持（差 H3b/H4b 两项，M2.2 COMPLETE 前不升 ACCEPTED）。报告：experiments/m2_2_validation/report.md。
- 测试 52 项全绿；环境审计 environment.json；代码地图 docs/code_map.md。

## H3b/H4b 口径复核与 v2（2026-11-22）

- H3b v1 作废为验收证据：标签 oracle 读取完整障碍图，但 26 维输入没有障碍地图且轨迹零碰撞；“BFS 可解”也与实际曼哈顿贪心标签不一致。旧 H3b 数字仅保留为历史诊断。
- H3b v2（`h3b-v2-observable-bfs`）已完成数据层修正：观测追加 6×6 障碍图（26→62 维）、标签改为确定性 BFS 最短路、每条轨迹必须实际到达目标、dev/holdout 逐拓扑实例隔离、manifest 指纹覆盖实际 obs/actions/start/goal。旧 checkpoint 输入维度不兼容，**尚未重训、尚无模型成绩**。
- H4b 重新定义为“反转规则分布偏移”，不再声称规则切换识别。评分拆为冲突步/巧合步/总分：F10 10-seed 为 **0.589±0.145 / 0.862±0.068 / 0.720±0.084**；C 冲突步 0.471±0.125，配对 C−F10=-0.118，bootstrap 95% CI [-0.214,-0.018]。由于没有规则提示与真实闭环，不能外推为自主规则切换能力。
- H4b cued v2 另建正常/反转成对样本，同一 start/goal、每步显式规则提示（27 维）；旧 26 维 checkpoint 同样不兼容，等待配对课程重训。
- 证据：`experiments/m2_2_validation/ood_h3b_h4b_v2.json`；测试 **64 项全绿**。M2.2 继续保持 `CONDITIONAL`，下一关是 H3b/H4b cued v2 同口径重训。

## M2.2 机制验证（2026-11-21 第四轮，AutoClaw）

- **pred 分支缺陷实锤并修复**：legacy 模式预测器把目标当输入（cos(z_hat,z)=0.998、动作敏感性 0.025、读出不超过拷贝基线），三臂 world_model_validated=false；新增 `pred_mode="next"`（(s_t,a_t)→s_{t+1}，与 error 门控语义一致，默认 current 不动历史口径）后 F10_next 两 seed **validated=True**（敏感性 1.05-1.15、z_hat 单独读出下一位置超拷贝基线 +30pp、主任务不退化 0.91-0.97）。
- **配对 seed 反转旧叙事**（预登记 20261131-35，experiments/m2_2_validation/f10_paired_seeds.json）：C_sparse_ce 0.853±0.113（旧 seed 族 0.941±0.002 是 seed 运气，最差 0.659）；**F10_ste 0.934±0.071，配对均值 C−F10=−8.0pp（F10 反超），4/5 seed 优于 C**；bootstrap 95% CI [−21.6, +5.6] 跨零，n=5 证据方向明确但未定案。机制自洽：C 的路由器从未被任何损失训练（CE 模式连均衡/熵都没有），F10 的任务梯度路由跨数据 seed 稳健。
- 轨迹干预实验（同一 fork 点 5 臂分叉）结果见 experiments/m2_2_validation/pred_interventions.json。
- 环境审计：experiments/m2_2_validation/environment.json（运行时检测：torch 2.10.0+cu128、cuda 运行时可用但本轮实验全部 device=cpu）；代码地图 docs/code_map.md。
- 测试 52 项全绿（+pred_mode 回归 3）。

## M2 收尾（2026-11-21 第二轮，AutoClaw）

- **全量重载验证**：22 个 checkpoint 全部重载评分，Δ=0.0 逐位复现（registry.json）；sha256 已登记。
- **冻结 holdout**：H1 新对/ H2 长48 基本不掉点；**H3 障碍布局 OOD 掉 13-18pp（全部臂）**；H4 规则反转为弱对照（角点退化区，非泄漏信号）。
- **机制诊断**（gradient_diagnostics.json）：稀疏模式下 action/pred/probe 对 router 梯度全为 0（任务盲路由），仅 balance/entropy 训练它；稠密 B 的 router 有任务梯度且 action×pred cosine -0.453 但仍收敛；D 是"从未收敛"不是"先崩后逃"。
- **多 seed（20261122-24）**：C 0.941±0.002（架构稳定）；D 0.627±0.055（复现失败）；F1 0.895±0.016；**F7 0.872±0.038，其单 seed 0.923 确认为最好种子噪声，被 F1 反超**。
- **新增消融**：F9（只软化、不 detach）0.593±0.039 ≈ D ⇒ detach 必要；**F10（F7+STE 路由）0.914±0.031（n=5），距 C 2.7pp——均值首次进入 ≤3pp 目标线**，方差大待定案。
- **CPU 基准（独占）**：稀疏 25.6ms vs 稠密 38.7ms p50 @1 线程 = **1.51×**（旧 1.68× 撤回）；8 线程小算子受竞争（p95 升至 50ms）；RSS ~503MB。
- **判定**：M2 **CONDITIONAL**（未闭环，差 F10 定案 + pred 显式有效性指标）。详见 experiments/m2_closeout/m2_closeout_report.md。
- 测试套件 49 项全绿（+router_ste 3、+closeout 回归 3、+micro_lang 11）。
