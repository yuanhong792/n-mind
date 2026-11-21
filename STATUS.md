# STATUS.md — 当前真实进度

更新：2026-11-21（AutoClaw 会话）。所有数字来自本机实际运行，证据在 `results/` 与 `experiments/`。

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

1. 在 5 项联合损失下重训稠密 N-Cell，回答"是路由还是多损失拖累了学习"（预期 30–90s CPU）。
2. 跑 `train --continual --json` 记录真实遗忘量（M4 验收）。
3. 依据 1 的结论调稀疏版训练（先怀疑 pred/var/balance/entropy 权重），再谈加大模型。
