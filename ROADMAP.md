# ROADMAP.md — 里程碑、验收条件与当前状态

> 状态标记：✅ 已验收（有 JSON 证据）｜🔶 部分完成｜⬜ 未开始。验收以 `results/` 与 `experiments/` 的真实 JSON 为准。

## M0 环境与骨架 ✅

- [x] 目录骨架、`pyproject.toml`（src 布局、`nmind` 入口、pytest 配置）
- [x] `doctor` 体检：CPU 路径（torch 2.10.0+cpu）、8 线程、AVX2、参数 282,589
- [x] 单元测试 27 项（NCell 稠密/稀疏等价、路由器、工作空间读写掩码）
- 验收命令：`python -m nmind.cli doctor` / `python -m pytest tests -q`

## M1 可训练递归网络（0.5M–2M 原型）✅

- [x] Encoder / NCell×8 块 / 路由器 / 全局工作空间 / 预测器 / 决策头 / 治理器
- [x] 极小批次过拟合验收（experiments/2026-11-21_m1_overfit.json 实验 A）：
  32 条轨迹 800 次更新，loss 3.309→0.415，训练遮挡段准确率 **0.950**，测试遮挡段 **0.941**
- [x] checkpoint 保存→重载逐位一致（demo：`save_load_bitwise_identical: true`）
- [x] CPU 推理闭环（`run`：约 1.0–5.4ms/步，SQLite 记忆检索参与决策）
- 🔶 待办：把默认配置的训练预算提高（当前 6 epoch ≈ 36 次更新，遮挡段 0.459 ≈ 多数类基线 0.444，属欠训练）

## M2 公平对照实验 🔶

- [x] GRU / 稠密 N-Cell / 共享主干三组基线 + CPU 基准跑通并记录（results/baselines.json、benchmark_cpu.json）
- [x] 发现并修复评测层三处 bug（见 STATUS"已修复"）
- ⬜ **Transformer 基线未实现**（宁缺毋滥，不许拿没验证的实现充数）
- ⬜ 严格同损失对照：当前基线 fit() 只用 CE，N-MIND 主训练是 5 项联合损失——需要把稠密 N-Cell 在 5 项损失下重训，回答"稀疏路由 vs 稠密"谁拖了后腿（今天发现稀疏版 720 步遮挡 0.61 << 稠密版 72 步 0.93）
- ⬜ 多种子重复（≥3 seeds）

## M3 动态计算（HaltingGate / 预算）🔶

- [x] governor（max_step_ms / max_recurrence_depth / 降级计数）+ evaluate 的 budget-难度对照表
- [ ] 门控训练收益验证（当前 gated 与 fixed_k1 准确率相同：门控还没学到"难任务多算"）

## M4 长期记忆与持续学习 🔶

- [x] SQLite 情景记忆（FTS5）、检索器、巩固器、墓碑删除、replay 缓冲
- [x] `train --continual`（A→B→重测 A，replay 开关）实现完成
- ⬜ 持续学习实验真实运行并记录遗忘量（下一步第一件事之一）

## M5 语言能力试验 ⬜

- 当前 `language.source: hash`（哈希袋词特征，不是语言模型）。
- 前置条件：M2–M4 验收通过。从原始文本训练自有 tokenizer，禁止导入预训练权重。

## M6 普通设备部署 ⬜

- ONNX 导出 / INT8 量化 / 可复现性能报告。前置：M3 验收。

## 已知阻塞

1. torch 为 CPU 版 → RTX 5070（Blackwell, sm_120）不可用；装 cu128 版 torch 约 2.5–3GB 下载，属可逆的环境变更，待小袁确认带宽/时间再装。
2. 稀疏路由版训练显著落后于同架构稠密版（0.61 vs 0.93 遮挡段）——这是当前最重要的科学问题，在解释清楚之前不扩展模型规模。
