# N-MIND · 脉脑

**Neural Memory & Incremental Dynamics** · **神经记忆与增量动力学**

> 从零学习、持续记忆、动态计算，让神经网络在普通设备上运行。  
> *A from-scratch, local-first neural architecture research project for memory, incremental learning, and efficient computation.*

[![Python](https://img.shields.io/badge/Python-%E2%89%A53.11-blue)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-PyTorch-orange)](https://pytorch.org/)
[![Development](https://img.shields.io/badge/Status-Research%20Prototype-yellow)](./STATUS.md)
[![Local First](https://img.shields.io/badge/Inference-CPU%20First-brightgreen)](./docs/architecture.md)

**[English summary](#english-summary) · [架构](#架构与实现) · [实验结果](#已验证的实验结果) · [快速开始](#快速开始) · [路线图](#研究路线图)**

## N-MIND 是什么？

**N-MIND** 是一个探索新型人工智能神经网络架构的开源研究项目，英文全称为 **Neural Memory & Incremental Dynamics**，中文直译为 **神经记忆与增量动力学**，暂定中文名称 **「脉脑」**。

名称对应四个核心概念：

| 概念 | 含义 | 对应研究方向 |
| --- | --- | --- |
| **Neural · 神经** | 用神经网络处理信息与进行推理 | N-Cell、递归状态计算、任务驱动学习 |
| **Memory · 记忆** | 持续保存、检索与更新长期记忆 | 工作空间、情景记忆、经验重放 |
| **Incremental · 增量** | 逐渐掌握新知识与技能，探索无需每次完整重训的学习机制 | 局部参数更新、持续学习、防遗忘实验 |
| **Dynamics · 动力学** | 通过动态状态变化和循环计算处理信息 | 稀疏路由、预测误差、递归计算与资源预算 |

**定位说明：** N-MIND 是本项目提出的**候选研究架构名称**，目前并非学术界已验证的标准神经网络架构，也不是已具备通用智能能力的产品。项目的数学组件与已有门控循环网络、稀疏路由、工作记忆等研究存在关联；任何性能创新主张必须经过公开、可重复的对照实验。

### 三项不可变的研究约束

1. **独立训练**：所有 N-MIND 可训练权重从随机初始化开始；不依赖现有大语言模型的预训练权重、教师模型或蒸馏结果。允许使用基础计算框架和合规的原始数据。
2. **持续学习**：研究系统怎样通过感知、预测、反馈、记忆与受控的神经参数更新，逐渐获得新能力。当前仍处于机制验证阶段，尚未证明可在开放互联网环境中稳定自主学习。
3. **普通设备运行**：优先保证普通 CPU 可运行；消费级 GPU 是可选训练/推理加速器。采用**本地优先、联网可选增强**，云端不能冒充本地神经能力。

## 项目目标与非目标

研究目标包括：低计算成本的状态学习、稀疏神经模块的任务驱动路由、内部工作记忆与长期记忆、递归推理、从零语言学习，以及后续的自主神经可塑性和跨模态扩展。

**当前不声称**：拥有 ChatGPT 级语言能力、可靠的开放域知识推理、真正的通用智能、已验证的自我改进系统，或在所有 CPU 上领先现有架构。测试中的“遮挡段准确率”是特定网格任务指标，不是通用智能评分。

## 架构与实现

当前 R1 原型的数据流：

```text
输入（结构化观测 / 实验性文本）
          │
          ▼
    EventEncoder（事件编码）
          │
          ├── 预测误差 / 不确定性
          ▼
    BlockRouter（Top-K 路由）
          │
          ▼
    N-Cell 块集（选中块更新；其他块保留状态）
          │
          ▼
    Workspace（固定槽位、竞争写入）
          ├───────────► Predictor（下一状态表示预测）
          └───────────► ActionHead（动作/实验字符读出）
                  ▲
                  │
     本地情景记忆 / 运行资源治理 / 可选云端桥接
```

| 组件 | 当前实现 | 需要继续验证的部分 |
| --- | --- | --- |
| **N-Cell** | GRU 风格门控残差状态更新；8 个独立参数块 | 与其他循环单元的公平性能对照 |
| **BlockRouter** | Top-K 稀疏选择；含 STE 任务梯度实验开关 | 多随机种子稳定性、路由可靠性 |
| **Workspace** | 8 槽 × 64 维共享工作空间、竞争写入 | 冲突、溢出、跨任务泛化 |
| **Predictor** | 预测下一步潜在状态并提供误差信号 | 显式单步/多步预测与行动干预有效性 |
| **Memory** | SQLite 情景记忆、FTS5 检索、事实/巩固与 replay 基础设施 | 学习收益、遗忘与记忆污染控制 |
| **Runtime** | CPU 运行、资源预算、checkpoint 与评测 | 多硬件测试、能耗与量化 |
| **Bridge** | 云端策略、隐私控制、审计接口；默认关闭 | 可选联网增强的真实集成测试 |
| **Language** | 字符级微型语言训练、确定性 oracle | 自由生成、真实语言与组合推理 |

R1 当前基线模型约 **282,589** 参数，M3 微型语言实验模型约 **279,220** 参数（配置不同）。现有 R1 **不是**完整 FastCell/SlowCell 双时间尺度网络；动态停止和跨模态感知也尚未证明有效。详见 [实际架构与数学说明](./docs/architecture.md)。

## 已验证的实验结果

> 以下为 **2026-11-21** 仓库中实验脚本及 JSON/报告记录的结果，并非第三方独立复现。所有成绩都必须连同具体数据集、训练配置及测试口径解读。

### 工程与 M1

- M1 从随机初始化训练：实验性 6×6 网格目标遮挡任务，测试遮挡段准确率 **0.941**（指定设置）；能够完成 checkpoint 保存、加载及对照评分。
- M2 收尾记录 **22/22** 个 checkpoint 重载评分一致，已登记 SHA-256，数据划分重叠审计为 0。
- 截至该次报告，自动化测试 **49 项通过**。后续改动应重新运行测试，而不应沿用历史通过数。

### M2 稀疏路由与联合训练

| 实验配置 | 多随机种子遮挡段准确率 | 解释 |
| --- | ---: | --- |
| A · 稠密 + CE | **95.8%**（n=3） | 稠密参考 |
| C · 稀疏 + CE | **94.1%**（n=3） | 稀疏任务基线，相对稳定 |
| D · 稀疏 + 联合损失 | **62.7%**（n=3） | 联合训练失败复现 |
| F1 · detach 预测目标 | **89.5%**（n=3） | 有改善，但仍低于 C |
| F7 · detach + 软化路由正则 | **87.2%**（n=3） | 单 seed 92.3% 不能代表平均性能 |
| F10 · F7 + STE 路由 | **91.4%**（n=5） | 与 C 相差约 2.7 个百分点；种子间波动明显 |

**机制观察：** 原始硬稀疏路由中，主要任务损失没有有效地向 Router 传递梯度；STE 使路由器能够接收替代任务梯度。但“具体哪个梯度冲突导致 D 失败”尚未完成因果验证。

**M2 状态：`CONDITIONAL`，尚未正式验收完成。** F10 仍需稳定性复测；世界模型目前尚未完成显式下一状态预测能力验证。未见障碍布局（H3）暴露明显泛化短板；规则反转（H4）存在测试退化情形。

详细证据：[M2 收尾报告](./experiments/m2_closeout/m2_closeout_report.md) · [消融记录](./experiments/m2_ablation/) · [工件登记](./experiments/m2_closeout/registry.json)

### CPU 推理性能

独立 CPU 基准：i7-14700KF、FP32、单样本、序列长 T=24、1 线程、预热 10 次、正式 60 次。

| 路径 | p50 | p95 |
| --- | ---: | ---: |
| 稠密 A | 38.659 ms | 40.732 ms |
| 稀疏 C | 25.626 ms | 26.720 ms |

指定基准下 p50 **约 1.51×** 加速；旧报告中的 1.68× 口径已撤回。进程峰值 RSS 约 503 MB，主要受 PyTorch 运行时影响，不能把它直接当作模型权重内存。8 线程下的小算子竞争可能提高 p95 延迟。尚未测量能耗，也不能据此推断所有设备上的加速比。

证据：[CPU benchmark JSON](./experiments/m2_closeout/cpu_benchmark.json)

### M3 从零语言学习（非常早期）

当前实验是**程序化微型语言**：实体、颜色、位置、移动、否定、数量、代词和少量组合规则，使用确定性 oracle 生成答案。

- 约 **279,220 参数**，从随机权重训练 **600 步**，字符级词表 47。
- 当前评测为 **teacher-forced** 的答案字符预测与精确匹配，**不等于自由生成或自然语言理解**。
- holdout 答案 span 精确匹配约 **32.7%**；dev 约 **24.6%**。dev 结果低于其多数答案捷径基线，尚不能认为已经学会通用语言。
- 在该次实验中递归深度 2/4 的精确匹配率低于深度 1；“更多递归可改善推理”仍待验证。

证据：[M3 训练脚本](./experiments/m3_language/run_m3_v1.py) · [M3 结果 JSON](./experiments/m3_language/m3_v1_results.json)

### M5 从零中文语言模型（2026-11-22，早期）

- 自建字符级 tokenizer + NCell/Router/Workspace 自回归语言模型，**452k 参数从随机权重训练**，未导入任何预训练权重。
- 首轮单 seed 200 步：TF 字符准确率 74.41%，自由生成精确匹配（free-gen EM）3.61%，判定 `PIPELINE_READY_LANGUAGE_NOT_VALIDATED`。
- M5.1 诊断裁定瓶颈为**欠训练**（训练集 TF-span-EM 12.93% ≈ sealed 12.55%），排除截断、过拟合与路由死块；修正"从零阶段误用局部更新口径"的设计错误后，stage0 单实体复制 free-gen EM **0.50**（过 EM 门槛，TF 0.5 未达 0.9 整门 FAIL），stage1 两实体绑定 **0.15**（低于 0.25 随机线），未见问法工具选择 **100%（45/45）**。
- 当前自动化测试 **92 项通过**；绑定学习预算（stage0/1 各 600–1000 步）待批准。证据：[M5 报告](./experiments/m5_language/M5_LANGUAGE_REPORT.md) · [M5.1 报告](./experiments/m5_language/M5_1_REPORT.md)

## 快速开始

### 系统要求

- Python **3.11+**；Windows、Linux 等具备 PyTorch 支持的环境。
- 普通 CPU 即可执行核心命令；CUDA GPU **非必需**。
- 安装 PyTorch 时以当前操作系统/硬件适用的官方版本为准。此前本项目在 Windows + i7-14700KF / 32 GB RAM 测试，RTX 5070 12 GB 用于可选 GPU 冒烟实验。

```bash
# 克隆仓库
git clone https://github.com/yuanhong792/n-mind.git
cd n-mind

# 建议创建独立的 Python 虚拟环境
python -m venv .venv

# 激活虚拟环境：Windows PowerShell
# .\.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate

# 开发安装
python -m pip install -e ".[dev]"

# 检查环境；明确使用 CPU / 关闭云端
python -m nmind.cli doctor --device cpu --offline

# 跑自动化测试
python -m pytest tests -q
```

在 PowerShell 中，先激活 `.venv` 再执行下方命令。部分实验脚本使用固定输出路径，会写入或覆盖 `experiments/`、`results/` 中的历史结果；复现前请备份已有文件或使用独立工作区。

### 最小运行流程

```bash
# 训练、保存 checkpoint、重载并比较预测
python -m nmind.cli demo --device cpu --offline --json

# 显式训练小规模实验模型
python -m nmind.cli train --device cpu --offline --epochs 40 --json

# 使用已存在的 checkpoint 评测
python -m nmind.cli evaluate --device cpu --offline --json

# CPU 延迟/吞吐与内存基准
python -m nmind.cli benchmark --device cpu --offline --json

# 本地 rollout（演示交互闭环，不是聊天）
python -m nmind.cli run --device cpu --offline --T 24
```

注意：默认 `demo` 和低训练步数只证明流程正常，不代表达到报告中的最佳成绩。严谨复现应使用对应实验脚本和冻结的训练配置。

### 复现实验

```bash
# M1：过拟合与训练预算检查
python experiments/2026-11-21_m1_overfit.py

# M2：checkpoint 重载复核
python experiments/m2_closeout/reload_eval.py

# M3：字符级微型语言实验（会写入其固定输出目录）
python experiments/m3_language/run_m3_v1.py --steps 600
```

完整实验索引见 [experiments/README.md](./experiments/README.md)，报告与进度见 [STATUS.md](./STATUS.md)。

## 项目目录

```text
n-mind/
├── src/nmind/
│   ├── core/          # N-Cell、编码器、路由器、工作空间、预测器
│   ├── learning/      # 数据、损失、训练、checkpoint、经验重放
│   ├── memory/        # SQLite 情景记忆、检索、巩固
│   ├── evaluation/    # 任务、基线、消融与性能评测
│   ├── language/      # 微型语言、语言接口
│   ├── runtime/       # 引擎、资源预算、硬件检测
│   └── bridge/        # 云端策略、隐私与供应商适配
├── configs/           # CPU / GPU / offline / cloud 示例配置
├── experiments/       # 可追溯实验脚本、报告及 JSON
├── results/           # CLI 生成结果
├── tests/             # 回归测试
├── docs/              # 实际架构、技术决策
├── AGENTS.md          # 开发不可变约束
├── ROADMAP.md         # 里程碑
├── STATUS.md          # 当前状态与已知问题
└── pyproject.toml
```

## 研究路线图

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| **M0** | 环境、项目骨架、测试基础 | ✅ 已完成 |
| **M1** | N-Cell 原型训练、checkpoint 和 CPU 闭环 | ✅ 已完成 |
| **M2** | 稀疏/稠密对照、STE 路由、M2.2 验证 | 🔶 `CONDITIONAL` |
| **M3** | 从零微型语言学习、递归深度与组合泛化 | 🧪 实验进行中 |
| **M4** | 持续学习、局部参数更新、遗忘与记忆对照 | ⬜ 关键实验待完成 |
| **M5** | 更自然的语言及多模态编码研究 | ⬜ 规划中 |
| **M6** | CPU 量化、部署、能耗与低配置优化 | ⬜ 规划中 |
| **长期研究** | 经许可的联网探索、自主神经学习、云端可选增强 | ⬜ 研究目标，未验证 |

里程碑命名曾在早期文档中调整；以 [ROADMAP.md](./ROADMAP.md)、[STATUS.md](./STATUS.md) 和对应实验报告记录的具体代码与证据为准。计划中的功能不等于已实现功能。

## 研究原则

1. **禁止预训练权重替代核心能力**：不导入现有 LLM 权重、教师模型或蒸馏输出来伪装从零学习。
2. **可重复**：保存 seed、数据指纹、配置、Git revision、模型 checkpoint 及环境信息。
3. **公平对照**：相同数据、损失定义、训练预算、CPU 条件下比较稠密网络、GRU、Transformer 和 N-MIND；未实现的基线明确标记。
4. **结果诚实**：准确率和延迟都保留任务定义、样本数、方差和失败案例；不选择单个最佳 seed 作为总体结论。
5. **安全联网**：云端默认关闭；不自动上传私人数据、不把外部网页当作可信指令；本地推理与外部增强必须分开评价。
6. **隔离式持续学习**：候选神经参数更新先验证新旧任务表现，支持回滚；自主在线改权重能力尚未获得系统性验证。

## 参与开发

欢迎提交有可复现证据的 issue、bug 修复、基线实现、CPU 算子优化、训练稳定性实验和完整研究复现。贡献前请阅读 [AGENTS.md](./AGENTS.md)、[架构说明](./docs/architecture.md) 与 [当前进度](./STATUS.md)。

提交新性能结论时，请同时提供训练/评测脚本、配置、数据划分、随机种子、机器环境及可核验的 JSON，不要提交包含密钥或个人数据的记录。大模型权重和生成工件应使用合适的发布渠道，避免不必要的大文件进入 Git 历史。

### 许可证

项目 `pyproject.toml` 当前声明 **MIT**。截至本次仓库核对，尚未看到仓库根目录独立的 `LICENSE` 文本文件；发布或复用代码时请先补齐并确认正式许可文件。

## English summary

**N-MIND (Neural Memory & Incremental Dynamics)**, tentatively named **脉脑** in Chinese, is an experimental, local-first neural architecture research project. It explores **from-scratch learning**, persistent memory, incremental adaptation, sparse state updates, and recursive computation on ordinary CPUs and optional consumer GPUs.

- **Neural:** learned representations and reasoning through trainable neural components.
- **Memory:** working state, episodic storage, retrieval, and experience replay.
- **Incremental:** research into acquiring new skills through experience without rebuilding the entire model for each update.
- **Dynamics:** event-driven state evolution, selective routing, and recurrent computation.

**Project constraints:** no pretrained LLM weights or teacher-model distillation for the core network; CPU-first inference; optional network/cloud enhancement, with offline core functionality retained.

**Current status:** research prototype only. M1 is demonstrated on synthetic grid tasks; M2 sparse-routing evaluation remains conditional; M3 character-level micro-language experiments are preliminary and teacher-forced. There is **no validated general intelligence, open-domain language model, or autonomous self-learning capability yet**.

**Repository:** https://github.com/yuanhong792/n-mind
