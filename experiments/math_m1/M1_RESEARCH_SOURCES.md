# M1 数学研究来源与归属声明

本文件区分：论文已提出的机制、本项目独立实现、本项目新的组合假设、经实验支持的结论。
本轮不复制任何大型网络源码；只依据公开公式做独立的小规模 PyTorch 实现。

## A. Gated Delta Networks

- 来源：Songlin Yang, Jan Kautz, Ali Hatamizadeh. *Gated Delta Networks: Improving Mamba2 with Delta Rule*. arXiv:2412.06464 v3，ICLR 2025 camera ready。
- 链接：https://arxiv.org/abs/2412.06464
- 论文机制：在 DeltaNet 的 delta 更新上加入数据相关标量衰减 \(\alpha_t\)：
  \(S_t = S_{t-1}(\alpha_t(I-\beta_t k_t k_t^\top)) + \beta_t v_t k_t^\top\)，读出 \(o_t=S_t q_t\)。
  论文声称 q/k L2 归一化、输出归一化与门控、chunkwise WY/UT 并行训练，并在长上下文任务中整体优于 Mamba2/DeltaNet，但原文也说明 length extrapolation 结果 mixed。
- 本项目使用：独立实现基础 DeltaRule 与标量/通道衰减版本，用于算子验证和 M1-B 参考。**不复制论文代码。**
- 归属：公式来自论文；PyTorch 小算子、测试与模型融合是本项目独立实现；M1 任务上的效果属于本项目待验证结果，不能宣称论文原创。

## B. Gated DeltaNet-2

- 来源：Ali Hatamizadeh, Yejin Choi, Jan Kautz. *Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention*. arXiv:2605.22791 v2，NeurIPS 2026。
- 论文/官方代码：https://arxiv.org/abs/2605.22791 ；https://github.com/NVlabs/GatedDeltaNet-2
- 官方代码许可证：**NVIDIA Source Code License-NC**（非商业）。本轮**不复制、不移植官方源码**；只依据 README 明示公式做独立实现。
- 论文机制：通道级擦除门 \(b_t\) 与写入门 \(w_t\) 解耦：
  \(S_t=(I-k_t(b_t\odot k_t)^\top)D_tS_{t-1}+k_t(w_t\odot v_t)^\top\)。
  当两门塌缩为同一标量时退化为 KDA，衰减也塌缩时退化为 Gated DeltaNet。README 报告 erase gate 贡献主要增益。
- 本项目使用：实现 M1-C 的独立门控版本；与 M1-B 仅改变门控方式，其余结构和训练配置一致。
- 归属：解耦门机制属于论文；本项目的“在轻量字符级计算任务上是否减少变量干扰”是新的组合验证假设，不等于原创机制。

## C. Kalman Delta Networks

- 来源：Ngoc Bui, Tinglin Huang, Rex Ying. *Kalman Delta Networks: Uncertainty-aware Associative Memory*. arXiv:2609.07816 v2。
- 链接：https://arxiv.org/abs/2609.07816
- 论文机制：把关联记忆写成线性高斯状态空间模型，传播记忆估计与不确定性，用 Kalman gain 控制写入；给出 Diagonal KDN（每 key 通道一个不确定性）与 Isotropic KDN（每 head 一个标量），并说明 Delta rule 是固定增益特例。
- 本项目使用：仅实现对角近似数学算子并通过验证器；**未经前置验证不加入正式主模型**。本轮 M1-B/M1-C 不包含不确定性记忆。
- 归属：Kalman 视角属于论文；本项目仅做独立低维实现和稳定性记录，不宣称数学贡献。

## D. Anatomy of Associative Recall in Fixed-State Recurrences

- 来源：Julian Boesch, Andrew Wee. arXiv:2609.16183 v1。
- 链接：https://arxiv.org/abs/2609.16183 ；结果代码：https://github.com/JIBSIL/dualgoose （本轮不复制）
- 论文发现：在固定状态下单旋钮比较短因果卷积、rank-1 delta vs 对角转移、衰减；卷积是主导因素（K=32 时约 +0.47/+0.44）；rank-1 优势在加卷积后缩到 +0.034；干扰距离存在“wall”，距离课程可把 0.021 提升到 1.000；训练是 seed lock-in lottery，课程提高 lock-in 率。
- 本项目使用：借鉴“距离课程 + 干扰强度 + 多种子”的实验设计；本轮首测不为 M1-B/M1-C 叠加短卷积，避免混淆门控效应。卷积算子只作为独立数学算子验证与后续候选。
- 归属：课程思想来自论文；本项目自己的课程实现、任务数据和多架构对照是独立工作。

## E. AlphaEvolve

- 来源：Google DeepMind. `google-deepmind/alphaevolve`；论文 *AlphaEvolve: A coding agent for scientific and algorithmic discovery*，arXiv:2506.13131。
- 链接：https://github.com/google-deepmind/alphaevolve
- 许可证：软件 Apache-2.0；其他材料 CC-BY。
- 借鉴点：自动生成候选 → 数学/运行时验证 → 自动评价 → 淘汰/选择 → 多目标 Pareto 的闭环思想。
- 本项目使用：**不依赖外部 LLM 生成算法**；只做确定性配置枚举，使用本地数学验证、短时训练、固定面板与 Pareto 选择。不复制 AlphaEvolve 代码。
- 归属：评价/淘汰思想属于已有工作；本项目搜索器是独立实现的有限枚举器。

## 状态标记

- 论文机制：以上 A–E 已逐条标注。
- 本项目独立实现：delta_rule.py、erase_write.py、uncertainty_memory.py、causal_conv.py 及各候选模型。
- 本项目组合假设：矩阵记忆改善绑定；独立双门降低变量干扰；距离课程改善长距离保持。
- 实验支持状态：**PENDING**，待 M1_TRAINING_RESULTS.json / M1_REPORT.md 更新后才可声称获得支持或未获支持。
