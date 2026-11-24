# M2 双记忆神经架构研究来源与归属声明

本文件区分四类：(1) 论文已提出的机制；(2) 本项目独立实现；(3) 本项目新的组合假设；(4) 实验支持状态。
本轮不复制任何大型网络源码；只依据公开公式做独立的小规模 PyTorch 实现。所有引用均为公开学术来源。

## A. Neural Turing Machine (NTM)
- 来源：Alex Graves, Greg Wayne, David Reece. *Neural Turing Machines*. arXiv:1410.5401, 2014。
- 链接：https://arxiv.org/abs/1410.5401
- 论文机制：外部记忆矩阵 + 可微分读写头，content-based addressing 与 location-based addressing 结合，插值门控制。
- 本项目使用：借鉴"可微分读/写指针"与"content + location 混合寻址"思想，实现固定容量序列记忆 SM 的神经写入分布与读取指针。**不复制论文代码。**
- 归属：寻址机制属于论文；本项目的低维 SM 实现、指针稳定化与任务验证是独立工作。

## B. Differentiable Neural Computer (DNC)
- 来源：Alex Graves et al. *Hybrid computing using a neural network with dynamic external memory*. Nature 2016；arXiv:1607.06450。
- 链接：https://arxiv.org/abs/1607.06450
- 论文机制：在 NTM 上增加 free-gate、allocation、temporal linkage，支持动态记忆分配与有序链接读取。
- 本项目使用：借鉴"顺序链接"支撑有序读取的思想；本轮采用更轻量的固定容量 shift-pointer，不实现完整 DNC 分配器。
- 归属：linkage 机制属于论文；本项目简化版本与结论为独立工作。

## C. Neural Stack / Queue
- 来源：Edward Grefenstette, Karl Moritz Hermann, Mustafa Suleyman, Phil Blunsom. *Learning to Transduce with Unbounded Memory*. arXiv:1503.01067, NeurIPS 2015。
- 链接：https://arxiv.org/abs/1503.01067
- 论文机制：可微分栈/队列，连续 push/pop 强度，用于序列转导。
- 本项目使用：借鉴"可学习 push/pop 强度"实现序列记忆的有序写入与连续读取指针。
- 归属：可微栈属于论文；本项目的固定容量实现与稳定性修正为独立工作。

## D. Linear Attention / Transformers are RNNs
- 来源：Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, François Fleuret. *Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention*. arXiv:2006.16236, ICML 2020。
- 链接：https://arxiv.org/abs/2006.16236
- 论文机制：以核特征映射把注意力写成循环状态更新，获得 RNN 式增量推理。
- 本项目使用：作为 R2 tiny Transformer 对照与固定状态关联回忆的理论背景；delta 类算子沿 M1 已有实现。
- 归属：线性注意力属于论文；本项目 tiny Transformer 为独立小规模实现。

## E. Gated Delta Networks / Gated DeltaNet-2 / Kalman Delta Networks
- 来源：见 `../math_m1/M2_RESEARCH_SOURCES.md` 之 M1 归档（arXiv:2412.06464 / 2605.22791 / 2609.07816）。
- 本项目使用：M1-B 共享门 Delta 作为 M2 的 AM 基线（delta_baseline / protected_delta）；独立擦写门与不确定性不作为本轮主模型。
- 归属：门控机制属于论文；本项目保护门与双记忆融合为新的组合假设。

## F. 固定状态关联回忆与课程
- 来源：Julian Boesch, Andrew Wee. *Anatomy of Associative Recall in Fixed-State Recurrences*. arXiv:2609.16183。
- 链接：https://arxiv.org/abs/2609.16183
- 论文发现：短因果卷积是主导因素；距离课程可把困难任务从近 0 提升到 1.0；多种子 lock-in lottery。
- 本项目使用：继承距离课程与多种子复验设计；新增 SEP 边界与序列记忆，专门针对连续生成与重复生成。

## 本项目独立实现清单
- `tokenizer_m2.py`：SEP 显式边界 tokenizer（M2 新增协议）。
- `math_lab/associative_memory.py`：关联记忆 AM 与保护门算子。
- `math_lab/memory_protection.py`：可学习保护门。
- `math_lab/differentiable_pointer.py`：可微分读写指针。
- `math_lab/sequence_addressing.py`：固定容量序列寻址。
- `math_lab/causal_conv.py`：短因果卷积与位置编码。
- `math_lab/dynamics_analysis.py`：状态动力学分析算子。
- `models/`：R1 GRU、R2 tiny Transformer、M2-A/B/C/D/E 六个候选。
- `math_verifier_m2.py`：20 项数学与工程验证。

## 本项目新组合假设（待实验判定）
- H1：显式 SEP 边界能消除前缀监督冲突，提升多字符答案的正确率。
- H2：关联记忆保护门能降低变量重赋值/干扰的破坏。
- H3：固定容量序列记忆 + 可学习指针能实现有序连续生成，缓解重复生成。
- H4：短因果卷积提供局部位置信号，改善字符级拷贝。
- H5：双记忆自适应融合在关联任务与序列任务上分别优于单记忆。

## 实验支持状态
- 状态：**PENDING**，待 `M2_MATH_TESTS.json`、`M2_TRAINING_RESULTS.json`、`M2_REPORT.md` 更新后方可判定。
- 严禁把组合假设写成已获论文支持或本项目已证实；最终以真实训练与自由生成结果为准。
