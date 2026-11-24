# M4 数学与算法研究成果来源（P1）

检索日期：2026-11-24。范围：最近 12 个月（2025-11 至 2026-11）优先，
经典成果作为背景单列。逐项记录来源、日期、验证状态、定理条件、可执行公式、
理论复杂度、CPU 限制、已有技术、对 N-MIND 的直接价值、可构造对照实验。

**纪律**：不把数学证明声明当已确认突破；不因渐近复杂度更低就认定适合 32/64 维矩阵；
不把已有公式重组未经技术检索就宣称原创。

---

## A. 继承 M3 已核实条目（原始来源见 `math_m3/M3_RESEARCH_SOURCES.md`）

### A1. VN-SST（arXiv:2608.25088v1，2026-08）
- 状态：预印本，未经同行评审；作者报告运动皮层解码与小文本 LM 的样本/参数效率提升。
- 条件：无普适定理，属架构假设；低维慢状态作指令指针、逐 token 合成 FFN 算子。
- M4 价值：作为低秩状态条件化与 fast-weights 的先验；**不复制代码**。
- 对照实验：M4-A 低秩 Q/K 交互 vs R2，固定 FFN/Value 维度。

### A2. KLS 常数 O(1) 界（arXiv:2610.01447v2 等，2026-10）
- 状态：多独立证明 + Lean 4 形式化，可靠度最高。
- 条件：**仅各向同性对数凹测度**；严禁推广到神经网络损失面。
- M4 价值：仅理论背景，**不进入任何算子**。

### A3. DySIB 动态对称信息瓶颈（arXiv:2604.24662v2，2026-04）
- 状态：预印本，在已知真值摆视频数据上恢复 2 维相空间。
- 条件：依赖"预测互信息捕捉本质状态"的领域假设；变分界非精确互信息。
- M4 价值：P6 只用可观测代理（有效秩、重构误差、关键事实读取误差），不冒充互信息。

### A4. AlphaEvolve（arXiv:2506.13131，2025）/ PySR（2023）/ DSR（2021）
- 状态：已发布/开源；评价-淘汰-搜索与表达式树思想是 P8 的方法来源。
- M4 价值：确定性受控枚举，不使用 LLM 生成公式，不宣称原创。

### A5. LoRA（ICLR 2022）/ Mamba（2023）/ FlashAttention（2022）/ MQA-GQA（经典）
- 状态：已发表并被广泛复现，属**已有技术**。
- M4 价值：P3 的 Q/K 低秩降维、P5 的递推状态、P7 的精确优化必须在报告中标注来源，
  不得作为本项目原创。

---

## B. 本轮新增（已用 arXiv 原文核实）

### B1. HOLA: A Hippocampus for Linear Attention（arXiv:2607.02303v1）
- 来源：Wanyun Cui. *A Hippocampus for Linear Attention: An Exact Memory for What the Recurrent State Forgets*。
  提交 2026-07-02（来源：arXiv abs 页，本轮 WebFetch 核实）。
- 验证状态：预印本；作者报告 340M 参数、15B SlimPajama token 训练的实验，
  Wikitext PPL 27.32→22.92、LAMBADA 30.95→30.26，RULER 32k 检索鲁棒性优于 GDN。
  **本轮未复现这些数字，仅作为来源记录。**
- 数学条件：线性注意力/SSM 把前缀压成固定状态，是有损压缩；多个 KV 关联竞争时早期事实被覆盖。
  HOLA 保留 delta-rule 压缩状态，另加有界精确 KV cache，按 `beta * ||e||`（已写入状态的预测残差）保留 token。
- 可执行机制：压缩状态 + 有界精确缓存的双通道读取；无学习式 eviction。
- 理论复杂度：论文未给统一 MAC 公式；缓存读取成本随缓存大小线性增长（本项目判断，非论文结论）。
- CPU 限制：额外缓存的写入/读取与内存访问在 CPU 上可能抵消线性注意力收益；需实测。
- 已有技术：Complementary Learning Systems、delta rule、线性注意力、KV cache。
- **对 M4 的直接价值**：M1–M3 的固定状态记忆失败与本文描述的"覆盖导致 needle recall 退化"同构。
  本轮 P5 只做增量核算子研究，**不默认把缓存机制加入主模型**；若要加入需单独预登记。
- 可构造对照：固定状态 vs 固定状态+有界精确缓存，在 M4 第三层"历史与当前状态"任务上比较。

### B2. V-CoLA: Vision Token Compression with Linear Attention（arXiv:2610.11251v2）
- 来源：Hao Jiang 等（12 位作者），提交 2026-10-08，v2 2026-10-20；**Accepted to EMNLP 2026 main conference**
  （来源：arXiv abs 页，本轮 WebFetch 核实）。代码仓库 `github.com/alibaba/EfficientRL`。
- 验证状态：会议接收；作者报告 50% vision token 保留 99.5% 性能、12.5% 保留 >88%，
  prefill 加速 1.86×–6.15×。**本项目未复现，且任务域不同（视觉 token 压缩）。**
- 条件：针对 hybrid VLM（文中提 Qwen3.5）的线性注意力分块并行实现；
  为线性注意力定制的独特性感知重要性准则 + 自适应 token 合并。
- 对 M4 价值：支持 P4 "先粗粒度选择、再精确计算"的思路；但视觉 token 稀疏性假设不能直接搬到符号任务。
- 可构造对照：可学习分组摘要 vs 固定局部窗口（P4.5），必须在同预算下比较。

### B3. 其他检索到的条目（未核实原文，仅登记，不作为实验依据）
- Orthros（NeurIPS 2026，长上下文注意力加速）：仅见二手报道，未取得原文 → `NOT_VERIFIED`，不转化为算子。
- River-LLM（ACL 2026，KV 共享退出加速）：仅见二手报道 → `NOT_VERIFIED`。
- LightRetriever（ICLR 2026）：仅见二手报道 → `NOT_VERIFIED`。

---

## C. 与 M4 各节的对应关系

| M4 章节 | 来源 | 原创性声明 |
|---|---|---|
| P3 LRII 低秩 Q/K | A5 LoRA + A1 VN-SST | 组合假设，非原创机制 |
| P4 SII 分组选择 | B2 V-CoLA + A5 动态稀疏注意力 | 借鉴思想，独立实现 |
| P5 III 增量核 | A5 线性注意力 + B1 HOLA | 明确属已有方向，不宣称等价 softmax |
| P6 正交压缩 | A3 DySIB 代理指标 | 只用可观测代理 |
| P7 等价 head 优化 | 数学等价变换，无文献依赖 | 工程等价优化，不宣称算法创新 |
| P8 公式搜索 | A4 AlphaEvolve / PySR | 确定性枚举，不用 LLM |

---

## D. 状态标记

- 已核实原文：B1、B2（本轮 WebFetch arXiv abs 页）。
- 继承已核实：A1–A5（M3 轮检索记录）。
- 未核实：B3 三项，标 `NOT_VERIFIED`，不进实验。
- 所有进入算子的机制都必须在 `M4_MATHEMATICAL_SPEC.md` 中区分：
  **论文机制 / 本项目实现 / 组合假设 / 实验支持**。
