# M3 数学成果研究来源（P1）

检索日期：2026-11-24。要求：最近 12 个月原始论文/证明/实现，逐项记录来源、日期、验证状态、
定理成立条件、可转化计算机制、与神经网络关联、理论复杂度、CPU 限制、已有技术、是否值得实验。
**不得把未经确认的数学证明声明当正式成果；不得把针对凸函数/对数凹分布/特定代数域的定理未经证明推广到任意神经网络。**

---

## A. 矩阵与张量计算 / 低秩与增量

### A1. Von-Neumann State-Space Transformer (VN-SST)
- 来源：Morteza Sarafyazd. *The Von-Neumann State-Space Transformer for neural decoding*. arXiv:2608.25088v1。
- 日期：提交 2026-08-25，发布 2026-08-27（来源：arxiv 镜像、alphaXiv PDF、paperlayer）。
- 验证状态：预印本（v1），**未经同行评审**；作者报告在 3 个运动皮层解码基准上样本效率优于现代 Transformer，小文本 LM 上参数效率更高。
- 定理成立条件：无普适定理；是架构假设（低维慢状态作指令指针，逐 token 合成 FFN 算子）。作者固定窗口/状态大小/记忆维度，超参未扫。
- 可转化机制：共享基算子 + 少量低秩「指令」张量，控制器从携带状态读出 code，逐 token 合成权重矩阵；不是混合专家输出，而是合成算子本身。
- 与神经网络关联：低秩状态条件化计算、按 token 生成权重（fast weights），与本轮 P3 低秩更新、P4 按需计算直接相关。
- 理论复杂度：论文未给统一 MAC 公式；低秩指令增量近似 O(r·d) 量级（本项目估算，非论文结论）。
- CPU 限制：状态空间记忆 + 低秩 bank 的 Python/逐 token 开销可能抵消理论收益；需实测。
- 已有技术：低秩适配（LoRA, Hu et al. 2021）、fast weights、MoE、SSM。
- 是否实验：**是**——作为 M3-A 低秩状态更新与 M3-B 指令式按需计算的先验依据；本项目独立低维实现，不复制代码。

### A2. 低秩适配与增量矩阵更新（背景）
- 来源：Hu et al. *LoRA: Low-Rank Adaptation of Large Language Models*, ICLR 2022 (arXiv:2106.09685)。
- 状态：已发表、被广泛复现。
- 定理条件：冻结基权重、低秩增量 B·A，秩 r≪d；表达力受秩限制，无普适等价保证。
- 关联：本项目 P3 的低秩状态更新 `h_new = h + U(Vh)` 与 LoRA 同构思想，但作用在循环状态而非权重；**不宣称原创**。
- 是否实验：是（作为 M3-A 机制来源，须在报告中标注已有基础）。

---

## B. 高维几何与优化

### B1. KLS 猜想被证明（重要，但本轮不进主模型）
- 来源：Zhao Song & Xinzhi Zhang. *An O(1) Bound for the KLS Constant*. arXiv:2610.01447v2（2026-10-01 提交，10-04 修订）。
  独立并行证明：Bizeul, Klartag, Lehec. *Presenting a Proof of the KLS Conjecture*. arXiv:2610.05474（2026-10-04）。
  确定性变分证明：Balasubramanian & Kasiviswanathan. arXiv:2610.07728v2（2026-10-06 提交，10-08 修订，明确常数 25）。
  Lean 4 形式化：Zhao, *Dimension-Free KLS Bounds in Lean*（Zenodo, 2026-10-20）。
- 日期：2026-10（来源：Microsoft Research 官方博客 2026-10-21、MathWorld、unite.ai、functor.network）。
- 验证状态：**多项独立证明 + 机器形式化**，属于本轮最可靠的数学成果。
- 定理成立条件：**仅对各向同性对数凹概率测度**（isotropic log-concave）成立：Cheeger/Poincaré 常数有维数无关上界。
  **严禁未经证明推广到任意神经网络损失面或一般分布。**
- 可转化机制：理论上支撑高维采样/随机化算法的混合时间界、集中不等式；可作为「低维结构是否足够」的理论直觉来源。
- 与神经网络关联：间接（优化/采样理论背景），**不能作为本轮任何神经架构数学正确性的依据**。
- 理论复杂度：改进随机采样混合时间上界（维数无关常数）。
- CPU 限制：定理是存在性/上界，不直接给可执行算法收益。
- 已有技术：随机局部化（stochastic localization, Eldan 2013）、薄壳猜测等。
- 是否实验：**否**——仅归档为理论背景，写入 M3_RESEARCH_SOURCES.md，不转化为神经算子（符合 P1 纪律）。

---

## C. 信息论与最小充分统计量

### C1. DySIB: Dynamical Symmetric Information Bottleneck
- 来源：K. Michael Martini, Eslam Abdelaleem, Paarth Gulati, Ilya Nemenman. *Information bottleneck for learning the phase space of dynamics from high-dimensional experimental data*. arXiv:2604.24662v2。
- 日期：2026-04-27 首次发布（来源：arxiv HTML v2、scifaro、Google Scholar）。
- 验证状态：预印本；在已知真值的实验摆视频数据上恢复 2 维相空间（角度/角速度），与真值维度、拓扑、几何对齐。
- 定理成立条件：方法基于变分 IB；依赖「预测互信息捕捉本质动力状态变量」的领域假设；超参由数据自洽设定。
- 可转化机制：对称 IB 目标 `L = I(X;Zx)+I(Y;Zy)-β I(Zx;Zy)`，只压潜空间不做重构；用预测互信息 + 复杂度惩罚学低维表示。
- 与神经网络关联：P2 最小充分神经状态。互信息精确估计困难，DySIB 用变分界——**本项目不冒充精确互信息**，只用可观测代理（有效秩/奇异值/压缩能力保持率/分量消融）。
- 理论复杂度：变分估计，取决于编码器/解码器大小。
- CPU 限制：互信息变分估计需额外网络与采样，CPU 开销不可忽略。
- 已有技术：信息瓶颈（Tishby et al. 2000）、变分 IB（Alemi et al. 2017）。
- 是否实验：部分——借鉴「压缩 + 预测充分性」思想，实现 information_bottleneck.py 的可观测代理指标；不实现变分互信息训练。

---

## D. 神经动力系统（算子级，参考 M1/M2 已归档 + 本轮新增）

### D1. 状态空间模型与谱半径/稳定性
- 来源（背景经典，非近 12 个月）：Gu & Dao. *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*, arXiv:2312.00752 (2023)；
  Hochreiter & Schmidhuber. *Long Short-Term Memory* (1997) 的门控稳定性分析。
- 近期进展：本轮检索到 VN-SST (2608.25088) 使用 selective-SSM 作为慢状态通道。
- 状态：经典已发表、Mamba 已被广泛复现；近期 SSM 变体多为预印本。
- 可转化机制：谱半径 <1 保障稳定；对角状态动力学 `h_new = a⊙h + ...`；增量/事件触发计算。
- CPU 限制：稳定性分析本身低成本，可进算子验证。
- 是否实验：是——state_stability.py / diagonal_dynamics.py 做算子级验证。

---

## E. 数学公式自动发现

### E1. AlphaEvolve
- 来源：Google DeepMind, *AlphaEvolve: A coding agent for scientific and algorithmic discovery*, arXiv:2506.13131 (2025)；仓库 google-deepmind/alphaevolve（软件 Apache-2.0，其他材料 CC-BY）。
- 状态：已发布，报告在矩阵乘法、调度、数学开放问题上的结果。
- 关联：评价-淘汰-搜索闭环思想。**本项目不依赖 LLM 生成答案**，只做确定性受控表达式枚举。
- 是否实验：是（思想层面）。

### E2. PySR / DSR / uDSR
- 来源：Cranmer, *PySR: Fast & Parallel Symbolic Regression*, (2023, arXiv:2308.09864 系列)；
  Petersen et al. *Deep Symbolic Regression* (2021)；Landajuela et al. *uDSR* (2022)。
- 状态：已发表/开源，符号回归主流基线。
- 可转化机制：表达式树搜索、遗传编程、神经引导搜索、稀疏动力学识别（SINDy）。
- 是否实验：是——formula_lab 的表达式树/受控变异/等价简化参照此思想，独立实现。

### E3. LLM 驱动符号回归与近期进化系统
- 来源：LLM-SR (Shojaee et al., 2025, arXiv:2405.02011 系列)；GAE (Chen & Cheng, *Graph-Augmented Evolution for Scientific Discovery*, arXiv:2607.10127, 2026-07-11)；
  Coupled Harness–Model Evolution for Scientific Equation Discovery (arXiv:2610.17740, 2026-10)；
  PiT-PO (arXiv:2602.10576, 2026-02-11)；Programmatic Context Augmentation for LLM-based SR (arXiv:2605.03101, 2026-05-04)。
- 状态：均为 2026 年预印本/会议投稿，**未经充分验证**；多为 LLM 作为变异算子。
- 关联：失败诊断、接受检查、外推记忆化检查等搜索纪律值得借鉴。
- 是否实验：部分——借鉴「受控搜索 + 严格淘汰 + 外推检查」纪律；不使用 LLM。

---

## 状态标记与纪律声明
- 可靠度分级：KLS 证明（多独立证明 + 形式化）> 已发表经典（LoRA/PySR/LoRA/IB/Mamba）> 近期预印本（VN-SST/GAE/DySIB 等）。
- 本轮只把「可通过可执行算子 + 数学验证 + CPU 实测」的成果转入模型实验。
- KLS 仅作理论背景，**不进入任何神经算子**。
- 低秩分解、SSM、IB 均属已有研究基础，**本项目的新组合假设**（低秩对角状态更新、自适应最小计算路径、受控公式搜索用于神经状态更新）单独在报告中标注，不宣称原创机制。
