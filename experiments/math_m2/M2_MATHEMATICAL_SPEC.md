# M2 双记忆神经架构数学规格

## 0. 记号
- 隐藏状态 h_t ∈ R^d；关联记忆 S_t ∈ R^{d_v×d_k}；序列记忆 H_t ∈ R^{L×d_seq}。
- 读/写指针 p^w_t, p^r_t ∈ Δ^L（L 维单纯形）；保护门 g_t ∈ [0,1]。
- 全部门由当前隐藏状态经线性投影 + sigmoid/softmax 产生，随机初始化，端到端可微。

## 1. SEP 边界协议（P0）
序列 = prompt_ids + [SEP] + answer_ids + [EOS]；推理输入 = prompt_ids + [SEP]。
SEP=101 不在 prompt 字符表(4..100)内，故答案上下文 prompt+SEP 不可能与任何
prompt 前缀重合。审计实证：历史 39922 样本中 legacy 协议 1663 个 prompt(4.57%)
存在前缀冲突，SEP 后结构性归零（见 M2_PREFIX_CONFLICT_AUDIT.json）。

## 2. 关联记忆 AM（P1）
```
decayed   = α_t · S_{t-1}                         # 标量或逐 key 衰减
pred      = decayed · k_t                          # [d_v]，基于衰减后状态
candidate = decayed + β_t · (v_t − pred) ⊗ k_t    # delta 写入候选
S_t       = S_{t-1} + g_t · (candidate − S_{t-1})  # 保护门插值
read_am   = S_t · q_t
```
- g_t=1 退化为标准 delta；g_t=0 完全保护旧记忆。
- 保护门 g_t = sigmoid(W_g·h_t + b_g)，模型自主学习「干扰写入时关闭、目标写入时打开」。

## 3. 序列记忆 SM（P2）
```
p^w_t = PointerCell(M=H_{t-1}, p^w_{t-1}, h_t)     # content + shift + interp
H_t   = H_{t-1}·(1 − p^w_t⊗e_t) + p^w_t ⊗ v_t      # 加权写入
p^r_{t+1} = (1−s_t)·p^r_t + s_t·roll(p^r_t, +1)     # 有序推进一位
read_sm = p^r_t · H_t
```
- 固定容量 L；写入用内容+位置混合寻址，读取指针随生成步向后推进实现有序连续输出。
- roll(+1) 使新 p[j]=旧 p[j−1]，即指针槽 0→1→2 前进（验证器 t14 固定方向）。

## 4. 短因果卷积与位置编码（P3）
- 深度可分离因果卷积：左填充 k−1，位置 t 只依赖 ≤t；核 2/4/8。
- 正弦位置编码 PE(L,d)，仅验证通过；是否进入主模型由消融决定。

## 5. 双记忆自适应融合（P4）
```
r_t = γ_t · Proj_am(read_am_t) + (1−γ_t) · Proj_sm(read_sm_t)
γ_t = sigmoid(W_γ·h_t + b_γ)
```
- γ_t 由门控网络学习；消融时强制 γ=1（仅 AM）或 γ=0（仅 SM）验证功能分工。

## 6. 动力学稳定性（P5）
- 监控 state_norm、step_distance、cos_adjacent、token_entropy、fixed_point_ratio。
- 重复生成假设：read_sm 与 h 陷入不动点（cos_adjacent→1、fixed_point_ratio↑）导致同一 token 反复输出。
- 对策：读取指针强制推进（s_t 下限）、写入 erase 门、EOS 权重。

## 7. 候选架构（P6）
| 代号 | 结构 | 关键能力 |
|---|---|---|
| R1 | GRU 参考 | 基线对照（无显式记忆） |
| R2 | tiny Transformer | 自注意力对照 |
| M2-A | GRU + AM（共享门 delta） | 关联记忆 |
| M2-B | GRU + AM(保护门) | 抗干扰关联记忆 |
| M2-C | GRU + SM（序列记忆+指针） | 有序连续读取 |
| M2-D | GRU + AM(保护门) + SM | 双记忆独立 |
| M2-E | GRU + AM + SM + 双记忆融合门 + conv | 完整模型 |

## 8. 训练监督
- 序列 = prompt + SEP + answer + EOS；loss 只在 answer+EOS 位置。
- 权重：首 answer token 2.0，其余 answer 1.0，EOS 0.25；按样本归一化。
- 主指标：自然 EOS 的自由生成逐字符精确匹配；full_answer_lp 不含 prompt/SEP 概率。

## 9. 验证状态
- 20/20 数学验证 PASS（M2_MATH_TESTS.json）。
- 修复记录：erase 广播维度、指针推进方向（真实 bug，未通过放松测试掩盖）。
- 组合假设 H1–H5 的实验支持状态：PENDING，以训练结果为准。
