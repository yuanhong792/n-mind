# N-MIND V1.7.2 根因报告

> 本轮：`main_model_optimizer_steps = 0`；未执行正式训练；未覆盖 Stable；未删除历史记录；
> 未上传模型权重；未使用外部 LLM 代替 N-MIND 生成答案。
> 分支：`experiment/language-v172-core-repair`
>
> **最重要的前置事实**：任务书指定的父模型权重 `candidate_v15_baseline.pt` 本地不存在
> （`*.pt` 被 `.gitignore` 忽略，历史记录有 `LOST_weights_equal_parent`），
> 因此所有涉及"真实父权重行为"的结论均为 `NOT_YET_VERIFIED`。
> 本轮用**历史参数量精确反推的真实架构配置 + 同配置随机权重**做架构级冻结诊断，
> 这类结论标记为 `ARCHITECTURE_LEVEL`，不冒充父模型实测。

---

## 最终九问

### 1. Router 在当前配置下是否真的具有稀疏选择能力？

**没有。** `CONFIRMED_LIMITATION`（架构级）

- 由历史参数量 279363 精确反推：`n_blocks=2, top_k=2, d_model=32, block_hidden=32,
  workspace_dim=16, workspace_slots=2`。`top_k == n_blocks`，激活率 `2/2 = 1.0`。
- 评分随 token 变化（std 0.1238），但选中块集合恒为 `{0,1}`。
- 修改 Router 评分后前向 logits 平均绝对差 **0.0**；交换选中块顺序后 Workspace 写入差 **0.0**
  （写入对 payload 顺序置换不变，已用非零状态复测，不是全零假阴性）。
- Router 梯度非零、临时副本一步更新后路由集合仍不变。

结论：V1.6 修复的"Router STE 梯度漏传"是真 bug 且已修，但在本配置下修好后**仍无选择空间可作用**。
这是"参数变了但计算路径没变"的确定性解释。`V172_ROUTER_PROBE.json`

### 2. N-Cell 能否区分不同人物及其属性？

**当前架构与权重下不能。** `CONFIRMED_LIMITATION`（架构级，父权重待验）

- 案例 A/B/C × 干扰 0/8/16/32/64，共 15 组：问小明 vs 问小红的
  `h_blocks`、`h_mean(dim=1)`、`ws_read`、`logits` 四层余弦相似度全部 ≈ **0.999999**。
- 独立闭式 ridge 线性探针（33 参数、0 优化步）准确率 **50%**（随机水平）。
- 正确/错误答案条件概率差接近 0。

注意：`h.mean(dim=1)` 平均的是**块维度**，不是时间维度；本探针同时保留了原始各 Block
状态，原始块状态同样不可区分，所以不是"平均抹掉了位置信息"这一单一原因。
`V172_NCELL_MEMORY_PROBE.json`

### 3. Workspace 经过多少 Token 后开始明显丢失信息？

**从第一步就丢。** `CONFIRMED_LIMITATION` + `CONFIRMED_BUG`

- `wr.overflow` 在 `neural_dialogue.py` 中被直接丢弃，无消费、无恢复路径。
- 溢出事件：0 干扰 60 次，8→164，16→268，32→476，64→892。
- 原因：每步写 `top_k=2` 条 payload，`slots=2 / capacity=1.5`，必然持续溢出。
- N-Cell `h` 仍携带历史，但线性可解码性未证实（见第 2 问）。

**不认定 Workspace 容量小是绑定失败的唯一根因**：读出层与 Router 无选择空间同样致命。
`V172_WORKSPACE_PROBE.json`

### 4. 训练和推理 Token 不一致影响了多少样本？

**实际 0 条 / 7521 条。** `CONFIRMED_BUG`（代码协议层），实际影响 `NOT_AFFECTED_IN_CURRENT_DATA`

- 旧：`tok.encode(prompt + answer, add_eos=True)`；新：`prompt_ids + answer_ids + [EOS]`。
- 扫描 v171 的 train/val/sealed/counter 全部 7521 条，两种编码**逐字节一致**，因为当前模板
  边界（`答：` 后接颜色词）不在 BPE merge 表中。
- 但风险真实存在：任何新句型若在边界处命中 merge，就会产生训练/推理序列漂移。
  静态扫描已定位仍使用旧拼接的调用点。
- 新协议已固化为唯一入口 `v172_protocol.encode_autoregressive`，并有测试钉死。

Tokenizer 词表与已有 checkpoint 未改动。`V172_BPE_ALIGNMENT_AUDIT.json`

### 5. 当前训练监督是否真正作用于关键答案 Token？

**部分作用，但关键判别位停在随机水平。** `CONFIRMED_BUG` + `CONFIRMED_LIMITATION`

- V1.6 逐 token CE 证据：判别首字 ≈ **0.703**（ln2≈0.693，随机水平），
  后缀"色" ≈ 0.017，EOS ≈ 0.009；低 Loss 是恒定后缀稀释出来的假象。
- V1.7 的 `WEIGHT_DISCRIM=2.0` 确实打在答案区判别位上，但快速评测
  `first_token_lp_all` 对**判别位 > 0 的 2 个颜色候选**错误复用了 prompt 末尾 logits。
- 已修复：非首位判别 token 必须先逐 token 前馈前缀再评分；
  `DISCRIMINATIVE_TOKEN_PROXY` 与 `FULL_ANSWER_CONDITIONAL` 严格分离，
  完整答案评分包含 EOS；Teacher-forced 概率不再冒充自由生成成绩。
- 最佳 checkpoint 选择改为固定 VAL 指标，SEALED 只评一次，父/候选/最终统一分母。

`V172_EVALUATION_FIX.json`

### 6. 哪些确定性工程缺陷已经修复？

`CONFIRMED_BUG → FIXED`（本轮，均有测试）：

| 缺陷 | 修复 |
|---|---|
| BPE 训练/推理拼接口径不统一 | `encode_autoregressive` 唯一协议 + 边界/掩码/EOS/拼接测试 |
| 判别位非首位却复用 prompt logits | `discriminative_proxy` 逐 token 前馈前缀 |
| `evaluate()` 内 `model.eval()` 不恢复 | `preserved_mode` 恢复 train/eval、router 统计开关与 buffer |
| Router 负载统计污染验证 | 同一 context manager 快照/恢复 `load_bias`、`block_load_count` |
| 样本间递归状态泄漏 | `fresh_state` 每样本显式新建状态 |
| V1.7 脚本读旧数据 | `train_v172.load_split` 只读 v172/data，缺失即报错，禁止回退 |
| A_dialogue 20/600 | 5 个真实句型族 × 16 上下文变体补足 600/100/100，非复制/标点噪声 |
| Checkpoint 安全 | 父 SHA、optimizer.step、分阶段、Final 先于 Best、原子保存+逐张量回读、RNG/优化器状态、墙钟硬停 |

`tests/test_v172_protocol.py` **10 项全绿**。

### 7. 训练数据是否已正确接入？

**是。** `CONFIRMED`

- `train_v172.py` 固定读取 `experiments/language_v1/v172/data/v172_*.jsonl`；
  实际计数 train 5998 / val 998 / sealed 1000 / counter 300。
- A_dialogue 已达标 600 / 100 / 100；B/D/E 全部达标。
- v171 原始数据未改动，v172 写入独立目录，SHA 冻结在 `V172_DATA_AUDIT.json`。
- 完全相同 prompt 跨集合零重合；反事实保持真实成对（orig/swap）。
- **缺口如实保留**：C_counter train 缺 2、val 缺 2（原始池 1198/198），未伪造补齐。

### 8. V1.8 训练程序是否具备执行条件？

**程序具备，执行被两个条件阻塞。** `NOT_YET_VERIFIED`

1. 父权重文件缺失，SHA `5681f656...` 无法校验 —— 必须先恢复权重。
2. 本轮未获新优化器更新授权 —— 必须单独授权。

已交付 `V18_TRAINING_PLAN.json` + `V18_EXPERIMENT_ARGV.json` + `train_v172.py`：
实验 A（原架构 + 修复协议，279363 参数，checkpoint 兼容）、
实验 B（`n_blocks=4/top_k=2`，289797 参数，不兼容）、
实验 C（在 B 上仅改 `query_readout`，292389 参数，不兼容）。
三组共享种子 20261124、数据、96 步预算、评测协议；参数量按实际配置计算，非猜测。

### 9. 下一轮还存在哪些不能忽略的架构风险？

1. **稀疏性缺失**：`top_k == n_blocks`，Router 无选择空间；实验 B 必须验证这是不是绑定失败的结构前提。
2. **读出不敏感**：`h.mean(dim=1)` 使查询人物信息无法形成可区分 logits；query-readout 历史零步实验在 Router 无梯度条件下做过，需在修复协议后重测（`HYPOTHESIS`）。
3. **Workspace 溢出无落地**：持续溢出且被丢弃；扩容单独实验，禁止与 Router 改动同时上。
4. **父权重不可复现**：权重丢失导致所有历史结论无法在父模型上复核（`NOT_YET_VERIFIED`）。
5. **CPU-only 环境**：本机 torch 2.10.0+cpu，CUDA 不可用；预算需按 CPU 计。
6. **既有 bot 测试缺陷**：2 项失败（sqlite 并发无锁、时序相关）与本次修复无关，但回归基线不干净。

---

## 状态汇总

| 标记 | 项目 |
|---|---|
| `CONFIRMED_BUG` | Router 梯度漏传（V1.6 已修）、BPE 协议不统一、非首位判别评分错误、eval 模式不恢复、overflow 丢弃 |
| `CONFIRMED_LIMITATION` | 无稀疏选择空间、查询表示不可区分、Workspace 持续溢出、父权重缺失 |
| `HYPOTHESIS` | 稀疏 Router 带来有效路径差异；query-readout 改善读出；Workspace 扩容改善保持 |
| `NOT_YET_VERIFIED` | 真实父权重行为、修复后能力提升、V1.8 三组结果、N-Cell 非线性可解码性 |

## 本轮未做（如实记录）

- 未执行任何训练 / optimizer.step = 0；未重跑 1536 步；未启动 V1.8。
- 未覆盖 Stable，未删除历史失败记录，未上传任何模型权重。
- 未修改冻结 tokenizer 词表与已有 checkpoint。
- 全量测试仍有 2 项既有 bot 失败 + 10 项因父权重缺失导致的 setup ERROR，未掩盖、未伪造通过。
