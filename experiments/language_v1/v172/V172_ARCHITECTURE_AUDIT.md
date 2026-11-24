# V1.7.2 架构审计（P0 / P2）

> 状态：**架构级诊断完成**，父模型权重级验证 `NOT_YET_VERIFIED`。
> 本轮 `main_model_optimizer_steps = 0`，Stable 只读，未执行正式训练。

## 0. 环境偏差（必须先说明）

| 任务书 | 本机实际 |
|---|---|
| `D:\脉脑\cpu本地模型开发\cpu本地模型开发\n-mind` | `D:\cpu本地模型开发\n-mind` |
| `.venv-gate24\Scripts\python.exe` | 不存在；用系统 Python 3.12.10 + torch 2.10.0+cpu |
| CUDA 可用 | `torch.cuda.is_available() = False` |

父 checkpoint `experiments/language_v1/v15/candidate_v15_baseline.pt` **本地不存在**。
原因：`.gitignore` 含 `*.pt`，权重从未提交；`V171_FAIR_COMPARISON.json` 记录
`LOST_weights_equal_parent`。因此无法读取"实际 V1.5 父模型 Checkpoint 配置"，
本审计改用**历史参数量反推 + 源码实现**的方法，并明确区分证据等级。

## 1. 配置反推（关键结论）

`V15_PARAMETER_AUDIT.json` 记录 `parent_params = baseline_params = 279363`。
按 `NMindLM` 实际层结构逐项计算（`load_bias` 是 buffer，不计入参数）：

```
tok = V*D ; ncell = N*(2H*(D+H+W)+2H)
router = (D+W+N+2)*h + h + h*N + N
ws = S*W + (D*W+W) + 3*W^2 + (W+1)
b2ws = H*W+W ; head = H*V+V
```

枚举后与 279363 **精确匹配**的配置（候选 2 个，仅 `top_k` 不同，因为 top_k 不影响参数量）：

| 参数 | 值 |
|---|---|
| vocab_size | 4096 |
| d_model | 32 |
| n_blocks | **2** |
| block_hidden | **32** |
| workspace_dim | 16 |
| workspace_slots | **2** |
| top_k | **2**（按任务书指明值与源码默认；参数量本身不可辨识） |

任务书怀疑的 `n_blocks=2 / block_hidden=32 / workspace_slots=2` 得到确认。
`top_k` 取值标注为 `TOP_K_NOT_IDENTIFIABLE_FROM_PARAM_COUNT__USED_TASKBOOK_AND_SOURCE_DEFAULT_2`。

## 2. Router 六项冻结诊断（同配置随机权重，架构级）

| 检查项 | 结果 | 判定 |
|---|---|---|
| 评分随 token 变化 | std = 0.1238 | 评分**会变** |
| 选中块集合变化 | 恒为 `{0,1}`（顺序可能互换） | **无稀疏选择** |
| 激活率 | `top_k/n_blocks = 2/2 = 1.0` | **每步全选** |
| 修改 Router 评分改变前向 | logits 平均绝对差 = 0.0 | **不改变计算路径** |
| 交换选中块顺序改变 Workspace 写入 | 差 = 0.0 | 写入对 payload 顺序**置换不变**（数学性质，非假阴性；已用非零 h 复测） |
| Router 梯度非零 | 有 | 梯度通，但无选择空间可作用 |
| 临时副本一步更新后路由变化 | 否 | 选中集合恒为全集 |

**根因**：`top_k == n_blocks` 时，无论评分怎么变，`topk` 返回的都是全部块；
`NCell.forward_sparse` 对每块独立更新，顺序无关；Workspace 写入对 payload 顺序求和不变。
因此 Router 评分虽然变化且能收到梯度，却**不携带任何有效计算路径选择信息**。
V1.6 修复的 STE 梯度断路是真实 bug，但修好后在本配置下仍无可用选择空间。

## 3. Workspace（P2）

`neural_dialogue.py` 中：

```python
wr = self.workspace.write(ws, payload)
ws = {"content": ..., "occupied": ..., "utility": ...}
```

`wr.overflow` **被丢弃**，语言模型没有消费溢出记录，也没有情景记忆落地路径。

| 干扰 token | 溢出事件数 |
|---:|---:|
| 0 | 60 |
| 8 | 164 |
| 16 | 268 |
| 32 | 476 |
| 64 | 892 |

每步写入 `top_k=2` 条 payload，而 `slots=2 / capacity=1.5`，因此**几乎每步都溢出**。
溢出后原事实信息在 Workspace 内不可恢复；N-Cell 的 `h` 仍保留历史输入，但其可解码性
`NOT_YET_VERIFIED`（P1 线性探针 50%，未发现线性可区分信号）。

**不能仅凭容量小认定它是实体绑定失败的根因**：即使扩容，若读出仍是
`h.mean(dim=1)` 且 Router 无选择空间，查询信息仍可能无法转化为答案 token 概率差。

## 4. 证据等级

- `CONFIRMED_LIMITATION`：`top_k == n_blocks`，无稀疏选择空间；Workspace overflow 丢弃。
- `CONFIRMED_BUG`（源码层）：`wr.overflow` 未被消费；V1.6 Router 梯度漏传（已修复）。
- `HYPOTHESIS`：稀疏 Router（`top_k < n_blocks`）能带来有效计算路径差异。
- `NOT_YET_VERIFIED`：真实父权重上的路由行为、N-Cell 可解码性、修复后的能力提升。

原始数据见 `V172_ROUTER_PROBE.json`、`V172_WORKSPACE_PROBE.json`。
