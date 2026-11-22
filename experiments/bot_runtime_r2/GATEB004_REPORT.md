# GATEB004 REPORT — 持续学习机制诊断与因果定位（2026-11-22，无训练）

> 前置：Gate-B 001/002/003 三轮候选训练全部 REJECTED（B 泛化增益全 0，A 遗忘 -31.25/-100/-82.5pp）。
> 本轮零 optimizer.step：对 stable 与三个候选做隐藏状态探针、线性可解码性探针、Workspace 漂移
> 分析与受控模块替换/激活替换干预。全部结论绑定 JSON 证据：`GATEB004_DIAGNOSTICS.json`。

## 一、最终因果答案

**N-MIND 的旧技能损失主要发生在 block_to_ws（Workspace 写入映射）环节，head（输出映射）是第二损失点；
块状态表示本身不是原发损失点（探针证明信息仍可解码），但会被 block_to_ws 的更新经 Workspace 递归污染。**

证据链（全部来自本轮实测，详见 DIAGNOSTICS.json）：

1. **参数替换归因（最强证据）**：
   | 干预 | A 组 EM | 解读 |
   |---|---|---|
   | 002 候选原样 | 0.0 | 全崩 |
   | 002 骨架 + stable 的 block_to_ws | **0.7875** | 换回写入映射挽回 78.75pp——最大单项 |
   | 002 骨架 + stable 的 head | 0.25 | 换回输出头只挽回 25pp |
   | 002 骨架 + stable 两者 | **1.0** | 双换回完全恢复（无不可逆损伤） |
   | stable 骨架 + 002 的 block_to_ws | 0.25 | 反向验证：b2w 单独即可毁掉 75pp |
   | stable 骨架 + 002 的 head | 0.7875 | head 单独毁 21.25pp |
   | stable 骨架 + 002 双模块 | 0.0 | 双重损坏叠加 |
2. **递归污染通道实锤**：003（只训 block_to_ws）的位置线性探针从 stable/002 的 **1.0 降到 0.75**。
   块状态 h 的计算本不经过 block_to_ws，但 Workspace 内容由它写入——`read → route → blocks`
   的递归路径把写入映射的更新传导回 h，位置信息被部分冲掉（探针可测出），对应 003 的 -82.5pp 遗忘。
3. **表示不是原发损失**：002 的位置/实体探针均为 1.0（主干逐位未动，h 中位置/实体信息完好可解码），
   A 组却 EM=0——信息在表示里，坏的是**读出通道**（b2w 写入 + head 解码）。
4. **激活替换佐证参数替换结论**：003+stable_h（单点换回块状态）挽回 67.5pp 但不及参数级替换
   （002+stable_b2w 挽回 78.75pp）——激活替换只救当前步，自回归后续 token 仍被候选写入污染；
   参数替换是根治。002+stable_h 仅 +15pp，再次指向 b2w 的写入通道而非块状态本身。
5. **Workspace 槽机制本身无恙**：slot 占用/读取注意力轨迹与 stable 结构一致；stable+002 的
   Workspace/块状态替换后 B 仍为 0——B 的失败是"新绑定从未被学会"（三轮训练均未产生跨位置
   泛化），不存在"被换掉的 B 能力"。
6. **诚实口径**：模块组合存在非线性相互作用（002+stable_b2w 0.7875 ≠ 0+78.75pp 的线性叠加），
   单次替换效果不解释为唯一因果来源；上述归因是"最大贡献点排序"，不是单点因果声明。
   探针监督只作表示诊断，不计入 N-MIND 自主语言能力成绩。

## 二、解码协议修复（第二阶段）

- **实锤**：`free_generation`（v1）把已含"助手："的 prompt 前向后，又在 generate 内部把 anchor
  二次前向——首个生成 token 的条件状态是"prompt + 二次 anchor"。001-003 全部臂共用此口径，
  臂间比较内部一致，历史判定不改写。
- **修复**：`generate` 新增 `primed` 直续模式（prompt 单次前向，从末位 logits 续写）；
  新增 `free_generation_v2`（协议 `freegen-v2-direct-continuation`），v1 保留（协议
  `freegen-v1-anchor-reforward`）供 001-003 复现。测试：首 Token 一致性 / EOS 停止 /
  checkpoint 重载逐 token 一致 / 跨轮状态续算一致（`tests/test_gateb004_fixes.py`）。

## 三、验收契约修复（第二阶段，learning_loop）

1. 遗忘评测缺失不再自动通过（旧代码 `forget is None or ...` 的假阳性通道已封死）；
2. 负对照纳入正式 gates（`negative_control`，002 轮人工核对的 gate_3 补代码化）；
3. 反事实绑定跟随率进入候选验收（`counterfactual_follow` + `counterfactual_follow_min`
   门槛由预登记声明，默认只记录不判定，不偷设门槛）；
4. 缺必要实验臂 / 缺独立封存集 → `status=INCONCLUSIVE`（缺证据 ≠ 失败 ≠ 通过，
   `inconclusive_reasons` 留痕；训练与取证照常进行）；
5. 旧技能保持率必须使用独立封存集（retention_pairs 缺失绝不回退训练行——R1 假遗忘口径的根）；
6. 历史门槛 `NEW_SKILL_GAIN_THRESHOLD=0.10` / `FORGETTING_LIMIT_PP=3.0` 一字不动。
7. lm_update：`attempted_steps`（含非有限跳过）与 `optimizer_steps_executed`（真实 step）拆分，
   跳过不再冒充完成优化；`trainable`/`adapter_prefixes` 白名单校验（未知模式/前缀直接报错）；
   可选 `record_grad_norms`（每组参数梯度范数 last/peak + 训练范围 + trainable_names，默认关闭）。

全仓回归：**120 项测试全绿**（108 既有 + 12 新增；test_09 的终判断言按 004 新契约更新，
buffer drift 取证断言不变）。

## 四、第五阶段：输出架构修复方案（只给接口与草案，不执行）

**方案 A（Workspace 写入门控，最小面）**：`block_to_ws` 输出乘任务门向量
`g = σ(W_g · [route.weights, slot_occupied])`，新技能写入走门控旁路；接口：
`block_to_ws_gated(payload, gate) -> payload'`；不动 stable 权重。
**方案 B（技能残差适配器）**：冻结 stable 全部参数，新增 `block_to_ws_residual`
（2 层小 MLP，按实体 token 触发的路由开关 `skill_gate(z)` 选择残差支路），
旧任务走原路径（逐位零污染），新任务走残差；验证指标 = A 保持 ≥ -3pp 且 B gain ≥ +0.10。
**方案 C（混合课程 + 回放）**：A+B 数据混训（非序列），验证冲突是否来自序列范式本身；
直接用现有 `run_candidate_update`，预登记 A:B 混合比例与 replay_ratio 扫描。
**建议次序：C（最便宜，一轮可判）→ B（若 C 证明序列范式是主因则直接跳 B）→ A（工程兜底）。**

Gate-B 005 预登记草案（DRAFT，未批准不执行）：`gateb_pre_registration_005_DRAFT.json`。

## 五、交付清单

| 交付物 | 位置 |
|---|---|
| Gate-B 003 提交哈希 | `d1ccca3`（001/002 见 `87922db`/`4dacfc1`） |
| 解码与验收修复代码 | `decoder.py`（primed + v2 协议）、`learning_loop.py`（六项）、`lm_update.py`（口径/白名单/梯度范数） |
| 修复与诊断测试 | `tests/test_gateb004_fixes.py`（12 项）、`tests/test_gateb004_diagnostics.py`（5 项） |
| 诊断脚本 | `experiments/bot_runtime_r2/diagnose_gateb004.py`（104s 零训练跑毕） |
| 诊断模块 | `src/nmind/language/state_probe.py`、`src/nmind/language/workspace_diagnostics.py` |
| 诊断与干预数据 | `GATEB004_DIAGNOSTICS.json`（+ results/bot_r2/ 镜像） |
| 005 预登记草案 | `gateb_pre_registration_005_DRAFT.json`（DRAFT） |
| 全仓回归 | 120 项全绿 |
| 历史完整性 | 001-003 封存数据与原始指标零改动（git status 干净）；候选权重/磁盘 checkpoint 全程零写入 |

> AI生成
