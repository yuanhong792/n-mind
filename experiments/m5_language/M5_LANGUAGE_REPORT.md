# M5-Language 中文神经语言交互 首轮报告

日期：2026-11-22（AutoClaw 会话）｜任务书：`../N-MIND_M5-Language_B8E687A5E78FBAE492_CodexA0E5918FE7BAAF%8D.md`
预算（预登记，全部遵守）：单 seed 20261241、**200/200 optimizer 更新**、墙钟 **154.4s / 300s**、CPU 8 线程。

## 0. 状态判定（任务书第 10 节口径）

| 状态 | 值 | 依据 |
|---|---|---|
| PIPELINE_READY | **true** | 数据→分词→训练→自由生成→评测→CLI 全链路跑通 |
| FREE_GENERATION_WORKS | **true** | 无 teacher forcing 的自回归生成全部分支可用（100% 非空输出） |
| IN_DOMAIN_PASS | **false** | sealed_test 自由生成 EM 3.61%，未达"多数基线+10pp 且 ≥2 类型"门槛 |
| OOD_PASS | **false** | 四个 OOD split EM 0~0.98%，无一过门槛 |
| OPEN_CHAT_NOT_VALIDATED | true | 开放域对话未验证，也不在本轮目标内 |
| 综合判定 | **PIPELINE_READY_LANGUAGE_NOT_VALIDATED** | 按任务书：管线可用，语言能力未验证 |

## 1. 交付物与代码清单

- `src/nmind/language/tokenizer.py`：自有字符级分词器（中文逐字/ASCII 聚合/PAD-BOS-EOS-UNK/词表冻结哈希；词表只从 train 构建，评测不扩词表——未见字符如实显示 \ufffd，本轮还修复了 UNK 被 decode 静默丢弃的 bug）
- `src/nmind/language/dialogue_data.py`：中文受限微世界多轮对话生成器（独立 oracle；7 个 split：train/dev/sealed_test/ood_name/ood_paraphrase/ood_composition/ood_length；模板与改写预登记；manifest 含 seed、sha256、逐意图多数基线）
- `src/nmind/language/neural_dialogue.py`：NMindLM——**NCell 块稀疏更新 + BlockRouter Top-K + GlobalWorkspace 读写**作为序列计算主体，token embedding 与 LM head 随机初始化；递归状态跨轮携带；checkpoint 版本化格式 nmind-lm/1（词表哈希不匹配拒绝加载；load 时强制 eval——本条是测试抓出的真 bug）
- `src/nmind/language/decoder.py`：自回归自由解码（greedy/带种子采样/温度/重复惩罚；停止于 EOS/换行/max_tokens；逐步 logprob 记录；绝无目标泄漏）
- `src/nmind/language/session.py`：ChatSession（跨轮状态吸收/工具路由/response_source 标注/reset/轮数与输入长度预算）
- `src/nmind/language/train_language.py` + `evaluate_language.py`：训练与评测入口
- CLI：`python -m nmind.cli train-language` / `evaluate-language` / `chat`（--diagnostics 显示 response_source/tool/latency/generated）
- `configs/language_m5.yaml`（网络/自动部署默认关）
- 测试：`tests/test_language_m5.py` 10 项（全量 **92 项全绿**，EXIT=0）

## 2. 训练与评测数据（全部实测）

| 项 | 值 |
|---|---|
| seed / 步数 | 20261241 / 200（预算 200，无加训） |
| 墙钟 / 参数量 | 154.4s / 452,116 |
| 词表 | 123（train-only，hash b346…系） |
| train loss | 4.83 → 0.71（dev loss 1.15） |
| **char_accuracy_teacher_forced**（仅诊断） | **74.41%**（sealed_test） |
| **free_generation_exact_match**（主指标） | sealed_test **3.61%**；ood_name 0%；ood_paraphrase 0%；ood_composition 0.98%；ood_length 0% |
| 对照：随机未训练同构模型 | 0.0% |
| 对照：规则/模板基线（rule_baseline） | qa_neg 1.0 / qa_count 1.0 / tool_lookup 1.0 / unanswerable 1.0；qa_pos 1.0；qa_pos_ref 0.75；qa_compare 0.0（规则解析器自身缺陷，如实保留） |
| 延迟 | p50 11.5ms / p95 21.0ms（每轮生成） |
| 资源 | CPU 8 线程；RSS 与 M4 同量级（~650MB，主要为 PyTorch 运行时；精确峰值未单独记录——列入未完成项） |

## 3. 真实交互样例（CLI 演示，错误全部保留）

完整记录：`chat_demo_transcript.txt`。真实节选（--diagnostics）：

```text
用户：小明在3号位置。
脉脑> （空生成）            [response_source=neural finish=eos generated='']
用户：它在哪里？
脉脑> 小明在哪？            [finish=newline generated='小明在哪？']   ← 回声式模仿，未答位置
用户：把小明移动到5号位置
脉脑> 的用户：小红的位置。  [generated='的用户：小红的位置。']       ← 结构模仿+幻觉拼接
用户：请计算：12加7等于多少？
脉脑> 用户：小丽在号位置。  [finish=eos]                              ← 未触发工具路由，答非所问
```

诊断：模型在 200 步内学会了**对话结构**（用户：/助手：/换行轮转，TF 字符准确率 74%）但完全不会**世界状态追踪**——自由生成退化为"回声 + 模板片段"。与 M3 v1（dev EM 24.63%，600 步 teacher-forced）可比性有限：任务与数据不同，不可直接比绝对值；本轮 M3 数字经核对仍与仓库记载一致。

## 4. 基线与门槛判定

- 门槛（预登记）：主任务 free-gen EM ≥ 多数基线 +10pp 且 ≥2 意图类型超基线。**未通过**：没有任何意图类型超基线（qa_compare 最高 37.5% vs 多数基线 87.5%）。
- 规则基线在本合成域接近满分（qa_pos/qa_neg/qa_count/tool_lookup/unanswerable = 1.0）——**规则在该任务族上碾压 200 步神经模型**，这正是任务书要求把 rule_baseline 单独命名、不计入神经成绩的原因。
- 随机基线 0.0%：神经管线确实学到了文本结构（TF 74%），不是纯随机。

## 5. 失败分析（为什么 EM≈0）

1. **监督量不足**：200 步 × 16 批 × ~120 token ≈ 38 万 token 次监督，而答案字符串要求逐字符精确——按 M3/M4 经验，这个量级只够学会表层结构。
2. **世界追踪是隐式的**：模型必须纯靠文本历史+递归状态维护 (实体→位置/颜色) 映射，没有任何结构化状态输入；这是设计上的真难点（任务书明确不许把 oracle/规则解析喂给模型冒充理解）。
3. **工具路由未学出**：tool_calc/tool_lookup 的选择率 0%——路由 token（TOOL_CALCULATOR/TOOL_LOOKUP）是低频 token，200 步不足以涌现。
4. 已排除项：无标签泄漏（test_2 证明生成只依赖 prompt）；无 buffer 虚假增益（test_9：漂移改变前向→复位恢复；另修复了 load_lm 缺 eval() 的真 bug）。

## 6. 已实现 / 已测试 / 已通过 / 已失败 / 尚未实现

- **已实现+已测试**：全部分词/数据/模型/解码/会话/CLI 模块 + 10 项专项测试。
- **已通过**：PIPELINE_READY、FREE_GENERATION_WORKS、92 项回归、tokenizer 词表冻结、生成无标签泄漏、会话隔离、checkpoint 词表校验、buffer 复位协议。
- **已失败**：IN_DOMAIN_PASS、OOD_PASS（预登记门槛）。
- **尚未实现**：自由生成的世界追踪准确率（需更大训练预算）、工具路由涌现、多模态接口实现（本轮仅规划）、M5-Network 的 P1-P3（CapabilityTracker/ResourceManager/SkillRegistry——被本指令接替，尚未开发）、持续学习接入语言域。

## 7. 三条可复制命令

```powershell
python -m nmind.cli train-language --device cpu
python -m nmind.cli evaluate-language --device cpu
python -m nmind.cli chat --mode neural --diagnostics --device cpu
```

## 8. 下一步建议（需小袁批准，未自行启动）

1. **扩大语言训练预算**：200 步只够验证管线。建议预登记 2000-4000 步（约 20-30 分钟 CPU）+ 扩大 train 对话量，目标先过 in-domain 门槛。
2. **给模型显式会话状态接口**：把 world_state 以结构化 token 注入（保持"神经理解另测"口径），可能大幅降低追踪难度。
3. **工具路由上采样加权**：tool 意图样本扩量。
4. M4.1 的 B_OOD 长度外推修复（混合时域经验流）仍待批准——与语言域共享"更长序列泛化"难题。
