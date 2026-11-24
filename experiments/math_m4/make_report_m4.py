"""生成 M4 阶段报告与派生证据文件。

全部输入来自本目录已真实生成的审计/实验 JSON。
未授权训练不会伪造能力、样例、ISO 或效率等级；所有未运行项显式 NOT_RUN。
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(name: str) -> dict:
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def write_json(name: str, obj: dict) -> None:
    (HERE / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def derived_artifacts() -> dict:
    data = load("M4_DATA_AUDIT.json")
    m31_leak = load("M31_LEAKAGE_AUDIT.json")
    m31_cf = load("M31_COUNTERFACTUAL_AUDIT.json")
    m31_ckpt = load("M31_CHECKPOINT_AUDIT.json")
    cost = load("M4_TRANSFORMER_COST_ANALYSIS.json")
    low = load("M4_LOW_RANK_ATTENTION.json")
    sel = load("M4_SELECTIVE_ATTENTION.json")
    inc = load("M4_INCREMENTAL_KERNEL.json")
    orth = load("M4_ORTHOGONAL_COMPRESSION.json")
    exact = load("M4_EXACT_REDUCTION.json")
    info = load("M4_INFORMATION_ANALYSIS.json")
    formula = load("M4_FORMULA_SEARCH_RESULTS.json")
    pareto = load("M4_PARETO_RESULTS.json")
    train = load("M4_TRAINING_RESULTS.json")
    accounting = load("M4_COMPUTE_ACCOUNTING.json")
    tests = load("M4_MATH_TESTS.json")

    iso_comparisons = {k: v for k, v in train.items() if k.endswith("_iso")}
    write_json("M4_ISO_CAPABILITY_COMPARISON.json", {
        "status": "NOT_RUN",
        "source": "M4_TRAINING_RESULTS.json",
        "authorization": train["authorization"],
        "overall_label": "NOT_COMPARABLE",
        "comparisons": iso_comparisons,
        "efficiency_grades": {
            name: {"E1": "NOT_AWARDED", "E2": "NOT_AWARDED",
                   "E3": "NOT_AWARDED", "E4": "NOT_AWARDED", "E5": "NOT_AWARDED"}
            for name in iso_comparisons
        },
        "note": "训练未授权、step=0；TARGET_NOT_REACHED 不是已训练失败，而是无能力证据。",
    })

    ledger = HERE / "M4_BUDGET_LEDGER.jsonl"
    if not ledger.exists():
        ledger.write_text(json.dumps({
            "type": "ledger_initialized",
            "status": "NOT_STARTED",
            "authorization": "NOT_AUTHORIZED",
            "max_steps": 4096,
            "steps_used": 0,
            "max_wall_seconds": 3600,
            "note": "未授权，不写 session_start；任何真实 step 必须在授权后登记。",
        }, ensure_ascii=False) + "\n", encoding="utf-8")

    leakage = dict(m31_leak)
    leakage["scope"] = "M3 历史数据审计 + M4 新数据 canonical 复核"
    leakage["m4_new_data_canonical_overlap"] = data["canonical_overlap"]
    leakage["m4_new_data_n_rows"] = data["n_rows"]
    leakage["conclusion"] = (
        "M3 字符串 prompt 零重合不等于语义隔离；M4 新数据 canonical 重合为 0。"
    )
    write_json("M4_LEAKAGE_AUDIT.json", leakage)

    routing = sel["routing_cost"]
    routing_total = sum(routing[k] for k in ("summary_build", "summary_update",
                                             "coarse_retrieval", "topk", "gather",
                                             "exact_attention"))
    attention = {
        "status": "COMPLETE_ZERO_GRADIENT",
        "reference": "R2 TinyTransformerM3, FP32, CPU",
        "transformer_cost": {
            "short_seq": cost["bottleneck_short_seq"],
            "long_seq": cost["bottleneck_long_seq"],
            "causal_savings_by_length": {
                length: cost["by_length"][length]["B1"]["analytic"]["causal_savings_ratio"]
                for length in ("16", "32", "64", "128", "256", "512")
            },
        },
        "low_rank_interaction": {
            "mechanism": "reduce_qk_dim（不是投影分解，也不是 score 低秩近似）",
            "full_graph_reduction": {
                r: low["mechanisms"][f"reduce_qk_dim_r{r}"]["full_graph_reduction"]
                for r in (4, 8, 16, 32, 64)
            },
            "output_relative_l2": {
                r: low["mechanisms"][f"reduce_qk_dim_r{r}"]["output_error"]["relative_l2"]
                for r in (4, 8, 16, 32, 64)
            },
            "frozen_main_r": 16,
            "trained_capability": "NOT_RUN",
        },
        "selective_interaction": {
            "exact_attention_skipped_ratio": sel["skipped_attention_ratio"],
            "routing_macs": routing,
            "routing_plus_exact_macs": routing_total,
            "dense_attention_macs": routing["dense_attention"],
            "routing_plus_exact_vs_dense_saving": round(
                1.0 - routing_total / routing["dense_attention"], 4),
            "causality_pass": sel["causality_pass"],
            "ste_gradient_pass": sel["ste_gradient_pass"],
            "fixed_window_capability": sel["fixed_window_capability"],
            "trained_capability": "NOT_RUN",
        },
        "incremental_kernel": {"status": inc["status"], "decision": inc["decision"],
                               "disclaimer": inc["disclaimer"]},
        "conclusion": ("注意力成本确实随长度成为主瓶颈，但只降低 QK 维度不会等比例降低全图；"
                       "选择性交互跳过 75% 精确注意力，但需计入路由成本；训练后能力均未测。"),
    }
    write_json("M4_ATTENTION_ANALYSIS.json", attention)

    write_json("M4_INFORMATION_INTERACTION.json", {
        "status": "COMPLETE_ZERO_GRADIENT",
        "proxy_disclaimer": info["disclaimer"],
        "information_proxies": info,
        "operator_observations": {
            "low_rank": low["conclusion"],
            "selective": sel["note_cost"],
            "incremental": inc["decision"],
            "orthogonal": orth["decision_rule"],
        },
        "formula_pareto_operator_stage": pareto["pareto_front_operator_stage"],
        "trained_capability": "NOT_RUN",
        "conclusion": "信息代理只能初筛，不能证明任务能力或无损压缩。",
    })

    write_json("M4_GENERALIZATION.json", {
        "status": "DATA_READY_MODEL_NOT_RUN",
        "data_audit": data,
        "counterfactual": {
            "types": data["counterfactual_types"],
            "query_positions": data["counterfactual_query_positions"],
            "pair_rule": "配对两条都正确才计通过",
            "m3_shortcut_evidence": {
                "query_first_ratio": m31_cf["rates"]["query_first_ratio"],
                "pair_first_value_shortcut": m31_cf["rates"]["pair_first_value_shortcut"],
            },
        },
        "distance": {"distances": data["distance_distances"],
                     "pair_ids": data["distance_pair_ids"]},
        "model_results": {
            "counterfactual_same_query_fact_change": "NOT_RUN",
            "counterfactual_unrelated_fact_invariance": "NOT_RUN",
            "query_switch": "NOT_RUN",
            "order_swap": "NOT_RUN",
            "distance_0_4_8_16_32_64_128": "NOT_RUN",
            "unseen_length": "NOT_RUN",
        },
    })

    write_json("M4_RESOURCE_BENCHMARK.json", {
        "status": "COMPLETE_ZERO_GRADIENT",
        "hardware": cost["hardware"],
        "measurement": cost["method"],
        "reference_transformer": {
            "params": cost["model"]["params"],
            "by_length": cost["by_length"],
            "incremental": cost["incremental"],
            "profiler_top": cost["profiler"],
        },
        "batch_accounting_self_test": accounting,
        "exact_reduction": exact,
        "energy": "NOT_AVAILABLE",
        "trained_model_inference": "NOT_RUN",
        "note": "固定长度性能、真实任务生成、同准确率性能三者不可互相替代。",
    })

    ckpt_dir = HERE / "checkpoints"
    write_json("M4_CHECKPOINT_AUDIT.json", {
        "status": "M4_NOT_AVAILABLE_M3_HISTORICAL_AUDIT_COMPLETE",
        "m4_checkpoint_dir": str(ckpt_dir),
        "m4_checkpoint_files": sorted(p.name for p in ckpt_dir.glob("*.pt")) if ckpt_dir.exists() else [],
        "m4_training_status": train["status"],
        "m4_authorization": train["authorization"],
        "m3_historical_audit": {
            "n_checkpoints": m31_ckpt["n_checkpoints"],
            "panel": m31_ckpt["panel"],
            "note": m31_ckpt["note"],
            "best_final_relations": m31_ckpt["best_final_relations"],
        },
        "conclusion": ("M3 checkpoint 证据链已复核；M4 尚无真实训练权重，"
                       "任何 M4 Best/Final 能力结果均 NOT_AVAILABLE。"),
    })

    lines = []
    for i in range(40):
        lines.append(json.dumps({
            "idx": i + 1,
            "status": "NOT_RUN",
            "reason": "M4 formal training not authorized; no real checkpoint available",
            "checkpoint": "NOT_AVAILABLE",
            "arch": "NOT_AVAILABLE",
            "prompt": None,
            "expected": None,
            "predicted": None,
            "m3_real_samples_reference": "../math_m3/M3_GENERATION.jsonl",
        }, ensure_ascii=False))
    (HERE / "M4_GENERATION.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {"routing_total": routing_total, "pareto": pareto, "tests": tests,
            "sel": sel, "low": low, "exact": exact, "train": train, "data": data,
            "cost": cost, "formula": formula, "orth": orth, "inc": inc, "info": info}


def report_text(ctx: dict) -> str:
    tests = ctx["tests"]
    sel = ctx["sel"]
    low = ctx["low"]
    exact = ctx["exact"]
    train = ctx["train"]
    data = ctx["data"]
    cost = ctx["cost"]
    formula = ctx["formula"]
    pareto = ctx["pareto"]
    orth = ctx["orth"]
    inc = ctx["inc"]
    routing_total = ctx["routing_total"]

    short = cost["bottleneck_short_seq"]
    long_ = cost["bottleneck_long_seq"]
    front = ", ".join(pareto["pareto_front_operator_stage"])
    low_r4 = low["mechanisms"]["reduce_qk_dim_r4"]["full_graph_reduction"]

    return f"""# N-MIND M4 阶段报告｜最小充分信息交互

> 状态：**ZERO_GRADIENT_STAGE_COMPLETE**  
> 分支：`experiment/minimum-sufficient-interaction-m4`  
> 正式训练授权：**未授予**。当前所有 M4 主实验、正控制训练、ISO/Pareto 能力结论均为 `NOT_RUN`。  
> 本阶段只完成第一至第八阶段，以及第九阶段的授权确认准备。

---

## 一、本阶段真实完成的工作

1. **M3.1 历史审计**：监督协议、反事实捷径、数据语义泄漏、距离配对、Checkpoint 证据链、成本记账均已复核，详见 `M31_REPAIR_REPORT.md`。
   - M3 prompt 监督占比 train=0.8964；M3 EOS 权重 0.25，不是 1.0。
   - M3 反事实 400/400 查询第一位，“输出第一个数字”捷径配对成功率 1.0。
   - M3 成本按全局步数乘算，单模型虚增 1×–8×。
   - 字符串 prompt 零重合不等于语义隔离；canonical train/val 重合 47，train/sealed 重合 54。
   - 16 个 M3 checkpoint 均可加载，Best/Final 标记未发现错配。

2. **数学研究**：`M4_RESEARCH_SOURCES.md` 归档原始论文及核实状态，二手报道标 `NOT_VERIFIED`。

3. **Transformer 成本解剖**：`M4_TRANSFORMER_COST_ANALYSIS.json` 完成 16/32/64/128/256/512 长度、Batch 1/8/16、解析 MAC/Profiler/墙钟三口径分离。

4. **数学算子**：低秩信息交互、选择性交互、增量核、正交压缩、数学等价 head 优化均完成零梯度实验。

5. **公式发现实验室**：29 个候选，17 个存活，算子阶段 Pareto 前沿为 **{front}**。

6. **新数据与框架**：M4 数据 train=1392、val=464、sealed=464，canonical 重合全为 0；四类反事实各 100 对，查询位置 first/middle/last=136/132/132；严格距离配对 60 组 × 7 个距离。

7. **P14 数学验证**：`M4_MATH_TESTS.json` = **{tests['n_pass']}/{tests['n_tests']} PASS**。

8. **训练器**：`train_m4.py --authorize` 已就绪，但当前只跑过 dry-run；`M4_TRAINING_RESULTS.json` 明确为 `{train['status']}`，实际 gradient step=0。

---

## 二、最终报告 26 问

### 1. M3 历史实验修复了哪些问题？

确认并修复了六类问题：监督协议错配、反事实位置捷径、A5_repeat 训练缺失、语义族泄漏、距离数据非严格配对、成本统计虚增。Checkpoint 标记错配担忧未被证实。历史原始报告未修改。

### 2. 现有参考 Transformer 真正的计算瓶颈在哪里？

随长度变化。短序列 attention share={short['attention_share']}，FFN={short['ffn_share']}；长序列 attention share={long_['attention_share']}，FFN={long_['ffn_share']}。Profiler 还显示 LayerNorm、addmm/mm/bmm、arange、copy_、softmax 等工程热点。

### 3. 注意力计算占整个模型计算量多少？

按解析口径：短序列 {short['attention_share']:.2%}，长序列 {long_['attention_share']:.2%}。这不是墙钟占比；墙钟还受内核实现、内存访问和 CPU 调度影响。

### 4. 不同输入长度下瓶颈是否变化？

是。有效三角区相对稠密矩阵的节省从 T=16 的 46.88% 增至 T=512 的 49.90%，但参考实现仍执行稠密矩阵，未用融合精确因果内核兑现该理论节省。

### 5. 哪些数学运算可以严格消除？

Answer-only 下 Prompt 位置无需计算词表 Head。数值测试 logits 最大差 {exact['logits_max_abs_diff']}、loss 差 {exact['loss_max_abs_diff']}、梯度差 {exact['gradient_max_abs_diff']}、参数更新差 {exact['parameter_update_max_abs_diff']}，贪心下一 token 一致。但 CPU 实测时间变化 {exact['measured_time_reduction']:.2%}（负收益），因此 **未获得可确认的实际节省**。

### 6. 哪些计算只能近似替代？

降低 QK 交互维度、分组摘要 + top-k 精确交互、线性核增量状态、正交低维压缩、核映射/归一化/门控公式变体，都是近似或带条件的机制，不能宣称与 softmax attention 等价。

### 7. 低秩信息交互是否保持任务能力？

**NOT_RUN**。算子层 output relative_l2 在 r=4/8/16/32/64 均较大，且全图收益从 r=4 的 {low_r4:.2%} 降到 r=64 的 -33.33%。没有训练后能力证据，不能判断是否保持。

### 8. 最有效的交互维度是多少？

按算子 MAC：r=4 全图节省最高（{low_r4:.2%}）；按预登记主实验：冻结 r=16，以误差和表达维度折中。按任务能力：**NOT_RUN**。

### 9. 选择性交互是否真正跳过矩阵计算？

是，算子层跳过精确注意力比例 {sel['skipped_attention_ratio']:.2%}；精确注意力 MAC 为 65,536，稠密为 262,144。因果扰动差为 {sel['causality_future_perturbation_max']}，STE 梯度范数 {sel['ste_gradient_norm']}。

### 10. 稀疏选择自身需要多少计算成本？

摘要构造 16,384，摘要更新 2,048，粗检索 32,768，top-k 2,048，gather 32,768，精确注意力 65,536，合计 **{routing_total:,} MAC**；相对稠密 262,144 的名义节省为 {1-routing_total/262144:.2%}。不能只报精确注意力节省。

### 11. 训练与实际部署是否一致？

部分一致但不完全等价。M4-B 训练用软门、推理用硬门；P14 测试只验证形状与有限性，不验证 logits 数值相等。必须在授权训练后单独报告软/硬门差距，禁止用软门能力搭配硬门成本。

### 12. 新公式是否减少实际 CPU 时间？

算子层 F17 中位耗时 52.9μs，低于参考 F01 的 84.55μs，但相对误差 0.660377。只有算子初筛证据，训练后能力与同等能力效率均为 **NOT_RUN**。

### 13. 反事实能力是否超过位置捷径？

数据层已超过：M4 查询位置均衡且四类反事实各 100 对；M3 的 100% 第一位捷径已被拆掉。模型能力层 **NOT_RUN**。

### 14. 查询切换能力是否真正建立？

数据层覆盖了 query_switch 100 对；模型层 **NOT_RUN**，不能宣称建立。

### 15. 多字符连续生成是否获得突破？

**NOT_RUN**。M4_GENERATION.jsonl 为 40 条明确 `NOT_RUN` 记录，不是生成结果。M3 的真实 48 组生成样例保留在 `../math_m3/M3_GENERATION.jsonl`。

### 16. 小规模正控制实验是否通过？

训练未运行，标记 `NOT_RUN_TRAINING`。当前只验证监督对齐；正控制数据已扩充为双字符复制、多字符复制和双变量绑定。

### 17. 是否具备真正的长度泛化能力？

数据层具备严格配对：距离 0/4/8/16/32/64/128 各 60 条，共享 60 个 pair_id。模型评测 **NOT_RUN**。

### 18. 是否减少达到相同能力所需的样本数量？

**NOT_COMPARABLE**。未训练，无学习曲线。

### 19. 是否减少达到相同能力所需的训练计算量？

**NOT_COMPARABLE**。训练 step=0。

### 20. 是否减少达到相同能力所需的推理计算量？

**NOT_COMPARABLE**。只有固定输入性能与算子成本，无同准确率生成成本曲线。

### 21. 是否达到 E1 或更高效率等级？

没有。所有 ISO 目标为 `TARGET_NOT_REACHED` / `NOT_COMPARABLE`，不授予 E1–E5。

### 22. 哪些收益属于数学优化？

因果有效三角区理论节省约 46.88%–49.90%；Answer-only Head 数值等价。但前者尚未由融合内核兑现，后者在本机词表 102 下实测负收益。

### 23. 哪些收益属于工程实现优化？

R2 已有融合 QKV、相对 RoPE 与 KV Cache；逐 batch 成本记账替代固定长度公式。当前没有新增并验证为正收益的工程优化。

### 24. 哪些新公式存在原创研究价值？

当前公式均为已有算子的受控组合与搜索，不宣称原创数学发现。有价值的是“静态维度检查 + 因果性 + 梯度 + 数值稳定 + 成本 Pareto”的筛选流程，但它属于实验方法，不是新定理。

### 25. 哪些候选应继续验证或者淘汰？

授权后继续：R1、R2、M4-A(r=16)、M4-B(group_size=4, top_k=2)。  
淘汰：Answer-only Head 单独优化（本机负收益）；F02–F09 等违反因果/梯度检查的 score/value 低秩公式；identity kernel（分母不稳定）。  
保留但 NOT_RUN：增量核、正交压缩、M4-C/D/E/F、固定局部窗口基线。

### 26. 是否具备进一步研究低成本自主学习的条件？

接口与纪律已具备：状态、会话记忆、长期权重、持久技能已区分；P18 只做数学建模。没有持久技能学习证据，禁止启动无限自主训练。

---

## 三、证据索引

| 主题 | 文件 | 结论 |
|---|---|---|
| 历史修复 | `M31_REPAIR_REPORT.md` | 五类历史问题被证实，checkpoint 错配未证实 |
| 数学验证 | `M4_MATH_TESTS.json` | {tests['n_pass']}/{tests['n_tests']} PASS |
| 成本解剖 | `M4_TRANSFORMER_COST_ANALYSIS.json` | 瓶颈由 FFN/Head 与 attention 随长度共同决定 |
| 低秩 | `M4_LOW_RANK_ATTENTION.json` | 只有 reduce_qk_dim 真降 score MAC |
| 选择性 | `M4_SELECTIVE_ATTENTION.json` | 跳过 75% 精确注意力，但路由成本必须计入 |
| 增量核 | `M4_INCREMENTAL_KERNEL.json` | 属已有线性注意力，不宣称等价 |
| 正交压缩 | `M4_ORTHOGONAL_COMPRESSION.json` | 小矩阵增量 QR 不快于全量 SVD |
| 等价优化 | `M4_EXACT_REDUCTION.json` | 数值等价，CPU 无收益，淘汰 |
| 公式搜索 | `M4_FORMULA_SEARCH_RESULTS.json` | 29 候选，17 存活，Pareto {front} |
| 数据 | `M4_DATA_AUDIT.json` | train/val/sealed canonical 重合全为 0 |
| 预登记 | `M4_PREREGISTERED_PLAN.json` | 授权状态 NOT_YET_GRANTED |
| 训练 | `M4_TRAINING_RESULTS.json` | DRY_RUN_COMPLETE，0 step |
| 生成 | `M4_GENERATION.jsonl` | 40 条 NOT_RUN，非生成样例 |

---

## 四、训练授权请求（第九阶段）

M4 预登记的正式预算为：

- 正控制：R1 128 步 + R2 128 步 = 256 步；
- 正式对照：R1/R2/M4-A/M4-B × 2 seed × 480 步 = 3840 步；
- 合计最多 **4096 optimizer.step**，墙钟上限 **3600 秒**；
- 授权命令：`python positive_control.py --authorize` 与 `python train_m4.py --authorize`。

**当前状态：未授权，未执行任何梯度更新。** 用户明确授权前，不会启动训练，也不会把 dry-run 写成训练结果。
"""


def main() -> None:
    ctx = derived_artifacts()
    report = report_text(ctx)
    (HERE / "M4_REPORT.md").write_text(report, encoding="utf-8")
    print(f"[report] 写出 {HERE / 'M4_REPORT.md'}")
    print(f"[report] 派生证据文件已更新，数学验证 {ctx['tests']['n_pass']}/{ctx['tests']['n_tests']}")


if __name__ == "__main__":
    main()
