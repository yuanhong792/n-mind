"""从真实训练结果生成 C1_REPORT.md 与 C1_EVALUATION.json、C1_ROUTER/MEMORY/RESOURCE 审计。

只读取真实落盘的 C1_TRAINING_RESULTS.json 与 checkpoint，不伪造任何数据。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent


def write(name, obj):
    (HERE / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    tr = json.loads((HERE / "C1_TRAINING_RESULTS.json").read_text(encoding="utf-8"))
    sealed = tr["sealed_evaluation"]
    cf = tr["counterfactual"]
    raudit = tr["router_audit"]
    rgrad = tr["router_gradient"]
    history = tr["history"]

    # 分任务评测（来自 sealed，全量真实）
    per_task = sealed["per_task"]
    n_train = tr["train_pool"]

    # C1_EVALUATION.json
    write("C1_EVALUATION.json", {
        "status": "EVALUATION_DONE",
        "checkpoint": "candidate_compute_best.pt",
        "checkpoint_step": tr["best_step"],
        "optimizer_steps_executed": tr["optimizer_steps_executed"],
        "per_task": per_task,
        "overall_free_generation_accuracy": sealed["overall_accuracy"],
        "counterfactual": {"pairs": cf["pairs"], "paired_accuracy": cf["paired_accuracy"]},
        "generation_samples": sealed["details"][:60],
        "protocol": "greedy free generation; full answer incl. EOS; no teacher forcing",
    })

    # C1_ROUTER_AUDIT.json
    write("C1_ROUTER_AUDIT.json", {
        "status": "ROUTER_AUDIT_DONE",
        **raudit, **rgrad,
        "verdict_sparse_space": "CONFIRMED" if raudit["has_sparse_space"] else "ABSENT",
        "note": "非零 Router 梯度不等于已获得路由学习能力；须结合评分变化对前向/选择集合影响判断。",
    })

    # C1_MEMORY_AUDIT.json（从训练时的 workspace 与变量任务表现推断）
    bind_tasks = {t: d for t, d in per_task.items() if t.startswith("C2")}
    write("C1_MEMORY_AUDIT.json", {
        "status": "MEMORY_AUDIT_DONE",
        "workspace_mean_occupied": raudit["workspace_mean_occupied"],
        "workspace_occupied_first": raudit["workspace_occupied_first"],
        "workspace_occupied_last": raudit["workspace_occupied_last"],
        "binding_task_accuracy": bind_tasks,
        "note": "N-Cell 块平均读出 / Workspace 读出 / LM Head 未逐层对比可解码性；"
                "本轮以自由生成绑定准确率作为功能口径，层级归因列为 NOT_YET_VERIFIED。",
    })

    # C1_RESOURCE_REPORT.json
    try:
        import torch
        proc_mem = None
        try:
            import psutil
            proc_mem = psutil.Process(os.getpid()).memory_info().rss / 1024**2
        except Exception:
            proc_mem = None
        resources = {
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "torch_version": torch.__version__,
            "cpu_threads": torch.get_num_threads(),
            "process_rss_mb": round(proc_mem, 1) if proc_mem else "psutil_unavailable",
            "params_nmind": tr["params_nmind"],
            "avg_infer_seconds_per_sample": {
                t: d["avg_infer_seconds"] for t, d in per_task.items()},
            "train_wall_seconds": tr["wall_seconds"],
            "optimizer_steps": tr["optimizer_steps_executed"],
            "batches_per_step": tr["batch_size"],
            "checkpoint_files": [str(p.name) for p in HERE.glob("candidate_compute_*.pt")],
        }
    except Exception as e:
        resources = {"error": str(e)}
    write("C1_RESOURCE_REPORT.json", resources)

    # 能力等级汇总
    def agg(prefixes):
        vals = [(t, d) for t, d in per_task.items() if t.startswith(tuple(prefixes))]
        if not vals:
            return {"n": 0, "accuracy": None}
        n = sum(d["n"] for _, d in vals)
        ok = sum(d["n"] * d["free_generation_accuracy"] for _, d in vals)
        return {"n": n, "accuracy": round(ok / max(1, n), 4), "tasks": list(vals and [t for t, _ in vals])}

    levels = {
        "C0_symbol_sequence": agg(["C0"]),
        "C2_variable_binding": agg(["C2"]),
        "C3_arithmetic": agg(["C3"]),
        "C4_logic": {"status": "INTERFACE_ONLY", "reason": "前置能力未全部达标，本轮未训练"},
        "C5_program_understanding": {"status": "INTERFACE_ONLY"},
        "C6_program_generation": {"status": "INTERFACE_ONLY"},
        "C7_error_fixing": {"status": "INTERFACE_ONLY"},
        "C8_autonomous_learning": {"status": "INTERFACE_ONLY", "note": "监督闭环可用，执行反馈学习仅接口"},
        "C9_autoprogram": {"status": "INTERFACE_ONLY"},
    }

    # 报告正文
    lines = [
        "# N-MIND C1.0 计算能力优先学习 — 真实训练报告",
        "",
        f"> 分支 `experiment/compute-first-c1`｜实际 optimizer.step = **{tr['optimizer_steps_executed']}**"
        f"｜墙钟 {tr['wall_seconds']}s｜Random init SHA `{tr['init_model_sha256'][:16]}`",
        "> 未使用外部 LLM / 预训练权重 / 蒸馏；Python 只生成标签与验证，不参与模型推理。",
        "",
        "## 0. 环境与口径",
        "- 项目路径：`D:\\cpu本地模型开发\\n-mind`（任务书路径 `.venv-gate24` 本机不存在），"
        "Python 3.12.10 + torch 2.10.0+cpu（CUDA 不可用）。",
        f"- 计算 Tokenizer：`{tr['tokenizer_sha256'][:16]}`，字符级 101 token，无 BPE 合并；"
        "训练/推理同源。",
        f"- 模型：N-Cell({tr['config']['n_blocks']}块/top{k2(tr)}选2)/Workspace({tr['config']['workspace_slots']}槽)，"
        f"参数量 {tr['params_nmind']}；从随机初始化开始。",
        "- 主指标为**贪心自由生成**（非 teacher forcing）；完整答案含 EOS。",
        "",
        "## 1. 能力等级（自由生成）",
        "| 等级 | 样本 | 准确率 | 状态 |",
        "|---|---:|---:|---|",
    ]
    for k, v in levels.items():
        acc = v.get("accuracy")
        lines.append(f"| {k} | {v.get('n', 0)} | {'—' if acc is None else f'{acc:.2%}'} | {v.get('status', 'TRAINED')} |")
    lines += [
        "",
        f"- 反事实成对正确率：**{cf['paired_accuracy']:.2%}**（{cf['pairs']} 对）",
        f"- Router 稀疏空间：**{'已确认' if raudit['has_sparse_space'] else '缺失'}**，"
        f"激活率 {raudit['activation_ratio']}，选择集合数 {raudit['distinct_set_count']}；"
        f"改评分改变前向 = {raudit['router_score_change_changes_forward']}；"
        f"Router 非零梯度 = {rgrad['has_nonzero_router_grad']}。",
        f"- Workspace 平均占用 {raudit['workspace_mean_occupied']}，"
        f"首/末 {raudit['workspace_occupied_first']}/{raudit['workspace_occupied_last']}。",
        "",
        "## 2. 分任务明细",
        "| 任务 | n | 自由生成 | 立即EOS | 空输出 | 推理秒/样本 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for t, d in sorted(per_task.items()):
        lines.append(f"| {t} | {d['n']} | {d['free_generation_accuracy']:.2%} | "
                     f"{d['immediate_eos_rate']:.2%} | {d['empty_rate']:.2%} | {d['avg_infer_seconds']} |")
    lines += [
        "",
        "## 3. 训练曲线（已教任务 val acc）",
        "| step | 阶段 | taught acc | loss | 墙钟 |",
        "|---:|---|---:|---:|---:|",
    ]
    for h in history:
        lines.append(f"| {h['step']} | {h['stage']} | {h['val_taught_acc']:.2%} | "
                     f"{h['train_loss_last100']} | {h['wall']}s |")
    lines += [
        "",
        "## 4. 真实样本（成功与失败）",
    ]
    det = sealed["details"]
    succ = [d for d in det if d["ok"]][:22]
    fail = [d for d in det if not d["ok"]][:23]
    lines.append("成功样例：")
    for d in succ:
        lines.append(f"- `{d['prompt']}` → 期望 `{d['expected']}`｜模型 `{d['predicted']}` ✓")
    lines.append("失败样例：")
    for d in fail:
        lines.append(f"- `{d['prompt']}` → 期望 `{d['expected']}`｜模型 `{d['predicted']}` ✗")
    lines += [
        "",
        "## 5. 十问回答",
        "1. 符号复制：见 C0_copy1/C0_copyN 行，达标与否以自由生成准确率为准，未达标不宣称学会。",
        "2. 记住多个变量：见 C2_bindN 各行；随变量数增加的表现差异在 C1_EVALUATION.json 分任务表中。",
        "3. 按查询提取变量：C2_bind1/bindN 自由生成准确率即答案；反事实成对正确率见上。",
        "4. 变量重赋值：见 C2_reassign 行。",
        "5. 数学运算：见 C3_arith/C3_var_arith 行。",
        "6. 简单程序理解：本轮为接口（MiniLang + environment），未宣称模型已理解。",
        "7. 程序生成：仅解释器/训练/评估接口就绪，未训练、未宣称。",
        "8. 记忆 vs 泛化：训练内组合高准确率属记忆证据；未见组合/长度泛化需对比 "
        "C0_copyN 的 length>in_distribution 与 sealed 分任务，当前若无单独泛化列则标记 NOT_YET_VERIFIED。",
        "9. N-Cell/Router/Workspace 作用：Router 稀疏空间与负载见 C1_ROUTER_AUDIT.json；"
        "Workspace 占用见 C1_MEMORY_AUDIT.json；N-Cell 层级可解码性本轮未逐层探针，NOT_YET_VERIFIED。",
        "10. 距离自主编程：仍缺稳定的多变量泛化、组合程序执行预测、执行反馈权重更新（仅接口）。",
        "",
        "## 6. 诚实边界",
        "- 本轮不因某项简单任务成功就宣称通用智能。",
        "- C4–C9 未训练/未达标的项目均明确标注 INTERFACE_ONLY。",
        f"- 权重仅本地保存，未上传 GitHub；Stable 与历史语言实验未改动。",
    ]
    (HERE / "C1_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"levels": {k: v.get("accuracy") for k, v in levels.items()},
                      "cf": cf["paired_accuracy"], "router_sparse": raudit["has_sparse_space"],
                      "steps": tr["optimizer_steps_executed"]}, ensure_ascii=False, indent=2))


def k2(tr):
    return tr["config"]["top_k"]


if __name__ == "__main__":
    main()
