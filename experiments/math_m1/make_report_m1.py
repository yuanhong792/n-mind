"""M1 最终独立评测、Pareto 分析与报告生成。"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from dataset_m1 import M1Tokenizer, vocab_sha
from candidate_registry import build_model
from evaluate_m1 import (load_rows, build_fixed_panel, evaluate_panel,
                         evaluate_counterfactual, evaluate_rows, full_answer_lp)
from pareto_selector import pareto_front

torch.set_num_threads(20)
DATA = HERE / "data"
SEED = 20261124


def load_model(run: dict, vocab_size: int, role="best"):
    ck = run["checkpoints"][role]["path"]
    blob = torch.load(ck, map_location="cpu", weights_only=False)
    model = build_model(run["architecture"], vocab_size)
    model.load_state_dict(blob["state_dict"], strict=True)
    model.eval()
    return model, blob, ck


def model_state_sha(model):
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def pick_examples(details, n=15):
    ok = [d for d in details if d["ok"]]
    bad = [d for d in details if not d["ok"]]
    n_ok = min(len(ok), math.ceil(n * 0.5))
    n_bad = min(len(bad), n - n_ok)
    return (ok[:n_ok] + bad[:n_bad])[:n]


def md_escape(s):
    return str(s).replace("\\", "\\\\").replace("|", "\\|").replace("\n", "<br>")


def main():
    t0 = time.perf_counter()
    tok = M1Tokenizer()
    train = load_rows("train")
    val = load_rows("val")
    sealed = load_rows("sealed")
    panel = build_fixed_panel(val, per_task=30, seed=SEED)
    results = json.loads((HERE / "M1_TRAINING_RESULTS.json").read_text(encoding="utf-8"))
    resource = json.loads((HERE / "M1_RESOURCE_BENCHMARK.json").read_text(encoding="utf-8"))["models"]
    audit = json.loads((HERE / "M1_DATA_AUDIT.json").read_text(encoding="utf-8"))

    stage1 = results["stage1"]
    stage2 = results["stage2"]

    # 代表模型：初筛四架构 best + 复验两架构 best
    reps = {}
    for r in stage1:
        reps[f"S1_{r['architecture']}"] = r
    for r in stage2:
        reps[r["run_id"]] = r

    eval_models = {}
    for key, run in reps.items():
        model, blob, ck = load_model(run, tok.vocab_size)
        ev = evaluate_panel(model, tok, panel, collect_details=True)
        details = ev.pop("details")
        sealed_ev = evaluate_panel(model, tok, sealed, collect_details=False)
        eval_models[key] = {"model": model, "run": run, "checkpoint": ck,
                            "blob": blob, "panel": ev, "sealed": sealed_ev,
                            "details": details,
                            "state_sha256": model_state_sha(model)}
        print(json.dumps({"phase": "eval_model", "key": key,
                          "panel_macro": ev["macro_accuracy"],
                          "sealed_macro": sealed_ev["macro_accuracy"]}, ensure_ascii=False), flush=True)

    # 复验两架构取两 seed 最好 checkpoint 的平均表现
    stage2_arch = {}
    for key, item in eval_models.items():
        if not key.startswith("S2_"):
            continue
        arch = item["run"]["architecture"]
        stage2_arch.setdefault(arch, []).append(item)
    arch_summary = {}
    for arch, items in stage2_arch.items():
        arch_summary[arch] = {
            "seeds": [i["run"]["seed"] for i in items],
            "validation_best_macro_mean": round(sum(i["run"]["best_score"] for i in items) / len(items), 4),
            "panel_macro_mean": round(sum(i["panel"]["macro_accuracy"] for i in items) / len(items), 4),
            "sealed_macro_mean": round(sum(i["sealed"]["macro_accuracy"] for i in items) / len(items), 4),
            "wall_mean": round(sum(i["run"]["wall_seconds"] for i in items) / len(items), 1),
            "params": items[0]["run"]["params"],
            "state_elements": items[0]["run"]["state_elements"],
        }

    best_new = results["selection"]["best_new"]
    best_ref = results["selection"]["best_reference"]
    new_item = sorted(stage2_arch[best_new], key=lambda x: x["run"]["best_score"], reverse=True)[0]
    ref_item = sorted(stage2_arch[best_ref], key=lambda x: x["run"]["best_score"], reverse=True)[0]
    compare_keys = {best_new: new_item, best_ref: ref_item}

    # 泛化：反事实、长度、未见格式、变量数、更新类
    cf = {}
    length_gen_rows = load_rows("length_gen")
    for arch, item in compare_keys.items():
        cf[arch] = evaluate_counterfactual(item["model"], tok, limit=400)
        lg_per = evaluate_rows(item["model"], tok, length_gen_rows)
        lg_ok = sum(d["ok"] for d in lg_per.values()); lg_n = sum(d["n"] for d in lg_per.values())
        lg_macro = round(sum(d["accuracy"] for d in lg_per.values()) / max(1, len(lg_per)), 4)
        cf[arch]["length_generalization"] = {"per_task": lg_per, "n": lg_n, "ok": lg_ok,
                                              "macro_accuracy": lg_macro}
        print(json.dumps({"phase": "generalization", "arch": arch,
                          "cf": cf[arch]["paired_accuracy"],
                          "length": lg_macro}, ensure_ascii=False), flush=True)

    sealed_per_arch = {a: i["sealed"]["per_task"] for a, i in compare_keys.items()}
    generalization = {
        "status": "M1_GENERALIZATION_DONE",
        "counterfactual": cf,
        "sealed_by_task": sealed_per_arch,
        "unseen_format": {a: i["sealed"]["per_task"].get("A1_copy1_unseen_fmt")
                          for a, i in compare_keys.items()},
        "variable_rename": {a: i["sealed"]["per_task"].get("B_rename")
                            for a, i in compare_keys.items()},
        "shortcut_baselines_train": audit["baselines_by_task_train"],
        "length_train_range": audit["length_train_range"],
        "length_test_range": audit["length_test_range"],
    }

    # 干扰/距离与记忆更新
    memory = {"status": "M1_MEMORY_INTERFERENCE_DONE", "models": {}}
    for arch, item in compare_keys.items():
        dist_rows = ([r for r in sealed if r["task"] == "E_distance"] +
                     [r for r in length_gen_rows if r["task"] == "E_distance"])
        by_dist = defaultdict(lambda: {"n": 0, "ok": 0})
        for r in dist_rows:
            d = r["meta"].get("distance")
            g = __import__("evaluate_m1").generate(item["model"], tok, r["prompt"])
            ok = g["text"].strip() == str(r["answer"])
            by_dist[d]["n"] += 1; by_dist[d]["ok"] += int(ok)
        memory["models"][arch] = {
            "distance_curve": {str(d): {"n": v["n"], "ok": v["ok"],
                                        "accuracy": round(v["ok"] / max(1, v["n"]), 4)}
                               for d, v in sorted(by_dist.items())},
            "binding_by_variable_count": {t: sealed_per_arch[arch].get(t)
                                          for t in ("B1_bind1", "B2_bind", "B4_bind", "B8_bind")},
            "update_tasks": {t: sealed_per_arch[arch].get(t)
                             for t in ("B_swap", "B2_query_switch", "C_reassign", "D_indirect")},
        }

    # Pareto
    pareto_records = {}
    for arch, s in arch_summary.items():
        pareto_records[arch] = {
            "accuracy": s["panel_macro_mean"],
            "counterfactual": cf[arch]["paired_accuracy"],
            "retention": memory["models"][arch]["distance_curve"].get("16", {}).get("accuracy", 0.0),
            "trainable_params": s["params"],
            "state_elements": s["state_elements"],
            "cpu_latency_ms": resource[arch]["forward_batch_ms"],
            "train_time": s["wall_mean"],
        }
    pareto = pareto_front(pareto_records)
    architecture_comparison = {
        "status": "M1_ARCHITECTURE_COMPARISON_DONE",
        "stage1_single_seed_128_steps": [
            {"architecture": r["architecture"], "best_val_macro": r["best_score"],
             "best_step": r["best_step"], "params": r["params"],
             "state_elements": r["state_elements"],
             "panel_macro": eval_models[f"S1_{r['architecture']}"]["panel"]["macro_accuracy"],
             "sealed_macro": eval_models[f"S1_{r['architecture']}"]["sealed"]["macro_accuracy"]}
            for r in stage1],
        "stage2_two_seed_384_steps": arch_summary,
        "selection": results["selection"],
        "pareto": {"records": pareto_records, "front": pareto},
        "checkpoint_integrity": {k: {"path": v["checkpoint"], "state_sha256": v["state_sha256"],
                                     "saved_step": v["blob"]["optimizer_steps_executed"],
                                     "seed": v["run"]["seed"]}
                                 for k, v in compare_keys.items()},
        "model_weights_uploaded": False,
    }

    evaluation = {
        "status": "M1_UNIFIED_EVALUATION_DONE",
        "protocol": {"fixed_panel_n": len(panel), "sealed_n": len(sealed),
                     "free_generation": True, "same_order": True,
                     "selection_split": "validation", "sealed_for_final_only": True},
        "models": {k: {"architecture": v["run"]["architecture"],
                       "run_id": v["run"]["run_id"], "seed": v["run"]["seed"],
                       "params": v["run"]["params"], "state_elements": v["run"]["state_elements"],
                       "checkpoint": v["checkpoint"], "state_sha256": v["state_sha256"],
                       "panel": v["panel"], "sealed": v["sealed"]}
                   for k, v in eval_models.items()},
        "full_answer_logprob_example": {
            "prompt": "COPY 12", "answer": "12",
            "new_architecture_lp": round(full_answer_lp(new_item["model"], tok, "COPY 12", "12"), 4),
            "reference_lp": round(full_answer_lp(ref_item["model"], tok, "COPY 12", "12"), 4),
            "note": "辅助指标；只含 answer+EOS，不含 prompt 概率"},
    }

    # 45 组真实输入输出
    gen_rows = []
    for arch, item in compare_keys.items():
        for d in pick_examples(item["details"], 22):
            gen_rows.append({"architecture": arch, "run_id": item["run"]["run_id"],
                             "task": d["task"], "prompt": d["prompt"],
                             "expected": d["expected"], "predicted": d["predicted"],
                             "ok": d["ok"], "finish": d["finish"], "tokens": d["tokens"]})
    (HERE / "M1_GENERATION.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in gen_rows) + "\n", encoding="utf-8")

    (HERE / "M1_EVALUATION.json").write_text(json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8")
    (HERE / "M1_GENERALIZATION.json").write_text(json.dumps(generalization, ensure_ascii=False, indent=2), encoding="utf-8")
    (HERE / "M1_MEMORY_INTERFERENCE.json").write_text(json.dumps(memory, ensure_ascii=False, indent=2), encoding="utf-8")
    (HERE / "M1_ARCHITECTURE_COMPARISON.json").write_text(
        json.dumps(architecture_comparison, ensure_ascii=False, indent=2), encoding="utf-8")

    write_report(evaluation, architecture_comparison, generalization, memory,
                 resource, results, gen_rows, pareto)
    print(json.dumps({"phase": "done", "seconds": round(time.perf_counter() - t0, 1),
                      "best_new": best_new, "best_ref": best_ref,
                      "generation_rows": len(gen_rows)}, ensure_ascii=False), flush=True)


def write_report(ev, comp, gen, memory, resource, results, gen_rows, pareto):
    s1 = {r["architecture"]: r for r in comp["stage1_single_seed_128_steps"]}
    s2 = comp["stage2_two_seed_384_steps"]
    lines = []
    lines.append("# N-MIND M1：数学驱动全新神经架构研究报告")
    lines.append("")
    lines.append("## 1. 结论摘要")
    lines.append("")
    lines.append(f"- 真实执行：初筛 4×128 + 复验 2 架构×2种子×384 = **{results['budget']['steps_used']}/2048 step**，"
                 f"墙钟 **{results['budget']['wall_used']}s**，未超 1800s。")
    lines.append(f"- 初筛：M0={s1['M0']['best_val_macro']}，M1-A={s1['M1-A']['best_val_macro']}，"
                 f"M1-B={s1['M1-B']['best_val_macro']}，M1-C={s1['M1-C']['best_val_macro']}；"
                 f"入围复验：新架构 **{comp['selection']['best_new']}**、参考 **{comp['selection']['best_reference']}**。")
    lines.append(f"- 复验验证宏平均：M1-B={s2.get('M1-B', {}).get('validation_best_macro_mean')}，"
                 f"M0={s2.get('M0', {}).get('validation_best_macro_mean')}。")
    lines.append("- 矩阵记忆在新任务上给出初步优势，但复制、反转、重赋值与长距离目标均未达到验收线；"
                 "**尚不能宣称新架构具备继续学习复杂编程任务的充分依据**。")
    lines.append("")
    lines.append("## 2. 来源与独立性")
    lines.append("")
    lines.append("- 来源：Gated Delta Networks (arXiv:2412.06464)、Gated DeltaNet-2 (arXiv:2605.22791，"
                 "官方代码 NVIDIA NC 许可证，未复制)、Kalman Delta Networks (arXiv:2609.07816)、"
                 "固定状态关联回忆论文 (arXiv:2609.16183)、AlphaEvolve (Apache-2.0，仅借鉴评价/淘汰思想)。")
    lines.append("- 本项目独立实现四个小模型、tokenizer、数据、验证器、训练器与评测；详见 "
                 "`M1_RESEARCH_SOURCES.md` 与 `M1_MATHEMATICAL_SPEC.md`。")
    lines.append("- 不确定性记忆与短因果卷积仅通过算子验证；**未加入首轮主模型，标记 NOT_RUN**。")
    lines.append("")
    lines.append("## 3. 数学验证与资源")
    lines.append("")
    lines.append("- 数学验证 12/12 PASS：维度、因果、gradcheck、有限值、极端输入、零输入、重复写入、"
                 "增量一致、批次一致、样本隔离、checkpoint、确定性性质。")
    lines.append("- 验证过程中修复两个真实错误：批量标志反转、残差未基于衰减后状态计算；未通过放松测试掩盖。")
    lines.append("")
    lines.append("| 架构 | 参数 | 状态元素 | 前向 batch ms | 训练 tok/s |")
    lines.append("|---|---:|---:|---:|---:|")
    for name, d in resource.items():
        lines.append(f"| {name} | {d['trainable_params']} | {d['state_elements']} | "
                     f"{d['forward_batch_ms']} | {d['train_tokens_per_second']} |")
    lines.append("")
    lines.append("## 4. 数据与协议")
    lines.append("")
    lines.append("- train/val/sealed = 9505/1386/1452；反事实 400 对；长度泛化 2204 条。")
    lines.append("- 跨语义组重合 0；C1.0/C1.1 历史 prompt 重合 0；训练长度 2–4，测试长度 5–6；"
                 "训练距离 0/4/8，测试距离 16/32/64。")
    lines.append("- 主指标为自然 EOS 的自由生成；完整答案 lp 只含 answer+EOS；立即 EOS 单独统计。")
    lines.append("")
    lines.append("## 5. 架构对比")
    lines.append("")
    lines.append("| 架构 | 初筛验证 | 复验验证均值 | 封存集均值 | 参数 | 状态 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for name in ("M0", "M1-A", "M1-B", "M1-C"):
        a = s1[name]["best_val_macro"]
        b = s2.get(name, {}).get("validation_best_macro_mean", "NOT_RUN")
        c = s2.get(name, {}).get("sealed_macro_mean", "NOT_RUN")
        p = s1[name]["params"]; st = s1[name]["state_elements"]
        lines.append(f"| {name} | {a} | {b} | {c} | {p} | {st} |")
    lines.append("")
    lines.append(f"- Pareto 非支配集合：{pareto['front']}。")
    lines.append("- 注意：初筛只提供早期学习信号，不能替代双种子复验；GRU 未进入第二阶段，"
                 "不能拿 128 步 GRU 与 384 步复验模型直接比较。")
    lines.append("")
    lines.append("## 6. 变量数量、干扰与长度泛化")
    lines.append("")
    for arch, item in gen["counterfactual"].items():
        lg = item["length_generalization"]
        lines.append(f"- {arch}: 反事实配对 {item['ok']}/{item['pairs']} ({item['paired_accuracy']})；"
                     f"长度泛化 {lg['ok']}/{lg['n']} ({lg['macro_accuracy']})。")
    for arch, m in memory["models"].items():
        curve = ", ".join(f"{d}:{v['accuracy']}" for d, v in m["distance_curve"].items())
        lines.append(f"- {arch} 距离曲线: {curve}。")
    lines.append("")
    lines.append(f"## 7. {len(gen_rows)} 组真实输入输出（含失败，未用解释器纠正）")
    lines.append("")
    lines.append("| 架构 | 任务 | Prompt | 期望 | 原始输出 | 正确 | 结束 |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in gen_rows:
        lines.append(f"| {r['architecture']} | {r['task']} | {md_escape(r['prompt'])} | "
                     f"{md_escape(r['expected'])} | {md_escape(r['predicted'])} | "
                     f"{'是' if r['ok'] else '否'} | {r['finish']} |")
    lines.append("")
    lines.append("## 8. 最终问题回答")
    lines.append("")
    for i, a in enumerate(build_answers(s1, s2, gen, memory, resource, pareto), 1):
        lines.append(f"{i}. **{a}**")
        lines.append("")
    lines.append("## 9. 下一步")
    lines.append("")
    lines.append("- 对短因果卷积、独立门、query/key 归一化、不确定性写入分别做单变量消融。")
    lines.append("- 复制/反转失败需专门的位置与序列读出机制，不得靠加步数掩盖。")
    lines.append("- 距离课程有效的因果主张需要同架构课程消融；本轮所有模型共用课程，不能把提升归因于课程本身。")
    (HERE / "M1_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def build_answers(s1, s2, gen, memory, resource, pareto):
    b = s2.get("M1-B", {}); m0 = s2.get("M0", {})
    new_binding = gen["sealed_by_task"].get("M1-B", {})
    old_binding = gen["sealed_by_task"].get("M0", {})
    g = lambda task: new_binding.get(task, {}).get("accuracy", 0.0)
    dist = memory["models"].get("M1-B", {}).get("distance_curve", {})
    return [
        (f"N-Cell 初筛 {s1['M0']['best_val_macro']}、GRU {s1['M1-A']['best_val_macro']}，"
         f"学习效率接近；GRU 训练吞吐 {resource['M1-A']['train_tokens_per_second']} "
         f"高于 N-Cell {resource['M0']['train_tokens_per_second']}。复验阶段 GRU NOT_RUN，"
         f"因此不能得出最终的架构优越性结论。"),
        (f"有初步证据。M1-B 双种子复验验证均值 {b.get('validation_best_macro_mean')} > "
         f"M0 {m0.get('validation_best_macro_mean')}；封存集双变量绑定 "
         f"{g('B2_bind')} vs 参考 {old_binding.get('B2_bind', {}).get('accuracy', 0.0)}，"
         f"四变量 {g('B4_bind')}，但仍低于 90%/80% 验收线。"),
        (f"没有获得支持。M1-C 初筛 {s1['M1-C']['best_val_macro']} 低于 M1-B "
         f"{s1['M1-B']['best_val_macro']}，未入围复验；独立门可能需更长训练或不同初始化，"
         f"本轮不能宣称减少干扰。"),
        (f"不能连续生成多个正确符号：多字符复制 {g('A2_copyN')}、反转 {g('A4_reverse')}、"
         f"重复 {g('A5_repeat')}。"),
        (f"变量数增加后下降：B1={g('B1_bind1')}、B2={g('B2_bind')}、"
         f"B4={g('B4_bind')}、B8={g('B8_bind')}。"),
        (f"有限保持。M1-B 距离 0/4/8/16/32/64 的封存准确率为 "
         f"{', '.join(str(dist.get(str(d), {}).get('accuracy', 0.0)) for d in (0,4,8,16,32,64))}；"
         f"重赋值 {g('C_reassign')}、间接引用 {g('D_indirect')}。"),
        (f"没有真正的长度泛化证据：长度 5–6 的独立测试 M1-B "
         f"{gen['counterfactual']['M1-B']['length_generalization']['macro_accuracy']}，"
         f"M0 {gen['counterfactual']['M0']['length_generalization']['macro_accuracy']}。"),
        (f"Pareto 非支配集合为 {pareto['front']}。M1-B 在准确率/反事实上更优，"
         f"N-Cell 状态更小但慢；GRU 最快但初筛未领先，资源效率结论受短训练限制。"),
        ("获得初步支持的机制：共享门 Delta 矩阵关联记忆在同等双种子复验中优于历史 N-Cell；"
         "数学算子的有限性、因果性、梯度与状态隔离获得验证器支持。"),
        ("未达预期的机制：独立擦除/写入门、多字符复制、短序列反转、重赋值、"
         "长距离干扰与长度外推；不确定性记忆与短卷积本轮 NOT_RUN，不能算失败也不能算成功。"),
        ("适合下一轮独立消融：短因果卷积核 1/2/4/8、独立门更长训练、对角不确定性写入、"
         "query/key 归一化、读出机制、距离课程本身。每次只能改变一个变量。"),
        ("仅有有限依据：M1-B 显示出比 N-Cell 更好的绑定学习斜率与复验均值，"
         "但所有验收门槛未达，暂不支持直接进入复杂编程能力训练。"),
    ]


if __name__ == "__main__":
    main()
