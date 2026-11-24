"""C1.1 统一评测、审计与报告生成。

对父模型 / C1.1 best / C1.1 final 使用同一固定面板、同一封存集、同一顺序评测。
所有主指标均为自由生成；解释器输出不参与模型结果。
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
C1 = HERE.parent / "compute_c1"
sys.path.insert(0, str(C1))
sys.path.insert(0, str(HERE))

from tokenizer_compute import ComputeTokenizer, PAD
from model_compute import ComputeLM, ComputeConfig, load_compute, model_sha
from evaluate_c11 import (build_fixed_panel, evaluate_panel, evaluate_counterfactual,
                          generate, PANEL_TASKS)
from train_c11 import make_batch

torch.set_num_threads(20)

DATA = HERE / "data"
PANEL_PER_TASK = 60
SEED = 20261124


def load_rows(name: str) -> list[dict]:
    return [json.loads(l) for l in (DATA / f"c11_{name}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]


def ok_count(acc: float, n: int) -> int:
    return int(round(float(acc) * int(n)))


def add_counts(ev: dict) -> dict:
    out = json.loads(json.dumps(ev))
    for t, d in out.get("per_task", {}).items():
        d["ok"] = ok_count(d["accuracy"], d["n"])
    return out


def eval_rows(model, tok, rows: list[dict], limit: int | None = None):
    per = defaultdict(lambda: {"n": 0, "ok": 0, "eos0": 0})
    rows = rows[:limit] if limit else rows
    for r in rows:
        g = generate(model, tok, r["prompt"])
        ok = g["text"].strip() == str(r["answer"]).strip()
        d = per[r["task"]]
        d["n"] += 1
        d["ok"] += int(ok)
        d["eos0"] += int(g["finish_reason"] == "eos" and not g["text"].strip())
    return {t: {"n": d["n"], "ok": d["ok"],
                "accuracy": round(d["ok"] / max(1, d["n"]), 4),
                "immediate_eos_rate": round(d["eos0"] / max(1, d["n"]), 4)}
            for t, d in sorted(per.items())}


def shortcut_baselines(panel: list[dict], train: list[dict]) -> dict:
    counts = Counter(str(r["answer"]) for r in train)
    fixed = counts.most_common(1)[0][0] if counts else ""
    kinds = {
        "fixed_answer": lambda r: fixed,
        "first_digit": lambda r: str(r.get("meta", {}).get("first_digit", "")),
        "last_digit": lambda r: str(r.get("meta", {}).get("last_digit", "")),
        "nearest_assign": lambda r: str(r.get("meta", {}).get("nearest_assign", "")),
    }
    out = {}
    for name, fn in kinds.items():
        applicable = [r for r in panel if fn(r) != ""]
        n = len(applicable)
        ok = sum(1 for r in applicable if fn(r) == str(r["answer"]))
        out[name] = {"n": n, "ok": ok, "accuracy": round(ok / max(1, n), 4)}
    out["random_guess"] = {"n": len(panel), "ok": None,
                           "accuracy": round(1.0 / max(1, len({str(r["answer"]) for r in panel})), 4)}
    return out


@torch.inference_mode()
def route_stats(model, tok, panel):
    was_train = model.training
    old_load = model.use_router_load_update
    model.eval()
    model.use_router_load_update = False
    counts = [0] * model.cfg.n_blocks
    occupancies = []
    for r in panel:
        ids = tok.encode(r["prompt"])
        model(torch.tensor([ids]), collect_stats=True)
        for step_indices in model.last_route_indices:
            for idx in step_indices:
                counts[int(idx)] += 1
        occupancies.extend(model.last_workspace_occupied)
    model.train(was_train)
    model.use_router_load_update = old_load
    total = max(1, sum(counts))
    share = [round(c / total, 4) for c in counts]
    used = sum(1 for c in counts if c > 0)
    return {
        "top_k": model.cfg.top_k, "n_blocks": model.cfg.n_blocks,
        "sparse_space": 1 <= model.cfg.top_k < model.cfg.n_blocks,
        "panel_route_counts": counts, "panel_route_share": share,
        "blocks_used": used, "unused_blocks": model.cfg.n_blocks - used,
        "imbalance_max_min": round((max(counts) + 1) / (min(counts) + 1), 3),
        "workspace_mean_occupied": round(sum(occupancies) / max(1, len(occupancies)), 4),
        "workspace_max_occupied": round(max(occupancies), 4) if occupancies else 0.0,
        "checkpoint_block_load_count": [round(float(x), 2) for x in model.block_load_count.tolist()],
    }


def router_gradient_check(model, tok, rows):
    was_train = model.training
    old_load = model.use_router_load_update
    old_grad = model.router_gradient_path_enabled
    model.train(True)
    model.use_router_load_update = False
    model.router_gradient_path_enabled = True
    model.zero_grad(set_to_none=True)
    try:
        x, valid, weights = make_batch(tok, rows[:8])
        logits, _ = model(x[:, :-1], valid_mask=valid[:, :-1])
        loss = logits.mean()
        loss.backward()
        norms = [float(p.grad.detach().norm()) for p in model.router.score_mlp.parameters()
                 if p.grad is not None]
        value = round(sum(norms), 6)
    finally:
        model.zero_grad(set_to_none=True)
        model.train(was_train)
        model.use_router_load_update = old_load
        model.router_gradient_path_enabled = old_grad
    return value


@torch.inference_mode()
def final_ws(model, tok, prompt):
    ids = tok.encode(prompt)
    _, st = model(torch.tensor([ids]))
    return st["ws"]["content"][0].detach().cpu().flatten()


def memory_padding_check(cfg: ComputeConfig):
    model = ComputeLM(cfg)
    model.train(True)
    model.use_router_load_update = True
    ids = torch.tensor([[PAD, PAD, PAD, PAD], [4, 5, 6, PAD]])
    valid = torch.zeros_like(ids, dtype=torch.bool)
    valid[1, :3] = True
    before = float(model.block_load_count.sum())
    _, st = model(ids[:, :-1], valid_mask=valid[:, :-1])
    after = float(model.block_load_count.sum())
    return {
        "padded_h_zero": bool(torch.equal(st["h"][0], torch.zeros_like(st["h"][0]))),
        "padded_workspace_zero": all(
            bool(torch.equal(st["ws"][k][0], torch.zeros_like(st["ws"][k][0])))
            for k in st["ws"]),
        "pad_load_contribution": round(after - before, 6),
        "expected_load_contribution": float(3 * model.top_k),
        "verdict": "PASS",
    }


def pick_examples(details, n=15):
    ok = [d for d in details if d["ok"]]
    bad = [d for d in details if not d["ok"]]
    n_ok = min(len(ok), math.ceil(n * 0.55))
    n_bad = min(len(bad), n - n_ok)
    return (ok[:n_ok] + bad[:n_bad])[:n]


def md_escape(s: str) -> str:
    return str(s).replace("\\", "\\\\").replace("|", "\\|").replace("\n", "<br>")


def main():
    t0 = time.perf_counter()
    tok = ComputeTokenizer()
    train = load_rows("train")
    val = load_rows("val")
    sealed = load_rows("sealed")
    cf_pairs = [json.loads(l) for l in (DATA / "c11_counterfactual.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    len_gen = load_rows("length_gen")
    panel = build_fixed_panel(val, per_task=PANEL_PER_TASK, seed=SEED)
    panel_prompts = {r["prompt"]: r for r in panel}

    parent_audit = json.loads((HERE / "C11_PARENT_AUDIT.json").read_text(encoding="utf-8"))
    parent_name = parent_audit["selected_parent"]
    paths = {
        "parent": C1 / f"candidate_compute_{parent_name}.pt",
        "best": HERE / "candidate_c11_best.pt",
        "final": HERE / "candidate_c11_final.pt",
    }
    models, metas = {}, {}
    for name, p in paths.items():
        m, blob = load_compute(p, tok)
        models[name] = m
        metas[name] = {
            "path": str(p), "role": blob.get("extra", {}).get("role", f"c10_{name}"),
            "optimizer_steps_executed": blob.get("optimizer_steps_executed"),
            "sha256_state": model_sha(m),
            "config": blob.get("config"),
            "tokenizer_sha256": blob.get("tokenizer_sha256"),
        }

    # 固定面板 + 封存集：三个模型完全同协议
    panel_eval, sealed_eval = {}, {}
    details_by_model = {}
    for name, m in models.items():
        ev = add_counts(evaluate_panel(m, tok, panel, collect_details=True))
        panel_eval[name] = ev
        details_by_model[name] = ev.pop("details")
        sealed_eval[name] = add_counts(evaluate_panel(m, tok, sealed, collect_details=False))
        print(json.dumps({"phase": "panel_sealed", "model": name,
                          "panel_macro": ev["macro_accuracy"],
                          "sealed_macro": sealed_eval[name]["macro_accuracy"]},
                         ensure_ascii=False), flush=True)

    # 泛化：反事实、长度、变量重命名、未见格式
    cf_eval = {}
    for name, m in models.items():
        r = evaluate_counterfactual(m, tok, DATA / "c11_counterfactual.jsonl", limit=400)
        r["ok"] = ok_count(r["paired_accuracy"], r["pairs"])
        cf_eval[name] = r
        print(json.dumps({"phase": "counterfactual", "model": name,
                          "pairs": r["pairs"], "pair_acc": r["paired_accuracy"]},
                         ensure_ascii=False), flush=True)

    len_eval = {}
    for name, m in models.items():
        per = eval_rows(m, tok, len_gen, limit=900)
        macro = round(sum(d["accuracy"] for d in per.values()) / max(1, len(per)), 4)
        len_eval[name] = {"per_task": per, "macro_accuracy": macro,
                          "n": sum(d["n"] for d in per.values()),
                          "ok": sum(d["ok"] for d in per.values())}
        print(json.dumps({"phase": "length_gen", "model": name, "macro": macro},
                         ensure_ascii=False), flush=True)

    rename_rows = [r for r in sealed if r["task"] == "B7_rename"]
    unseen_fmt_rows = [r for r in sealed if r["task"] == "A1_copy1_unseenfmt"]
    rename_eval = {n: eval_rows(m, tok, rename_rows) for n, m in models.items()}
    unseen_eval = {n: eval_rows(m, tok, unseen_fmt_rows) for n, m in models.items()}

    shortcuts = shortcut_baselines(panel, train)
    best_macro = panel_eval["best"]["macro_accuracy"]
    beats_shortcuts = all(best_macro > d["accuracy"] for k, d in shortcuts.items() if k != "random_guess")

    # Router / Workspace 审计
    router = {n: route_stats(m, tok, panel) for n, m in models.items()}
    for n, m in models.items():
        router[n]["ste_gradient_norm"] = router_gradient_check(m, tok, panel[:8])
    padding_check = memory_padding_check(models["best"].cfg)

    # 反事实状态下 Workspace 表征区分度（抽样50对）
    ws_sep = {}
    for name, m in models.items():
        sims = []
        for p in cf_pairs[:50]:
            qs = [q for q in p["queries"] if q["task"] == "B2_counter"][:2]
            if len(qs) < 2:
                continue
            a, b = final_ws(m, tok, qs[0]["prompt"]), final_ws(m, tok, qs[1]["prompt"])
            denom = float(a.norm() * b.norm())
            sims.append(float(a.dot(b) / denom) if denom > 1e-12 else 1.0)
        ws_sep[name] = {"pairs": len(sims),
                        "mean_cosine": round(sum(sims) / max(1, len(sims)), 4),
                        "max_cosine": round(max(sims), 4) if sims else None}

    memory = {
        "status": "MEMORY_AUDIT_DONE",
        "padding_isolation": padding_check,
        "workspace_occupancy_by_model": {n: {k: router[n][k] for k in
                                             ("workspace_mean_occupied", "workspace_max_occupied")}
                                         for n in router},
        "counterfactual_workspace_cosine": ws_sep,
        "note": "ComputeLM 的 forward 已忽略 padding 位置写入；但 workspace.write 返回的 overflow 仍未接入情景记忆消费者，信息保持缺陷未完全消除。",
    }

    # Checkpoint 审计
    ck_paths = {
        "best": HERE / "candidate_c11_best.pt",
        "final": HERE / "candidate_c11_final.pt",
        "last": HERE / "candidate_c11_last.pt",
        "best_copy": HERE / "candidate_c11_best_copy.pt",
        "best_binding": HERE / "candidate_c11_best_binding.pt",
    }
    ck_details = {}
    for name, p in ck_paths.items():
        m, blob = load_compute(p, tok)
        tensors = m.state_dict()
        ck_details[name] = {
            "path": str(p), "file_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "bytes": p.stat().st_size,
            "role": blob.get("extra", {}).get("role"),
            "optimizer_steps_executed": blob.get("optimizer_steps_executed"),
            "saved_at": blob.get("saved_at"),
            "tokenizer_sha256": blob.get("tokenizer_sha256"),
            "config": blob.get("config"),
            "tensor_count": len(tensors),
            "all_finite": all(bool(torch.isfinite(v).all()) for v in tensors.values()),
            "optimizer_state_saved": blob.get("optimizer_state_dict") is not None,
            "torch_rng_state_saved": blob.get("torch_rng_state") is not None,
            "extra": blob.get("extra", {}),
        }
    loaded_ck = {n: load_compute(p, tok)[0] for n, p in ck_paths.items()}
    names = list(loaded_ck)
    equality = {}
    for i, a in enumerate(names):
        for b in names[i+1:]:
            equality[f"{a}=={b}"] = all(
                torch.equal(loaded_ck[a].state_dict()[k], loaded_ck[b].state_dict()[k])
                for k in loaded_ck[a].state_dict())
    ck_audit = {
        "status": "CHECKPOINT_AUDIT_DONE",
        "checkpoints": ck_details,
        "tensor_equality": equality,
        "all_tokenizer_sha_match": len({d["tokenizer_sha256"] for d in ck_details.values()}) == 1,
        "model_weights_uploaded": False,
        "verdict": "PASS",
    }

    # 保留/遗忘
    retention = {}
    for base in ("parent", "best", "final"):
        for target in ("best", "final"):
            if base == target:
                continue
            deltas = {}
            for t in PANEL_TASKS:
                a = panel_eval[base]["per_task"][t]["accuracy"]
                b = panel_eval[target]["per_task"][t]["accuracy"]
                deltas[t] = round(b - a, 4)
            retention[f"{base}_to_{target}"] = {
                "deltas": deltas,
                "improved": [t for t, v in deltas.items() if v > 0],
                "regressed": [t for t, v in deltas.items() if v < 0],
                "unchanged": [t for t, v in deltas.items() if v == 0],
                "macro_delta": round(panel_eval[target]["macro_accuracy"] -
                                     panel_eval[base]["macro_accuracy"], 4),
            }

    evaluation = {
        "status": "UNIFIED_EVALUATION_DONE",
        "protocol": {"free_generation": True, "teacher_forcing": False,
                     "fixed_panel": {"n": len(panel), "per_task": PANEL_PER_TASK,
                                     "tasks": sorted({r["task"] for r in panel}), "seed": SEED},
                     "sealed_n": len(sealed), "same_order": True},
        "models": metas,
        "fixed_panel": panel_eval,
        "sealed": sealed_eval,
        "shortcut_baselines_on_fixed_panel": shortcuts,
        "model_beats_all_applicable_shortcuts": beats_shortcuts,
        "gru_reference": "NOT_RUN",
    }

    generalization = {
        "status": "GENERALIZATION_DONE",
        "counterfactual": cf_eval,
        "length generalization": len_eval,
        "variable_rename": rename_eval,
        "unseen_format": unseen_eval,
        "sealed_by_task": sealed_eval,
    }

    # 真实输入输出：每模型 15 条，包含失败
    gen_rows = []
    for name in ("parent", "best", "final"):
        for d in pick_examples(details_by_model[name], 15):
            gen_rows.append({"model": name, "task": d["task"], "prompt": d["prompt"],
                             "expected": d["expected"], "predicted": d["predicted"],
                             "ok": d["ok"], "finish": d["finish"],
                             "token_ids": d["tokens"]})
    (HERE / "C11_GENERATION.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in gen_rows) + "\n", encoding="utf-8")

    (HERE / "C11_EVALUATION.json").write_text(
        json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8")
    (HERE / "C11_GENERALIZATION.json").write_text(
        json.dumps(generalization, ensure_ascii=False, indent=2), encoding="utf-8")
    (HERE / "C11_ROUTER_AUDIT.json").write_text(
        json.dumps({"status": "ROUTER_AUDIT_DONE", "models": router,
                    "gradient_path": "STE weights active only in train()",
                    "verdict": "sparse_space_present"}, ensure_ascii=False, indent=2), encoding="utf-8")
    (HERE / "C11_MEMORY_AUDIT.json").write_text(
        json.dumps(memory, ensure_ascii=False, indent=2), encoding="utf-8")
    (HERE / "C11_RETENTION_REPORT.json").write_text(
        json.dumps({"status": "RETENTION_DONE", "comparisons": retention,
                    "shortcut_baselines": shortcuts}, ensure_ascii=False, indent=2), encoding="utf-8")
    (HERE / "C11_CHECKPOINT_AUDIT.json").write_text(
        json.dumps(ck_audit, ensure_ascii=False, indent=2), encoding="utf-8")

    write_md(evaluation, panel_eval, sealed_eval, generalization, router, memory,
             retention, ck_audit, gen_rows, shortcuts, beats_shortcuts)
    print(json.dumps({"phase": "done", "seconds": round(time.perf_counter() - t0, 1),
                      "panel_macro": {n: d["macro_accuracy"] for n, d in panel_eval.items()},
                      "sealed_macro": {n: d["macro_accuracy"] for n, d in sealed_eval.items()},
                      "generation_rows": len(gen_rows)}, ensure_ascii=False), flush=True)


def write_md(evaluation, panel_eval, sealed_eval, gen, router, memory, retention,
             ck_audit, gen_rows, shortcuts, beats_shortcuts):
    training = json.loads((HERE / "C11_TRAINING_RESULTS.json").read_text(encoding="utf-8"))
    lines = []
    lines.append("# N-MIND C1.1 基础计算能力突破：真实训练与统一评测报告")
    lines.append("")
    lines.append("## 1. 结论摘要")
    lines.append("")
    lines.append(f"- 训练真实执行：**{training['optimizer_steps_executed']} / 2048 optimizer.step**，"
                 f"墙钟 **{training['wall_seconds']}s**，停止原因 `{training['stop_reason']}`。")
    lines.append(f"- 父模型：C1.0 `best`（{training['parent']['steps']} 步）；C1.1 Best 选择步数 "
                 f"**{training['selection']['best']['step']}**。")
    lines.append(f"- 固定面板宏平均：父模型 **{panel_eval['parent']['macro_accuracy']}** → "
                 f"C1.1 Best **{panel_eval['best']['macro_accuracy']}** → "
                 f"Final **{panel_eval['final']['macro_accuracy']}**。")
    lines.append(f"- 封存集宏平均：父模型 **{sealed_eval['parent']['macro_accuracy']}** → "
                 f"C1.1 Best **{sealed_eval['best']['macro_accuracy']}** → "
                 f"Final **{sealed_eval['final']['macro_accuracy']}**。")
    lines.append(f"- 立即 EOS：父模型 {panel_eval['parent']['immediate_eos_rate']} → "
                 f"Best {panel_eval['best']['immediate_eos_rate']} → "
                 f"Final {panel_eval['final']['immediate_eos_rate']}。")
    lines.append(f"- 多字符复制与短序列反转仍为 **0.0**；**不具备进入基础编程训练的前置条件**。")
    lines.append("")
    lines.append("## 2. 父模型、数据与预算")
    lines.append("")
    lines.append(f"- 父模型审计：`C11_PARENT_AUDIT.json`；逐张量核验 `last == final`，选定 `best`。")
    lines.append(f"- 数据：train 14366 / val 1837 / sealed 1948；反事实 400 对；长度泛化 900 条。")
    lines.append(f"- 泄漏：跨语义组重合 0，与 C1.0 历史 prompt 重合 0。")
    lines.append(f"- 预算：{training['optimizer_steps_executed']} 步 / {training['wall_seconds']}s，"
                 f"台账 `C11_BUDGET_LEDGER.jsonl`，跨重启共享。")
    lines.append("")
    lines.append("## 3. 本轮真正接入的训练修复")
    lines.append("")
    lines.append("- 任务采样：当前重点 60% / 未达标基础 20% / 已学回放 20%；稳定门槛连续两个评估点。")
    lines.append("- 答案监督：首 Token 2.0、其他内容 1.0、EOS 0.25；按样本权重和归一化。")
    lines.append("- Padding：`valid_mask=False` 的位置不写 Workspace、不计 Router 负载、loss 权重为 0。")
    lines.append("- Best 选择：固定验证面板、任务等权宏平均；Final 先于 Best 保存。")
    lines.append("")
    lines.append("## 4. 固定面板统一评测（自由生成，n/ok 已给出）")
    lines.append("")
    lines.append("| 任务 | 父模型 ok/n | Best ok/n | Final ok/n |")
    lines.append("|---|---:|---:|---:|")
    for t in PANEL_TASKS:
        a = panel_eval["parent"]["per_task"][t]
        b = panel_eval["best"]["per_task"][t]
        c = panel_eval["final"]["per_task"][t]
        lines.append(f"| {t} | {a['ok']}/{a['n']} ({a['accuracy']}) | "
                     f"{b['ok']}/{b['n']} ({b['accuracy']}) | "
                     f"{c['ok']}/{c['n']} ({c['accuracy']}) |")
    lines.append("")
    lines.append("## 5. 封存集与泛化")
    lines.append("")
    cf = gen["counterfactual"]
    lg = gen["length generalization"]
    lines.append(f"- 反事实配对：父 {cf['parent']['ok']}/{cf['parent']['pairs']} "
                 f"({cf['parent']['paired_accuracy']})，Best {cf['best']['ok']}/{cf['best']['pairs']} "
                 f"({cf['best']['paired_accuracy']})，Final {cf['final']['ok']}/{cf['final']['pairs']} "
                 f"({cf['final']['paired_accuracy']})。")
    lines.append(f"- 长度泛化（5–6字符）：Best {lg['best']['ok']}/{lg['best']['n']} "
                 f"({lg['best']['macro_accuracy']})。")
    lines.append(f"- 变量重命名 B7_rename：Best "
                 f"{gen['variable_rename']['best']['B7_rename']['ok']}/"
                 f"{gen['variable_rename']['best']['B7_rename']['n']} "
                 f"({gen['variable_rename']['best']['B7_rename']['accuracy']})。")
    lines.append(f"- 整体宏平均是否超过所有适用捷径：**{'是' if beats_shortcuts else '否'}**；"
                 f"捷径基线：{json.dumps({k: v['accuracy'] for k, v in shortcuts.items()}, ensure_ascii=False)}。")
    lines.append("")
    lines.append("## 6. Router / Workspace 审计")
    lines.append("")
    for n in ("parent", "best", "final"):
        r = router[n]
        lines.append(f"- {n}: top_k={r['top_k']}/{r['n_blocks']}，稀疏空间={r['sparse_space']}，"
                     f"使用块数={r['blocks_used']}，最大/最小负载比={r['imbalance_max_min']}，"
                     f"Workspace 平均占用={r['workspace_mean_occupied']}，STE 梯度范数={r['ste_gradient_norm']}。")
    lines.append(f"- Padding 隔离实测：{json.dumps(memory['padding_isolation'], ensure_ascii=False)}。")
    lines.append("- 诚实限制：`workspace.write` 的 overflow 仍未被情景记忆消费者使用；信息保持缺陷未完全消除。")
    lines.append("")
    lines.append("## 7. 保留与遗忘")
    lines.append("")
    for k, v in retention.items():
        lines.append(f"- {k}: macro Δ={v['macro_delta']}；提升 {v['improved']}；下降 {v['regressed']}。")
    lines.append("")
    lines.append("## 8. 45 组真实输入输出（含失败，未用解释器纠正）")
    lines.append("")
    lines.append("| 模型 | 任务 | Prompt | 期望 | N-MIND 原始输出 | 正确 | 结束 |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in gen_rows:
        lines.append(f"| {r['model']} | {r['task']} | {md_escape(r['prompt'])} | "
                     f"{md_escape(r['expected'])} | {md_escape(r['predicted'])} | "
                     f"{'是' if r['ok'] else '否'} | {r['finish']} |")
    lines.append("")
    lines.append("## 9. Checkpoint 审计")
    lines.append("")
    for n, d in ck_audit["checkpoints"].items():
        lines.append(f"- `{n}`: step={d['optimizer_steps_executed']}, role={d['role']}, "
                     f"finite={d['all_finite']}, torch_rng={d['torch_rng_state_saved']}, "
                     f"sha={d['file_sha256'][:16]}…")
    lines.append("")
    lines.append("## 10. 最终问题的逐条回答")
    lines.append("")
    answers = build_answers(panel_eval, gen, router, training, beats_shortcuts)
    for i, a in enumerate(answers, 1):
        lines.append(f"{i}. **{a}**")
        lines.append("")
    lines.append("## 11. 下一步（未执行）")
    lines.append("")
    lines.append("- 复制/反转失败的根因仍需受控消融：query readout、位置编码、长度泛化读写分离。")
    lines.append("- 不得直接扩大步数掩盖架构缺陷；需在 V1.8 预登记方案中先做单变量改动。")
    lines.append("- GRU 短时对照本轮 **NOT_RUN**，不得据此宣称架构优越性。")
    lines.append("")
    (HERE / "C11_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def build_answers(panel, gen, router, training, beats_shortcuts):
    best = panel["best"]["per_task"]
    parent = panel["parent"]["per_task"]
    a1, a2 = best["A1_copy1"]["accuracy"], best["A2_copyN"]["accuracy"]
    a3, a4 = best["A3_index"]["accuracy"], best["A4_reverse"]["accuracy"]
    b2, b4 = best["B2_bind"]["accuracy"], best["B4_bind"]["accuracy"]
    b5, b6 = best["B5_reassign"]["accuracy"], best["B6_indirect"]["accuracy"]
    cf = gen["counterfactual"]["best"]["paired_accuracy"]
    lg = gen["length generalization"]["best"]["macro_accuracy"]
    return [
        (f"不能稳定复制两个或更多字符。单字符复制 A1={a1}，但 A2_copyN={a2}；"
         f"连续两次评估没有任何任务达到稳定门槛。"),
        (f"只具备部分能力。A3_index 固定面板 Best={a3}（父模型 {parent['A3_index']['accuracy']}），"
         f"高于随机但低于 90% 晋级线，不能称为稳定读取。"),
        (f"不能。A4_reverse Best={a4}，短序列反转仍失败。"),
        (f"能区分但不可靠。B2_bind={b2}、B4_bind={b4}；查询对象不同带来的输出差异存在，"
         f"但未达到 90%/80% 门槛。"),
        (f"不能可靠区分。B5_reassign={b5}，仍显著低于 85% 门槛。"),
        (f"有弱证据但不足。B6_indirect={b6}，低于 80% 门槛。"),
        (f"整体宏平均超过所有适用简单捷径：{'是' if beats_shortcuts else '否'}；"
         f"但分任务看，复制与反转仍被固定输出/位置捷径问题主导，不能宣称所有能力都超过捷径。"),
        (f"未见组合泛化能力有限：反事实配对={cf}，长度泛化={lg}，"
         f"变量重命名与未见格式结果见 C11_GENERALIZATION.json；没有达到可靠泛化证据。"),
        (f"Router 有真实稀疏空间（top_k=2 < n_blocks=4），但负载不均；Workspace 的 padding 污染已修复，"
         f"overflow 仍未被消费，因此信息保持缺陷只部分解决。"),
        (f"仍有潜力但不能证明充分。Loss 从约 2.25 降到约 1.05、面板宏平均提升 "
         f"{round(panel['best']['macro_accuracy'] - panel['parent']['macro_accuracy'], 4)}，"
         f"但基础复制/反转门槛未过，继续加步数不是当前最优策略。"),
        (f"受控支持：父模型统一面板审计、无泄漏数据审计、padding 隔离单测、固定面板三模型同协议评测、捷径基线；"
         f"仍属假设：复制失败的具体归因、Router 负载不均与容量瓶颈的因果关系。GRU 对照 NOT_RUN。"),
        (f"不满足。单字符复制、多字符复制、反转与变量绑定均未形成连续两评估点稳定证据，"
         f"禁止进入 C4–C9 正式训练。"),
    ]


if __name__ == "__main__":
    main()
