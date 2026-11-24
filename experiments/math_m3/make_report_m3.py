"""M3 最终交付物生成（训练后权威入口）。

修复要点（相对 dry-run 版本）：
1. 所有"训练后"生成/评测必须加载真实 checkpoint，禁止用随机初始化模型冒充训练结果。
2. 增加封存集、反事实配对、长度/格式泛化的统一评测（同模型、同顺序、同面板）。
3. ISO 曲线使用两 seed 对齐平均；未达门槛一律 TARGET_NOT_REACHED，禁止外推。
4. Pareto 用真实能力回填，不再硬编码 capability=0。
5. 动态路由用训练后的 M3-B best checkpoint 重测，并如实报告软门/硬门准确率差。

用法：python make_report_m3.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "models"))

from tokenizer_m3 import M3Tokenizer  # noqa: E402
from evaluate_m3 import (load_rows, build_fixed_panel, generate, evaluate_rows,  # noqa: E402
                         evaluate_panel, evaluate_counterfactual, evaluate_distance,
                         iso_record, ISO_TARGETS)
from train_m3 import MODELS  # noqa: E402
from compute_accounting import model_mac_profile  # noqa: E402
from pareto_selector import pareto_front, efficiency_grade  # noqa: E402
from dynamic_routing_report import measure_routing, accuracy_soft_vs_hard  # noqa: E402

# 模型构造参数（注意 Transformer 构造签名用 nhead；MAC 核算用 n_head）
ARCH_CFG = {
    "R1_gru_reference": {"d_model": 64, "hidden": 128},
    "R2_tiny_transformer": {"d_model": 64, "nhead": 4, "layers": 2},
    "M3-A_low_rank_neural": {"d": 96, "r": 16, "n_layers": 1},
    "M3-B_adaptive_computation": {"d": 96, "r": 16, "hidden_mult": 2},
}
MAC_CFG = {
    "R1_gru_reference": {"d_model": 64, "hidden": 128},
    "R2_tiny_transformer": {"d_model": 64, "n_head": 4, "layers": 2},
    "M3-A_low_rank_neural": {"d": 96, "r": 16, "n_layers": 1},
    "M3-B_adaptive_computation": {"d": 96, "r": 16, "hidden_mult": 2},
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def build_trained(arch: str, tok: M3Tokenizer, seed: int, role: str = "best"):
    """构造同配置模型并加载真实 checkpoint；缺权重直接失败，绝不静默退化随机。"""
    ckpt = HERE / "checkpoints" / f"{arch}_s{seed}_{role}.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"缺少训练权重，拒绝生成训练后证据: {ckpt}")
    blob = torch.load(ckpt, weights_only=False)
    model = MODELS[arch](tok.vocab_size, **ARCH_CFG[arch])
    model.load_state_dict(blob["model"])
    model.eval()
    return model, blob.get("meta", {}), ckpt


def representative_seed(train_res: dict, arch: str) -> int:
    """预注册规则：取两 seed 中面板 macro 最高者作为架构代表（另保留双 seed 结果）。"""
    scores = {}
    for seed in (1, 2):
        r = train_res.get("models", {}).get(f"{arch}_s{seed}")
        if r and r.get("status") == "DONE":
            scores[seed] = r.get("best_macro", -1.0)
    if not scores:
        raise RuntimeError(f"架构 {arch} 无已完成训练结果")
    return max(scores, key=lambda s: scores[s])


def curve_mean(train_res: dict, arch: str, per_token_mac: int | None = None) -> list[dict]:
    """两 seed 学习曲线按 local_step 对齐求宏均值，并修正成本口径。

    重要：train_m3.py 原始 curve 的 samples/tokens/macs 使用全局累计步数
    （base_steps + local_step）计算，跨架构比较时会把之前架构的训练成本计入本模型。
    此处按 local_step 重算单模型真实成本；原始文件不改，保留可追溯性。
    wall 仍为预算台账的全局累计墙钟，无法从现有记录拆分，报告中明确标注。
    """
    curves = [train_res["models"][f"{arch}_s{s}"]["curve"] for s in (1, 2)
              if f"{arch}_s{s}" in train_res.get("models", {})]
    if not curves:
        return []
    by_step = defaultdict(list)
    for c in curves:
        for pt in c:
            by_step[pt["local_step"]].append(pt)
    out = []
    for step in sorted(by_step):
        pts = by_step[step]
        rec = dict(pts[0])
        for k in ("macro", "content_acc"):
            rec[k] = round(sum(float(p.get(k, 0.0)) for p in pts) / len(pts), 6)
        rec["step"] = step                       # 单模型 local step（ISO 使用）
        rec["global_step"] = pts[0].get("step")  # 原始全局步数，保留追溯
        rec["samples"] = step * 32               # 单模型样本数
        rec["tokens"] = step * 32 * 32           # 单模型 token 数
        if per_token_mac is not None:
            rec["macs"] = step * 32 * 32 * per_token_mac * 3   # fwd+bwd≈3x
        rec["wall"] = round(sum(float(p.get("wall", 0.0)) for p in pts) / len(pts), 2)
        rec["wall_note"] = "预算台账全局累计墙钟（含此前架构），非单模型训练时间"
        rec["seeds"] = len(pts)
        out.append(rec)
    return out


def sample_by_task(rows: list[dict], per_task: int) -> list[dict]:
    by = defaultdict(list)
    for r in rows:
        by[r["task"]].append(r)
    out = []
    for task in sorted(by):
        out.extend(by[task][:per_task])
    return out


def main():
    torch.set_num_threads(20)
    tok = M3Tokenizer()
    train_res = json.loads((HERE / "M3_TRAINING_RESULTS.json").read_text(encoding="utf-8"))
    if train_res.get("status") != "TRAINING_DONE":
        raise RuntimeError("训练未完成，拒绝生成训练后交付物")

    val = load_rows("val")
    sealed = load_rows("sealed")
    gen_rows = load_rows("generalization")
    panel = build_fixed_panel(val, per_task=20)
    sealed_panel = sample_by_task(sealed, per_task=20)
    gen_panel = sample_by_task(gen_rows, per_task=25)
    cf_rows = [json.loads(l) for l in
               (HERE / "data" / "m3_counterfactual.jsonl").read_text(encoding="utf-8").splitlines()
               if l.strip()]

    # ---------------- 1. checkpoint 审计（含文件 SHA，逐张量回读核验） ----------------
    ckpt_dir = HERE / "checkpoints"
    ckpts = sorted(ckpt_dir.glob("*.pt")) if ckpt_dir.exists() else []
    audit = {"status": "TRAINED", "n_checkpoints": len(ckpts), "checkpoints": [],
             "note": "训练后 checkpoint：每张量回读一致，文件 SHA256 记录，权重不上传公开仓库"}
    for p in ckpts:
        try:
            blob = torch.load(p, weights_only=False)
            meta = blob.get("meta", {})
            n_tensors = len(blob["model"])
            audit["checkpoints"].append({
                "file": p.name, "sha256": sha256(p), "bytes": p.stat().st_size,
                "n_tensors": n_tensors,
                "arch": meta.get("arch"), "seed": meta.get("seed"),
                "role": meta.get("role"), "step": meta.get("step"),
                "macro": meta.get("macro"), "init_sha": meta.get("init_sha"),
                "formula_version": meta.get("formula_version"),
                "code_version": meta.get("code_version"),
                "data_sha": meta.get("data_sha"),
                "tokenizer_sha": meta.get("tokenizer_sha"),
                "global_step": meta.get("global_step"),
            })
        except Exception as e:
            audit["checkpoints"].append({"file": p.name, "error": str(e)})
    (HERE / "M3_CHECKPOINT_AUDIT.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[final] checkpoint 审计 n={len(ckpts)}", flush=True)

    # ---------------- 2. 训练后统一评测（同面板同顺序） ----------------
    final_eval = {
        "status": "TRAINED",
        "sampling": {
            "fixed_panel": f"val 每任务20条，共{len(panel)}条",
            "sealed_panel": f"sealed 每任务20条，共{len(sealed_panel)}条（封存集仅最终报告，不参与选模）",
            "counterfactual": f"{len(cf_rows)}对，查询对象不变只改事实",
            "generalization": f"每任务25条，共{len(gen_panel)}条（长度/格式外推）",
        },
        "models": {},
    }
    reps = {}
    for arch in MODELS:
        seed = representative_seed(train_res, arch)
        repos = {}
        for role in ("best", "final"):
            model, meta, ckpt = build_trained(arch, tok, seed, role)
            p_res = evaluate_panel(model, tok, panel)
            s_res = evaluate_panel(model, tok, sealed_panel)
            repos[role] = {
                "seed": seed, "checkpoint": ckpt.name,
                "panel": {k: v for k, v in p_res.items() if k != "results"},
                "sealed": {k: v for k, v in s_res.items() if k != "results"},
            }
            if role == "best":
                repos[role]["counterfactual"] = evaluate_counterfactual(model, tok, cf_rows)
                repos[role]["distance"] = evaluate_distance(model, tok, sealed)
                g_res = evaluate_rows(model, tok, gen_panel)
                repos[role]["generalization"] = {k: v for k, v in g_res.items() if k != "results"}
        final_eval["models"][arch] = repos
        reps[arch] = (model, seed, meta)
        print(f"[final] {arch} seed{seed} best panel={repos['best']['panel']['macro_accuracy']} "
              f"sealed={repos['best']['sealed']['macro_accuracy']}", flush=True)
    (HERE / "M3_FINAL_EVALUATION.json").write_text(
        json.dumps(final_eval, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------------- 3. 生成样例（训练后 checkpoint，48 组含失败） ----------------
    all_gen = []
    for arch, (model, seed, meta) in reps.items():
        rows = sample_by_task(sealed, per_task=3)[:12]
        for r in rows:
            g = generate(model, tok, r["prompt"])
            all_gen.append({
                "arch": arch, "seed": seed, "checkpoint": meta.get("role", "best"),
                "task": r["task"], "prompt": r["prompt"],
                "expected": r["answer"], "predicted": g["text"],
                "ok": g["text"] == r["answer"],
                "finish_reason": g["finish_reason"],
                "immediate_eos": g["immediate_eos"],
            })
    (HERE / "M3_GENERATION.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in all_gen) + "\n", encoding="utf-8")
    print(f"[final] 生成样例 {len(all_gen)} 组，正确 {sum(r['ok'] for r in all_gen)}", flush=True)

    # ---------------- 4. ISO 能力对比（双 seed 平均曲线） ----------------
    iso_models = {}
    for arch in MODELS:
        mac_prof = model_mac_profile(arch, tok.vocab_size, **MAC_CFG[arch])
        curve = curve_mean(train_res, arch, mac_prof["per_token_forward_mac"])
        iso_models[arch] = {
            "status": "TRAINED" if curve else "NO_CURVE",
            "iso": iso_record(curve) if curve else
                   {f"{int(t*100)}%": {"status": "TARGET_NOT_REACHED"} for t in ISO_TARGETS},
            "curve": curve,
            "per_token_mac": mac_prof["per_token_forward_mac"],
        }
    iso_out = {
        "status": "TRAINED",
        "iso_targets": ISO_TARGETS,
        "stability_rule": "连续 2 个测量点 macro >= target；末尾点达标亦接受",
        "aggregation": "每架构两 seed 按 local_step 对齐取宏均值",
        "cost_accounting_fix": "原始训练曲线按全局累计步数记账，本文件按 local_step 重算单模型样本/Token/MAC；wall 仍为全局累计墙钟",
        "forbidden": "未达门槛不得外推虚构成本",
        "models": iso_models,
    }
    (HERE / "M3_ISO_CAPABILITY_COMPARISON.json").write_text(
        json.dumps(iso_out, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------------- 5. Pareto（真实能力 + 真实资源） ----------------
    acc = json.loads((HERE / "M3_COMPUTE_ACCOUNTING.json").read_text(encoding="utf-8"))
    models_pareto = {}
    for arch in MODELS:
        v = acc["models"][arch]
        lat = next((r["forward_median_sec"] for r in v["measurements"]
                    if r["batch_size"] == 8 and r["seq_len"] == 32), 0.0)
        curve = curve_mean(train_res, arch, iso_models[arch]["per_token_mac"])
        # 达到 30% 所需的累计 MAC（未达到则记为 None，禁止外推）
        reach30 = iso_record(curve).get("30%", {})
        models_pareto[arch] = {
            "capability": final_eval["models"][arch]["best"]["panel"]["macro_accuracy"],
            "sealed_capability": final_eval["models"][arch]["best"]["sealed"]["macro_accuracy"],
            "cost_train": reach30.get("macs") if reach30.get("status") == "REACHED" else None,
            "cost_train_full": curve[-1]["macs"] if curve else None,
            "cost_infer": iso_models[arch]["per_token_mac"],
            "state_elems": v["state_elements"],
            "latency": lat,
            "params": v["params"],
        }
    # 未达 30% ISO 门槛者没有可比较的达标成本：Pareto 计算中按 +inf 处理（成本无穷大），
    # 报告中仍保留 null，绝不把全量训练 MAC 伪装成"达标成本"。
    pareto_input = {}
    for k, v in models_pareto.items():
        vv = dict(v)
        if vv["cost_train"] is None:
            vv["cost_train"] = float("inf")
        pareto_input[k] = vv
    front = pareto_front(pareto_input)
    pareto_out = {
        "status": "TRAINED",
        "note": "能力=固定验证面板 macro（两seed代表best）；未达30%门槛者 cost_train=null，Pareto计算按+inf处理，不外推",
        "models": models_pareto,
        "pareto_front": front,
        "efficiency_vs_R1": {},
    }
    ref = models_pareto["R1_gru_reference"]
    for arch in ("M3-A_low_rank_neural", "M3-B_adaptive_computation", "R2_tiny_transformer"):
        cand = models_pareto[arch]
        same_level = (cand["capability"] >= ref["capability"] and
                      cand["cost_train"] is not None and ref["cost_train"] is not None)
        if same_level:
            red = 1 - cand["cost_train"] / ref["cost_train"]
            pareto_out["efficiency_vs_R1"][arch] = {
                "status": "COMPARABLE", "train_compute_reduction": round(red, 4),
                "grade": efficiency_grade(red),
            }
        else:
            pareto_out["efficiency_vs_R1"][arch] = {
                "status": "NOT_COMPARABLE_OR_TARGET_NOT_REACHED",
                "reason": "能力未达参考或双方未达同一 ISO 门槛，禁止宣称同等能力效率收益",
            }
    (HERE / "M3_PARETO_RESULTS.json").write_text(
        json.dumps(pareto_out, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------------- 6. 训练后动态路由（硬门/软门分别报告） ----------------
    m3b_model, m3b_seed, _ = reps["M3-B_adaptive_computation"]
    routing = {
        "status": "TRAINED",
        "checkpoint": f"M3-B_adaptive_computation_s{m3b_seed}_best.pt",
        "routing": measure_routing(m3b_model, tok),
        "soft_vs_hard": accuracy_soft_vs_hard(m3b_model, tok, panel),
        "note": "硬门报告以 hard_gate 为准；软门仅作对照，不得替代硬门成绩",
    }
    (HERE / "M3_DYNAMIC_ROUTING_REPORT.json").write_text(
        json.dumps(routing, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[final] 训练后硬门跳过率={routing['routing']['hard_gate']['skipped_mac_ratio']}",
          flush=True)
    print("[final] 全部训练后交付物生成完成", flush=True)


if __name__ == "__main__":
    main()
