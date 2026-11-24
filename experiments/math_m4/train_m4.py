"""M41 唯一正式训练入口（正控制 + 四架构双种子）。

预算：正控制 256 步（R1/R2 各 128）+ 正式 3840 步（4架构×2seed×480）= 4096。
**必须 --authorize 才执行梯度更新**；未授权仅 dry-run（零 optimizer.step）。
每次 optimizer.step 前先 BudgetManager.reserve_step 持久化登记，成功后 confirm_step。
异常中断保留未确认 reserve，预算按已预约步数保守计入。
正控制只在固定 64 条样本上训练与评测；参考模型未过门禁则中止正式训练。

运行：
    python train_m4.py                 # dry-run（零步）
    python train_m4.py --authorize     # 需用户明确授权
    python train_m4.py --authorize --stage positive
    python train_m4.py --authorize --stage main   # 需已有正控制通过记录
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "models"))

from tokenizer_m4 import M4Tokenizer  # noqa: E402
from budget_manager import BudgetManager  # noqa: E402
from evaluate_m4 import (load_rows, load_counterfactual, load_distance,  # noqa: E402
                         build_fixed_panel, make_batch, grouped_loss,
                         eval_rows, tf_position_accuracy, iso_record,
                         critical_metrics, counterfactual_eval, distance_eval)
from compute_accounting_m4 import batch_cost  # noqa: E402

from gru_reference import GRUReferenceM3  # noqa: E402
from transformer_reference import TinyTransformerM3  # noqa: E402
from low_rank_interaction import LowRankInteractionM4  # noqa: E402
from selective_interaction import SelectiveInteractionM4  # noqa: E402
from positive_control import (control_rows, alignment_report, gate_spec,  # noqa: E402
                              evaluate_gate, N_CONTROL)
from preflight_guard import assert_training_entry  # noqa: E402

MODELS = {
    "R1_gru_reference": GRUReferenceM3,
    "R2_tiny_transformer": TinyTransformerM3,
    "M4-A_low_rank_interaction": LowRankInteractionM4,
    "M4-B_selective_interaction": SelectiveInteractionM4,
}
CFG = {
    "R1_gru_reference": {},
    "R2_tiny_transformer": {"d_model": 64, "nhead": 4, "layers": 2},
    "M4-A_low_rank_interaction": {"d_model": 64, "nhead": 4, "layers": 2, "r": 16},
    "M4-B_selective_interaction": {"d_model": 64, "nhead": 4, "layers": 2,
                                    "group_size": 4, "top_k": 2},
}
REFERENCE = ("R1_gru_reference", "R2_tiny_transformer")
LR, BATCH_SIZE, CLIP = 0.001, 16, 1.0
EVAL_INTERVAL = 64
STEPS_MAIN = 480
STEPS_POS = 128


def sha_state(model) -> str:
    h = hashlib.sha256()
    for k, p in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def save_ckpt(model, opt, path, meta):
    tmp = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "meta": meta}, tmp)
    blob = torch.load(tmp, weights_only=False)
    for k, v in model.state_dict().items():
        if not torch.equal(v, blob["model"][k]):
            raise RuntimeError(f"checkpoint 核验失败 {k}")
    tmp.replace(path)


def gradient_health(model):
    norms, finite = [], True
    for p in model.parameters():
        if p.grad is None:
            continue
        n = float(p.grad.detach().norm().item())
        finite = finite and bool(torch.isfinite(p.grad).all().item())
        norms.append(n)
    total = sum(norms) ** 0.5 if norms else 0.0
    return {"finite": finite, "nonzero": total > 0.0, "total_norm": round(total, 6)}


def classify_status(results, budget_summary, authorized, stage, positive_ok,
                    exception=None):
    """明确区分完成/部分完成/正控制失败/超预算/异常中止。"""
    if exception is not None:
        return "ABORTED_EXCEPTION"
    if not authorized:
        return "DRY_RUN_COMPLETE"
    if not positive_ok:
        return "POSITIVE_CONTROL_FAILED"
    if stage == "positive":
        return "COMPLETED"
    main_keys = [k for k in results.get("models", {}) if k.startswith("main_")]
    all_done = len(main_keys) == 8 and all(
        results["models"][k].get("status") == "DONE" for k in main_keys)
    if all_done:
        return "COMPLETED"
    if budget_summary.get("steps_used", 0) >= budget_summary.get("max_steps", 4096):
        return "OVER_BUDGET"
    return "PARTIALLY_COMPLETED"


def causality_ok():
    p = HERE / "M41_CAUSALITY_AUDIT.json"
    if not p.exists():
        return False
    audit = json.loads(p.read_text(encoding="utf-8"))
    return audit.get("status") == "PASS" and all(
        m.get("status") == "PASS" for m in audit.get("models", {}).values())


def load_positive_gate_status():
    p = HERE / "M4_POSITIVE_CONTROL.json"
    if not p.exists():
        return {"pass": False, "reason": "缺少正控制证据"}
    obj = json.loads(p.read_text(encoding="utf-8"))
    gates = obj.get("gates", {})
    if not gates:
        return {"pass": False, "reason": "正控制证据中没有门禁结果"}
    bad = [k for k, v in gates.items() if not v.get("gate", {}).get("pass", False)]
    return {"pass": not bad, "failed": bad,
            "reason": None if not bad else "已有正控制记录未通过门禁"}


def save_positive_gate_result(tok, rows, gates, alignment, executed):
    out = {"status": ("COMPLETE" if all(v["gate"]["pass"] for v in gates.values())
                      else "FAILED") if executed else "NOT_RUN_TRAINING",
           "reason": None if executed else "未授权，未执行正控制训练",
           "sole_training_entry": "train_m4.py",
           "n_samples": len(rows), "alignment": alignment,
           "gate_spec": gate_spec(), "gates": gates,
           "note": "正控制权重不进入主实验；成绩不等于未见数据泛化"}
    p = HERE / "M4_POSITIVE_CONTROL.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


def train_one(arch, seed, stage, budget, tok, train_rows, eval_rows_arg,
              cf_pairs, distance_rows, execute):
    steps = STEPS_POS if stage == "positive" else STEPS_MAIN
    torch.manual_seed(seed)
    random.seed(seed)
    model = MODELS[arch](tok.vocab_size, **CFG[arch])
    init_sha = sha_state(model)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    rng = random.Random(seed + 1000)
    shuffled = list(train_rows)
    rng.shuffle(shuffled)
    batches = [shuffled[i:i + BATCH_SIZE] for i in range(0, len(shuffled), BATCH_SIZE)]
    bptr = 0
    curve, best, best_path = [], -1.0, None
    step, cumulative_macs = 0, 0
    grad_health = []
    parameter_updated = False
    base = budget.steps_used
    wall0 = budget.current_wall()
    run_id = f"{arch}_s{seed}_{stage}"

    while step < steps and execute:
        ok, reason = budget.can_step(stage)
        if not ok:
            break
        batch = batches[bptr % len(batches)]
        bptr += 1
        x, valid, w, tgt = make_batch(tok, batch, "A")
        model.train()
        logits, fwd_state = model(x[:, :-1], valid_mask=valid[:, :-1])
        loss = grouped_loss(logits, tgt[:, 1:], w[:, 1:])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        health = gradient_health(model)
        if not (health["finite"] and health["nonzero"]):
            print(f"  [{run_id}] 梯度异常，停止：{health}", flush=True)
            break
        grad_health.append(health)
        # M4-B 直接复用这次真实训练前向的 stats，禁止再跑一次模型来统计成本。
        fwd_stats = fwd_state.get("stats") if isinstance(fwd_state, dict) else None
        bc = batch_cost(model, x, valid, w, backward=True, optimizer=True,
                        input_len=x.shape[1] - 1, hard_gate=True,
                        forward_stats=fwd_stats)
        batch_info = {"B": bc.batch_size, "T": bc.tensor_length,
                      "valid": bc.valid_tokens, "supervised": bc.supervised_tokens}
        macs = int(bc.analytic_fwd_macs + bc.analytic_bwd_macs + bc.optimizer_macs)
        pre_sha = sha_state(model) if step == 0 else None

        rid = budget.reserve_step(run_id, arch, seed, step, stage, batch_info, macs)
        try:
            opt.step()  # 唯一受共享预算保护的真实梯度更新点
        except Exception as exc:
            budget.mark_reserve_uncertain(rid, f"optimizer_step_exception:{exc}")
            raise
        budget.confirm_step(rid, float(loss.item()), batch_info, macs)
        post_sha = sha_state(model) if step == 0 else None
        if step == 0:
            parameter_updated = pre_sha != post_sha
        cumulative_macs += macs
        step += 1

        if step % EVAL_INTERVAL == 0 or step == steps:
            model.eval()
            point = {"step": base + step, "local_step": step,
                     "cumulative_macs": cumulative_macs, "hard_gate": True}
            if stage == "positive":
                tf = tf_position_accuracy(model, tok, eval_rows_arg, hard_gate=True)
                ev = eval_rows(model, tok, eval_rows_arg, hard_gate=True)
                point.update({"strict_macro": ev["strict_macro"],
                              "full_eos": ev["full_answer_natural_eos"],
                              "tf_accuracy": tf["accuracy"], "critical": {}})
                score = tf["accuracy"]
            else:
                cm = critical_metrics(model, tok, eval_rows_arg, cf_pairs,
                                      distance_rows, hard_gate=True)
                point.update(cm)
                score = cm["strict_macro"]
            curve.append(point)
            print(f"  [{run_id}] {step}/{steps} strict={point.get('strict_macro')} "
                  f"wall={budget.current_wall():.1f}s", flush=True)
            if score > best:
                best = score
                best_path = HERE / "checkpoints" / f"{arch}_s{seed}_{stage}_best.pt"
                save_ckpt(model, opt, best_path,
                          {"arch": arch, "seed": seed, "stage": stage, "step": base + step,
                           "role": "best", "score": best, "init_sha": init_sha,
                           "tokenizer_sha": tok.vocab_sha, "lr": LR})

    final_path = None
    if execute and step > 0:
        final_path = HERE / "checkpoints" / f"{arch}_s{seed}_{stage}_final.pt"
        save_ckpt(model, opt, final_path,
                  {"arch": arch, "seed": seed, "stage": stage, "step": base + step,
                   "role": "final", "score": curve[-1].get("strict_macro", 0.0) if curve else 0.0,
                   "init_sha": init_sha, "tokenizer_sha": tok.vocab_sha, "lr": LR})
        budget.record_run_end(run_id, arch, seed, step, budget.current_wall() - wall0,
                              "completed" if step == steps else "budget_stop", stage, best,
                              hashlib.sha256(final_path.read_bytes()).hexdigest()[:16])

    result = {"arch": arch, "seed": seed, "stage": stage,
              "status": "DONE" if execute and step == steps else
                        ("DRY_RUN" if not execute else "STOPPED"),
              "optimizer_steps": step, "ledger_steps_delta": budget.steps_used - base,
              "init_sha": init_sha, "best_score": best, "curve": curve,
              "gradient_health": {"n_checked": len(grad_health),
                                  "all_finite_nonzero": bool(grad_health) and all(
                                      g["finite"] and g["nonzero"] for g in grad_health),
                                  "last": grad_health[-1] if grad_health else None},
              "parameter_updated": parameter_updated,
              "best_path": str(best_path) if best_path else None,
              "final_path": str(final_path) if final_path else None}
    if stage == "positive":
        final_tf = curve[-1]["tf_accuracy"] if curve else 0.0
        final_eos = curve[-1]["full_eos"] if curve else 0.0
        result["positive_gate"] = evaluate_gate(
            final_tf, final_eos,
            result["gradient_health"]["all_finite_nonzero"],
            parameter_updated, causality_ok())
    else:
        result["iso"] = iso_record(curve, seed=seed)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--authorize", action="store_true")
    ap.add_argument("--stage", choices=["all", "positive", "main"], default="all")
    args = ap.parse_args()
    torch.set_num_threads(20)
    tok = M4Tokenizer()
    train = load_rows("train")
    val = load_rows("val")
    panel = build_fixed_panel(val, per_task=20)
    pos_rows = control_rows(tok, n=N_CONTROL)
    alignment = alignment_report(tok, pos_rows)
    # 训练期只能使用验证反事实/距离；封存集留到训练与选模完成后。
    cf_pairs = load_counterfactual("val")
    distance_rows = load_distance("val")
    budget = BudgetManager()
    base_total = budget.steps_used
    print(f"[train_m4] authorized={args.authorize} stage={args.stage} "
          f"budget used={budget.steps_used}/{budget.max_steps}", flush=True)
    if not alignment["answer_only_alignment"]:
        raise RuntimeError(f"正控制监督对齐失败：{alignment['failures'][:3]}")

    results = {"status": "TRAINING" if args.authorize else "DRY_RUN",
               "authorization": "explicit --authorize" if args.authorize else "NOT_AUTHORIZED",
               "sole_training_entry": "train_m4.py",
               "stage_scope": args.stage,
               "tokenizer_sha": tok.vocab_sha, "panel_n": len(panel),
               "positive_n": len(pos_rows), "models": {}, "positive_gates": {}}

    def write_results():
        results["budget_final"] = budget.summary()
        results["optimizer_steps_executed"] = budget.steps_used - base_total
        p = HERE / "M4_TRAINING_RESULTS.json"
        p.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[train_m4] status={results['status']} steps="
              f"{results['optimizer_steps_executed']} 写出 {p}", flush=True)

    if args.authorize:
        try:
            results["entry_guard"] = assert_training_entry(
                require_positive_pass=(args.stage == "main"))
        except Exception as exc:
            results["status"] = "PREFLIGHT_REJECTED"
            results["guard_error"] = str(exc)
            write_results()
            return
        budget.start_session("M41F authorized training")

    try:
        positive_ok = True
        if args.stage in ("all", "positive"):
            for arch in REFERENCE:
                for seed in (1,):
                    r = train_one(arch, seed, "positive", budget, tok, pos_rows,
                                  pos_rows, cf_pairs, distance_rows, args.authorize)
                    key = f"positive_{arch}_s{seed}"
                    results["models"][key] = r
                    gate = r.get("positive_gate", {"pass": False,
                                                   "reason": "未授权训练，未产生门禁结果"})
                    results["positive_gates"][key] = gate
                    positive_ok = positive_ok and gate["pass"]
            save_positive_gate_result(tok, pos_rows, results["positive_gates"],
                                      alignment, args.authorize)

        if args.stage == "main":
            prior = load_positive_gate_status()
            positive_ok = prior["pass"]
            results["positive_prior"] = prior

        if not positive_ok and args.authorize:
            results["status"] = "POSITIVE_CONTROL_FAILED"
            budget.end_session("aborted_positive_control_failed")
            write_results()
            return
        if not positive_ok and not args.authorize:
            results["dry_run_note"] = (
                "未授权，正控制门禁未实际评测；继续零步 dry-run 以检查入口完整性")

        if args.authorize and args.stage == "all":
            # 正控制通过后，进入主训练前必须再次用最新正控制报告过门禁。
            try:
                results["main_entry_guard"] = assert_training_entry(True)
            except Exception as exc:
                results["status"] = "POSITIVE_CONTROL_FAILED"
                results["positive_gate_error"] = str(exc)
                budget.end_session("positive_gate_rejected")
                write_results()
                return

        if args.stage in ("all", "main"):
            for arch in MODELS:
                for seed in (1, 2):
                    ok, _ = budget.can_step("main")
                    if not ok:
                        break
                    r = train_one(arch, seed, "main", budget, tok, train, panel,
                                  cf_pairs, distance_rows, args.authorize)
                    results["models"][f"main_{arch}_s{seed}"] = r

        if args.authorize:
            budget.end_session()
        results["status"] = classify_status(
            results, budget.summary(), args.authorize, args.stage, positive_ok)
        write_results()
    except Exception as exc:
        results["status"] = "ABORTED_EXCEPTION"
        results["exception"] = repr(exc)
        if args.authorize:
            budget.end_session("aborted_exception")
        write_results()
        raise


if __name__ == "__main__":
    main()
