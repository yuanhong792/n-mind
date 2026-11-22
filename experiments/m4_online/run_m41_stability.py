"""M4.1 稳定性预注册实验（m4.1-stability-001，1 预声明 seed）。

小袁 2026-11-22 指令预登记（运行前已定，不得看结果后暗改）：
  λ_anchor = 0.0（M4 首轮失败诊断：0.1 过度约束 40 步 adapter 学习）
  λ_replay = 0.15（保留轻量回放防遗忘）
  步数     = Neural-only 48 / Hybrid 48 / Negative(乱标签) 16 = 112 / 120
  其余全部沿用 m4-online-smoke-001：父 checkpoint、B 数据 seed（20261201/02/03）、
  scope=adapter、replay_k=4、封存集、buffer 复位协议、门槛（B_OOD 冲突步 ≥ +10pp、
  A 遮挡段下降 ≤ 3pp、neural>frozen、>负对照、冻结参数逐位 0 违规、继承逐位）。
  通过后向小袁提出 3-seed 复验申请；不自动扩大预算。

变化点只有三个：λ_anchor、λ_replay、步数。数据与评测协议与 M4 首轮完全一致，
因此可直接对比 λ 配置的效果（同一父模型、同一经验流、同一封存集）。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from run_online_smoke import build_rule_rows, mem_rows_of  # noqa: E402  复用夹具与行构造

from nmind.config import default_config  # noqa: E402
from nmind.evaluation.lifelong import eval_skill, rule_conflict_masks  # noqa: E402
from nmind.learning import datasets as D  # noqa: E402
from nmind.learning import online_adaptation as OA  # noqa: E402
from nmind.learning.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from nmind.learning.online_trainer import bounded_stream_update  # noqa: E402
from nmind.learning.trainer import evaluate, make_data  # noqa: E402
from nmind.online import (Experience, MALICIOUS_URL, RULE_PAGE_URL, UNRELATED_URL,  # noqa: E402
                           append_audit, build_learning_stream, build_provenance,
                           check_gates, decide, decide_status, detect_instructions,
                           extract_rule_table, extract_text, file_sha256, fixture_fetcher,
                           save_jsonl, verify_rule)
from nmind.online.experiences import now_iso  # noqa: E402

EXP_ID = "m4.1-stability-001"
SEED_A = 20261131
SEED_B_TRAIN = 20261201
SEED_B_TEST = 20261202
SEED_B_OOD = 20261203
SAMPLING = {"Neural-only": 20261211, "Hybrid": 20261212, "Negative-shuffled": 20261213}
STEPS = {"Neural-only": 48, "Hybrid": 48, "Negative-shuffled": 16}
BUDGET_CAP = 120
SCOPE = "adapter"
LAMBDA_REPLAY = 0.15
LAMBDA_ANCHOR = 0.0
REPLAY_K = 4
GATE_B_OOD = 0.10
GATE_A_DROP = 0.03
WALL_LIMIT_S = 300.0
WALL_GUARD_S = 240.0


def rss_mb() -> float:
    try:
        import psutil
        return round(psutil.Process().memory_info().rss / 1e6, 1)
    except Exception:
        return -1.0


def main() -> int:
    t0 = time.perf_counter()
    (HERE / "ckpts").mkdir(parents=True, exist_ok=True)
    dev = torch.device("cpu")
    cfg = default_config()
    cfg.learning.traj_len = 24

    # ---- 稳定权重（与 M4 首轮同一父 checkpoint，直接可比）----
    parent, parent_meta = load_checkpoint(
        ROOT / "experiments" / "m2_ablation" / "C_sparse_ce_s20261131.pt",
        cfg=cfg, device="cpu", allow_config_mismatch=True)
    parent_state = {k: v.detach().cpu().clone() for k, v in parent.state_dict().items()}
    stable_sha = OA.state_hash(parent)
    taskA = D.GridTask(name="A", goal_visible_steps=4, seed=SEED_A)
    taskB = D.GridTask(name="B", goal_visible_steps=4, reversed_control=True, seed=SEED_B_TRAIN)
    dA = make_data(cfg, taskA, seed=SEED_A)
    a_test, a_train = dA["test"], dA["train"]

    # ---- OBSERVE / PARSE / VERIFY（与 M4 首轮同一夹具与规则）----
    fetcher = fixture_fetcher()
    rec_rule = fetcher.fetch(RULE_PAGE_URL)
    rec_mirror = fetcher.fetch("https://sim.local/rules/dir-bias-north-mirror.html")
    rec_mal = fetcher.fetch(MALICIOUS_URL)
    rec_unrel = fetcher.fetch(UNRELATED_URL)
    parsed = extract_rule_table(rec_rule.content or "")
    verdict = verify_rule(parsed)
    rule_fields = parsed["fields"]
    malicious_flags = detect_instructions(rec_mal.content or "")
    provenance = [build_provenance(r, "extractor-v1", "simulated-fixture").to_dict()
                  for r in (rec_rule, rec_mirror, rec_mal, rec_unrel)
                  if r.status in ("fetched", "duplicate")]
    print(f"[observe] rule={rule_fields.get('rule_id')} verify={verdict['verification']} "
          f"dedup={rec_mirror.status} malicious={malicious_flags['instruction_detected']}",
          flush=True)
    if verdict["verification"] != "verified":
        (HERE / "m41_stability.json").write_text(json.dumps(
            {"experiment_id": EXP_ID, "status": "INCONCLUSIVE",
             "reason": "规则未通过验证"}, ensure_ascii=False), encoding="utf-8")
        return 0

    # ---- 经验流（与 M4 首轮同 seed ⇒ 同一份流，直接可比）----
    used_pairs: set = set()
    b_train = build_rule_rows(96, SEED_B_TRAIN, 24, used_pairs, taskB)
    b_test = build_rule_rows(32, SEED_B_TEST, 24, used_pairs, taskB)
    b_ood = build_rule_rows(32, SEED_B_OOD, 32, used_pairs, taskB)
    excerpt = extract_text(rec_rule.content or "", limit=160)
    experiences: list[Experience] = []
    for i, r in enumerate(b_train):
        experiences.append(Experience(
            id=f"exp-b-{i:03d}", observed_at=now_iso(), source_url=RULE_PAGE_URL,
            source_sha256=rec_rule.sha256, raw_excerpt=excerpt,
            structured_observation={"episode": i, "goal_code": int(r["goal_code"]),
                                     "T": int(r["len"]), "rule_id": rule_fields["rule_id"]},
            action=None, expected_outcome=None, actual_outcome=None,
            verifier_id=verdict["verifier_id"], verification="verified", reward=None,
            task_family="rule_control_reversed", train_split="stream_train"))
    for tag, rows in (("test", b_test), ("ood", b_ood)):
        for i, r in enumerate(rows):
            experiences.append(Experience(
                id=f"exp-{tag}-{i:03d}", observed_at=now_iso(), source_url=RULE_PAGE_URL,
                source_sha256=rec_rule.sha256, raw_excerpt=excerpt,
                structured_observation={"episode": i, "goal_code": int(r["goal_code"]),
                                         "T": int(r["len"])},
                action=None, expected_outcome=None, actual_outcome=None,
                verifier_id=verdict["verifier_id"], verification="verified", reward=None,
                task_family="rule_control_reversed", train_split="sealed_eval"))
    experiences.append(Experience(
        id="exp-malicious-000", observed_at=now_iso(), source_url=MALICIOUS_URL,
        source_sha256=rec_mal.sha256,
        raw_excerpt=extract_text(rec_mal.content or "", limit=160),
        structured_observation={"instruction_detected": malicious_flags["instruction_detected"]},
        action=None, expected_outcome=None, actual_outcome=None, verifier_id=None,
        verification="rejected", reward=None, task_family="adversarial",
        train_split="stream_train"))
    experiences.append(Experience(
        id="exp-unrelated-000", observed_at=now_iso(), source_url=UNRELATED_URL,
        source_sha256=rec_unrel.sha256, raw_excerpt=extract_text(rec_unrel.content or "", 120),
        structured_observation={"page": "no rule table"}, action=None,
        expected_outcome=None, actual_outcome=None, verifier_id=verdict["verifier_id"],
        verification="uncertain", reward=None, task_family="rule_control_reversed",
        train_split="stream_train"))
    save_jsonl(HERE / "experiences_m41.jsonl", experiences)
    stream = build_learning_stream(experiences)
    assert len(stream) == 96
    pair_sets = lambda rows: {(tuple(r["start"]), tuple(r["goal"])) for r in rows}
    leakage_audit = {"train_test_pair_overlap": len(pair_sets(b_train) & pair_sets(b_test)),
                     "train_ood_pair_overlap": len(pair_sets(b_train) & pair_sets(b_ood)),
                     "sealed_ids_in_stream": sum(1 for e in stream
                                                 if e.train_split == "sealed_eval")}
    assert leakage_audit["train_test_pair_overlap"] == 0
    assert leakage_audit["train_ood_pair_overlap"] == 0
    assert leakage_audit["sealed_ids_in_stream"] == 0

    decision = decide(verified_count=len(stream), rejected_count=1, uncertain_count=1,
                      budget_remaining=BUDGET_CAP, cooldown_ok=True)
    budget = OA.AdaptationBudget(BUDGET_CAP, label=EXP_ID)

    # ---- 冻结基线（buffer 复位协议）----
    masks = {"test": rule_conflict_masks(b_test), "ood": rule_conflict_masks(b_ood)}
    a_before = round(evaluate(parent, None, a_test, device=dev,
                              occluded_only=True)["accuracy_occluded"], 4)
    b_before = eval_skill(parent, b_test, dev, conflict=masks["test"][0],
                          coincident=masks["test"][1])
    b_ood_before = eval_skill(parent, b_ood, dev, conflict=masks["ood"][0],
                              coincident=masks["ood"][1])
    print(f"[stable] sha={stable_sha[:16]} A={a_before} "
          f"B_conflict={b_before['acc_occluded_conflict']} "
          f"B_OOD_conflict={b_ood_before['acc_occluded_conflict']}", flush=True)
    arms: dict[str, Any] = {"Frozen": {"updates": 0, "A_after_occ": a_before,
                                       "A_forgetting": 0.0, "B": b_before,
                                       "B_OOD": b_ood_before, "mem_injected": False,
                                       "rss_mb": rss_mb()}}
    same_parent = True

    def clone_from_stable(arm: str):
        nonlocal same_parent
        m = load_checkpoint(ROOT / "experiments" / "m2_ablation" / "C_sparse_ce_s20261131.pt",
                            cfg=cfg, device="cpu", allow_config_mismatch=True)[0]
        chk = OA.verify_bitwise_inheritance({"model": parent_state}, m)
        if not chk["all_bitwise_equal"]:
            same_parent = False
            raise RuntimeError(f"{arm} 继承断言失败")
        return m

    # ---- Memory-only ----
    m = clone_from_stable("Memory-only")
    mem_before = OA.clone_snapshot(m)
    b_after_mem = eval_skill(m, mem_rows_of(b_test), dev, conflict=masks["test"][0],
                             coincident=masks["test"][1])
    b_ood_after_mem = eval_skill(m, mem_rows_of(b_ood), dev, conflict=masks["ood"][0],
                                 coincident=masks["ood"][1])
    after_snap = OA.clone_snapshot(m)
    mem_unchanged = all(torch.equal(mem_before["model"][k], after_snap["model"][k])
                        for k in mem_before["model"] if k != "block_load_count")
    arms["Memory-only"] = {"updates": 0, "mem_injected": True, "A_after_occ": a_before,
                           "A_forgetting": 0.0, "B": b_after_mem, "B_OOD": b_ood_after_mem,
                           "params_bitwise_unchanged": mem_unchanged, "rss_mb": rss_mb()}

    # ---- 神经臂（预登记 λ 配置）----
    candidate_paths: dict[str, str] = {}
    for arm in ("Neural-only", "Hybrid", "Negative-shuffled"):
        if time.perf_counter() - t0 > WALL_GUARD_S:
            arms[arm] = {"arm": arm, "skipped": "wall_clock_guard"}
            continue
        m = clone_from_stable(arm)
        res = bounded_stream_update(
            m, cfg=cfg, rows=b_train, device=dev, stable_state=parent_state,
            scope=SCOPE, max_updates=STEPS[arm], sampling_seed=SAMPLING[arm],
            replay_rows=a_train, lambda_replay=LAMBDA_REPLAY, lambda_anchor=LAMBDA_ANCHOR,
            replay_k=REPLAY_K,
            label_permutation_seed=(20261399 if arm == "Negative-shuffled" else None),
            budget=budget, arm=arm)
        with_mem = arm == "Hybrid"
        OA.reset_transient_buffers(m, parent_state)   # 评测协议：训练态 buffer 复位到父值
        eval_b_rows = mem_rows_of(b_test) if with_mem else b_test
        eval_ood_rows = mem_rows_of(b_ood) if with_mem else b_ood
        b_after = eval_skill(m, eval_b_rows, dev, conflict=masks["test"][0],
                             coincident=masks["test"][1])
        b_ood_after = eval_skill(m, eval_ood_rows, dev, conflict=masks["ood"][0],
                                 coincident=masks["ood"][1])
        a_after = round(evaluate(m, None, a_test, device=dev,
                                 occluded_only=True)["accuracy_occluded"], 4)
        child_path = None
        reload_score = None
        if arm != "Negative-shuffled":
            child_path = str(HERE / "ckpts" / f"{EXP_ID}_{arm}_candidate.pt")
            save_checkpoint(m, cfg=cfg, path=child_path, probe=None,
                            meta={"experiment_id": EXP_ID, "arm": arm,
                                  "parent_state_sha256": stable_sha, "steps": STEPS[arm],
                                  "lambda_anchor": LAMBDA_ANCHOR,
                                  "lambda_replay": LAMBDA_REPLAY})
            m2, _meta = load_checkpoint(child_path, cfg=cfg, device="cpu",
                                        allow_config_mismatch=True)
            reload_score = eval_skill(m2, eval_b_rows, dev, conflict=masks["test"][0],
                                      coincident=masks["test"][1])["acc_occluded_conflict"]
            candidate_paths[arm] = child_path
        arms[arm] = {"updates": STEPS[arm], "mem_injected": with_mem,
                     "A_after_occ": a_after,
                     "A_forgetting": round(a_before - a_after, 4),
                     "B": b_after, "B_OOD": b_ood_after,
                     "B_OOD_gain_conflict": None if b_ood_after["acc_occluded_conflict"] is None
                     else round(b_ood_after["acc_occluded_conflict"]
                                - (b_ood_before["acc_occluded_conflict"] or 0.0), 4),
                     "delta_summary": {k: res["delta"][k] for k in
                                       ("trainable_total", "trainable_changed_count",
                                        "frozen_total", "frozen_violations",
                                        "frozen_all_bitwise_unchanged")},
                     "max_param_abs_delta": round(max(
                         (v["delta_max_abs"] for v in res["delta"]["modules"].values()
                          if isinstance(v, dict) and "delta_max_abs" in v), default=0.0), 6),
                     "fit": {k: res[k] for k in
                             ("optimizer_steps_executed", "effective_samples", "status",
                              "rejected", "loss_first", "loss_last", "grad_norms_first_step",
                              "transient_buffer_drift", "parent_state_sha256",
                              "child_state_sha256", "wall_s")},
                     "child_checkpoint": child_path,
                     "child_reload_conflict_score": reload_score,
                     "reload_consistent": None if reload_score is None
                     else bool(reload_score == b_after["acc_occluded_conflict"]),
                     "rss_mb": rss_mb()}
        print(f"[{arm}] steps={res['optimizer_steps_executed']} "
              f"B_OOD_conflict={b_ood_after['acc_occluded_conflict']} "
              f"(gain {arms[arm]['B_OOD_gain_conflict']}) A={a_after} "
              f"frozen_ok={res['delta']['frozen_all_bitwise_unchanged']} "
              f"{res['wall_s']}s", flush=True)

    # ---- 门槛（与 M4 首轮完全一致，预登记不变）----
    ran = [a for a in ("Neural-only", "Hybrid", "Negative-shuffled")
           if "B_OOD_gain_conflict" in arms.get(a, {})]
    ran_all = len(ran) == 3
    neural_arms = [a for a in ("Neural-only", "Hybrid") if a in ran]
    frozen_violations = sum(len(arms[a]["delta_summary"]["frozen_violations"]) for a in ran)
    best = None
    if neural_arms:
        gains = {a: arms[a]["B_OOD_gain_conflict"] for a in neural_arms
                 if arms[a]["B_OOD_gain_conflict"] is not None}
        if gains:
            best = max(gains, key=lambda a: gains[a])
    matrix = {
        "inheritance_ok": same_parent,
        "frozen_violations": frozen_violations,
        "b_ood_gain": None if best is None else arms[best]["B_OOD_gain_conflict"],
        "frozen_baseline_B_OOD_conflict": b_ood_before["acc_occluded_conflict"],
        "negative_b_ood_gain": arms.get("Negative-shuffled", {}).get("B_OOD_gain_conflict"),
        "a_forgetting": None if best is None else arms[best]["A_forgetting"],
        "neural_beats_frozen": bool(
            best is not None
            and arms[best]["B_OOD"]["acc_occluded_conflict"] is not None
            and b_ood_before["acc_occluded_conflict"] is not None
            and arms[best]["B_OOD"]["acc_occluded_conflict"]
            > b_ood_before["acc_occluded_conflict"]),
        "best_arm": best,
        "memory_only_B_OOD_conflict": arms["Memory-only"]["B_OOD"]["acc_occluded_conflict"],
    }
    gates = check_gates(matrix, b_ood_gain_threshold=GATE_B_OOD, a_drop_max=GATE_A_DROP)
    status = decide_status(gates, ran_all_arms=ran_all)
    audit_path = HERE / "model_promotion_audit.jsonl"
    append_audit(audit_path, {"experiment_id": EXP_ID, "event": "sealed_eval_decision",
                               "candidate_status": status, "stable_sha256": stable_sha,
                               "candidate_sha256": {a: arms[a]["fit"]["child_state_sha256"]
                                                    for a in ("Neural-only", "Hybrid")
                                                    if a in arms},
                               "gates": gates["gates"], "matrix": matrix,
                               "preregistered": {"lambda_anchor": LAMBDA_ANCHOR,
                                                  "lambda_replay": LAMBDA_REPLAY,
                                                  "steps": STEPS}})
    rollbacks = []
    if status == "REJECTED":
        quarantine = HERE / "ckpts" / "rejected"
        quarantine.mkdir(parents=True, exist_ok=True)
        for arm, p in candidate_paths.items():
            src = Path(p)
            if src.exists():
                dst = quarantine / src.name
                src.rename(dst)
                rollbacks.append({"arm": arm, "action": "QUARANTINE", "moved_to": str(dst),
                                   "stable_sha256_unchanged": file_sha256(
                                       ROOT / "experiments" / "m2_ablation"
                                       / "C_sparse_ce_s20261131.pt") == stable_sha})
        append_audit(audit_path, {"experiment_id": EXP_ID, "event": "rollback",
                                   "detail": rollbacks})
    elif status == "PROPOSED":
        append_audit(audit_path, {"experiment_id": EXP_ID,
                                   "event": "staged_promotion_pending",
                                   "note": "候选保留于 ckpts/ 等待人工批准；stable 未覆盖",
                                   "candidates": candidate_paths,
                                   "next_step_proposal": "3-seed 复验需小袁批准"})

    payload = {
        "experiment_id": EXP_ID, "captured_at": now_iso(),
        "device": "cpu", "torch_threads": int(torch.get_num_threads()),
        "stable_state_sha256": stable_sha,
        "preregistered": {"lambda_anchor": LAMBDA_ANCHOR, "lambda_replay": LAMBDA_REPLAY,
                           "steps": STEPS, "scope": SCOPE, "replay_k": REPLAY_K,
                           "budget_cap": BUDGET_CAP,
                           "seeds": {"A": SEED_A, "B_train": SEED_B_TRAIN,
                                      "B_test": SEED_B_TEST, "B_OOD": SEED_B_OOD,
                                      "sampling": SAMPLING},
                           "gates": gates["thresholds"],
                           "change_points_vs_m4_smoke": ["lambda_anchor 0.1→0",
                                                          "lambda_replay 0.3→0.15",
                                                          "steps 40→48（神经臂）"],
                           "authorized_by": "小袁 2026-11-22 指令（M4.1 第一阶段）"},
        "decision": decision, "budget": {"cap": BUDGET_CAP, "used": budget.used,
                                          "log": budget.log},
        "leakage_audit": leakage_audit,
        "A_before_occ": a_before, "B_before": b_before, "B_OOD_before": b_ood_before,
        "arms": arms, "matrix": matrix, "gates": gates,
        "candidate_status": status, "rollbacks": rollbacks,
        "comparison_base": "m4-online-smoke-001（同一父/数据/协议，仅 λ 与步数不同）",
        "wall_s_total": round(time.perf_counter() - t0, 1), "wall_limit_s": WALL_LIMIT_S,
        "rss_mb_end": rss_mb(),
        "honesty": "1 seed；buffer 复位协议严格执行（漂移单独记录，不冒充学习）；"
                    "通过后 3-seed 复验需另行批准",
        "wall_ok": (time.perf_counter() - t0) <= WALL_LIMIT_S,
    }
    (HERE / "m41_stability.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (HERE / "benchmark_matrix_m41.json").write_text(json.dumps(
        {"experiment_id": EXP_ID, "matrix": matrix, "arms": arms, "gates": gates,
         "candidate_status": status}, ensure_ascii=False, indent=2), encoding="utf-8")
    (HERE / "experience_manifest_m41.json").write_text(json.dumps(
        {"experiment_id": EXP_ID, "provenance": provenance, "rule_fields": rule_fields,
         "verifier": verdict, "leakage_audit": leakage_audit,
         "experience_counts": {"stream_train_verified": len(stream), "sealed_eval": 64,
                                "rejected": 1, "uncertain": 1},
         "experiences_jsonl_sha256": file_sha256(HERE / "experiences_m41.jsonl")},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"candidate_status": status, "gates": gates["gates"],
                      "best_arm": best, "budget_used": budget.used,
                      "wall_s": payload["wall_s_total"]}, ensure_ascii=False, indent=2),
          flush=True)
    print(f"→ {HERE / 'm41_stability.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
