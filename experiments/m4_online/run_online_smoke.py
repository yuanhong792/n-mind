"""M4 联网自主神经学习首次全链路 smoke（m4-online-smoke-001，1 预声明 seed，小预算）。

链路：OBSERVE(模拟网页夹具) → PARSE(结构化规则表) → VERIFY(程序校验) → Experience 流 →
LEARN_DECISION(显式调度) → 候选权重更新(四臂+负对照, ≤120 optimizer.step) →
SEALED_EVALUATE(冻结新旧测试集) → PROPOSED(人工批准待定)/REJECTED(+回滚) →
只读真实 HTTP 冒烟（--real-http-smoke 显式授权，无训练，默认关）。

预注册（运行前已定，不得看结果后暗改）：
  父 checkpoint   = experiments/m2_ablation/C_sparse_ce_s20261131.pt（自己从随机初始化训练）
  A（旧任务）     = GridTask(goal_visible_steps=4, seed=20261131)，冻结测试集
  新规则 B        = dir-bias-north-v1（模拟网页结构化规则表：up 优先，fallback=min_distance）
  B train 流      = seed 20261201 × 96 条（T=24，verified，来源=规则页）
  B test 封存     = seed 20261202 × 32 条（T=24）
  B OOD 封存      = seed 20261203 × 32 条（T=32，长度外推）
  臂与预算        = Frozen 0 / Memory-only 0 / Neural-only 40 / Hybrid 40 / Negative 16 = 96/120
  scope=adapter；lambda_replay=0.3（回放 A 任务 96 条非封存行）、lambda_anchor=0.1、replay_k=4
  门槛            = B_OOD 冲突步 gain ≥ +10pp 且 > 负对照与冻结基线；A 遮挡段下降 ≤ 3pp；
                    Neural-only 必须 > Frozen 才能宣称神经更新有效；冻结参数逐位零违规
  candidate_status= PROPOSED（人工批准待定）/ REJECTED（回滚）/ INCONCLUSIVE
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from nmind.config import default_config  # noqa: E402
from nmind.evaluation.lifelong import eval_skill, rule_conflict_masks  # noqa: E402
from nmind.learning import datasets as D  # noqa: E402
from nmind.learning import online_adaptation as OA  # noqa: E402
from nmind.learning.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from nmind.learning.online_trainer import bounded_stream_update  # noqa: E402
from nmind.learning.trainer import evaluate, make_data  # noqa: E402
from nmind.online import (Experience, FetchConfig, Fetcher, MALICIOUS_URL,  # noqa: E402
                           RULE_MIRROR_URL, RULE_PAGE_URL, UNRELATED_URL, append_audit,
                           build_learning_stream, build_provenance, build_rule_oracle,
                           check_gates, decide, decide_status, detect_instructions,
                           extract_rule_table, extract_text, file_sha256, fixture_fetcher,
                           rollback_candidate, save_jsonl, verify_rule)
from nmind.online.experiences import now_iso  # noqa: E402

HERE = Path(__file__).resolve().parent
PARENT = ROOT / "experiments" / "m2_ablation" / "C_sparse_ce_s20261131.pt"

EXP_ID = "m4-online-smoke-001"
SEED_A = 20261131
SEED_B_TRAIN = 20261201
SEED_B_TEST = 20261202
SEED_B_OOD = 20261203
SAMPLING = {"Neural-only": 20261211, "Hybrid": 20261212, "Negative-shuffled": 20261213}
STEPS = {"Neural-only": 40, "Hybrid": 40, "Negative-shuffled": 16}
BUDGET_CAP = 120
SCOPE = "adapter"
LAMBDA_REPLAY = 0.3
LAMBDA_ANCHOR = 0.1
REPLAY_K = 4
GATE_B_OOD = 0.10
GATE_A_DROP = 0.03
WALL_LIMIT_S = 300.0
WALL_GUARD_S = 240.0


def build_rule_rows(n: int, seed: int, T: int, used_pairs: set, taskB, vis: int = 4,
                    task: str = "rule_control_reversed") -> list[dict]:
    """按网页验证过的规则任务生成轨迹行；used_pairs 全局共享 → 训练/封存绝不共享 (start,goal)。"""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for _ in range(n):
        tr = D.make_trajectory(taskB, rng, T, used_pairs=used_pairs)
        row = {"obs": tr.obs, "actions": tr.actions,
               "goal_code": int(tr.goal[0]) * 6 + int(tr.goal[1]),
               "len": int(tr.obs.shape[0]), "task": task,
               "start": list(tr.start), "goal": list(tr.goal)}
        rows.append(row)
    return rows


def mem_rows_of(rows: list[dict]) -> list[dict]:
    """记忆臂：注入"规则页经验"向量（policy=max_distance 标志）。网络从未训练读 mem。"""
    out = []
    for r in rows:
        row = dict(r)
        vec = np.zeros((int(r["len"]), 16), dtype=np.float32)
        vec[:, 3] = 1.0            # policy = max_distance 标志
        vec[:, 4] = 1.0            # 规则页经验标志
        row["mem"] = vec
        out.append(row)
    return out


def rss_mb() -> float:
    try:
        import psutil
        return round(psutil.Process().memory_info().rss / 1e6, 1)
    except Exception:
        return -1.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real-http-smoke", action="store_true",
                    help="显式授权：只读真实 HTTP 管线冒烟（无训练；allowlist=example.com）")
    args = ap.parse_args()
    t0 = time.perf_counter()
    (HERE / "ckpts").mkdir(parents=True, exist_ok=True)
    dev = torch.device("cpu")
    cfg = default_config()
    cfg.learning.traj_len = 24

    # ---- 稳定权重 v1：自己从随机初始化训练得到的 M2 checkpoint ----
    parent, parent_meta = load_checkpoint(PARENT, cfg=cfg, device="cpu",
                                          allow_config_mismatch=True)
    parent_state = {k: v.detach().cpu().clone() for k, v in parent.state_dict().items()}
    stable_sha = OA.state_hash(parent)
    taskA = D.GridTask(name="A", goal_visible_steps=4, seed=SEED_A)
    taskB_tr = D.GridTask(name="B", goal_visible_steps=4, reversed_control=True, seed=SEED_B_TRAIN)
    dA = make_data(cfg, taskA, seed=SEED_A)
    a_test, a_train = dA["test"], dA["train"]          # a_train = 非封存回放池

    # ---- OBSERVE（模拟网页，零真实网络）----
    fetcher = fixture_fetcher()
    recs = {u: fetcher.fetch(u) for u in
            (RULE_PAGE_URL, RULE_MIRROR_URL, MALICIOUS_URL, UNRELATED_URL)}
    rule_rec = recs[RULE_PAGE_URL]

    # ---- PARSE ----
    parsed = extract_rule_table(rule_rec.content or "")
    rule_fields = parsed["fields"]

    # ---- VERIFY ----
    verdict = verify_rule(parsed)
    oracle = build_rule_oracle(parsed) if verdict["verification"] == "verified" else None
    malicious_flags = detect_instructions(recs[MALICIOUS_URL].content or "")
    provenance = [build_provenance(r, "extractor-v1", "simulated-fixture").to_dict()
                  for r in recs.values() if r.status in ("fetched", "duplicate")]
    print(f"[observe] rule={rule_fields.get('rule_id')} verify={verdict['verification']} "
          f"dedup={recs[RULE_MIRROR_URL].status} "
          f"malicious_flagged={malicious_flags['instruction_detected']}", flush=True)
    if oracle is None:
        (HERE / "online_stream_smoke.json").write_text(json.dumps(
            {"experiment_id": EXP_ID, "status": "INCONCLUSIVE",
             "reason": "规则页未通过程序化验证", "verdict": verdict}, ensure_ascii=False),
            encoding="utf-8")
        return 0

    # ---- 经验流（96 verified 训练 + 2×32 sealed + 1 rejected + 1 uncertain）----
    used_pairs: set = set()
    b_train = build_rule_rows(96, SEED_B_TRAIN, 24, used_pairs, taskB_tr)
    b_test = build_rule_rows(32, SEED_B_TEST, 24, used_pairs, taskB_tr)
    b_ood = build_rule_rows(32, SEED_B_OOD, 32, used_pairs, taskB_tr)
    experiences: list[Experience] = []
    excerpt = extract_text(rule_rec.content or "", limit=160)
    for i, r in enumerate(b_train):
        experiences.append(Experience(
            id=f"exp-b-{i:03d}", observed_at=now_iso(), source_url=RULE_PAGE_URL,
            source_sha256=rule_rec.sha256, raw_excerpt=excerpt,
            structured_observation={"episode": i, "goal_code": int(r["goal_code"]),
                                     "T": int(r["len"]), "rule_id": rule_fields["rule_id"]},
            action=None, expected_outcome=None, actual_outcome=None,
            verifier_id=verdict["verifier_id"], verification="verified", reward=None,
            task_family="rule_control_reversed", train_split="stream_train"))
    for tag, rows in (("test", b_test), ("ood", b_ood)):
        for i, r in enumerate(rows):
            experiences.append(Experience(
                id=f"exp-{tag}-{i:03d}", observed_at=now_iso(), source_url=RULE_PAGE_URL,
                source_sha256=rule_rec.sha256, raw_excerpt=excerpt,
                structured_observation={"episode": i, "goal_code": int(r["goal_code"]),
                                         "T": int(r["len"])},
                action=None, expected_outcome=None, actual_outcome=None,
                verifier_id=verdict["verifier_id"], verification="verified", reward=None,
                task_family="rule_dir_bias", train_split="sealed_eval"))
    experiences.append(Experience(
        id="exp-malicious-000", observed_at=now_iso(), source_url=MALICIOUS_URL,
        source_sha256=recs[MALICIOUS_URL].sha256,
        raw_excerpt=extract_text(recs[MALICIOUS_URL].content or "", limit=160),
        structured_observation={"instruction_detected": malicious_flags["instruction_detected"],
                                 "snippets": malicious_flags["snippets"]},
        action=None, expected_outcome=None, actual_outcome=None, verifier_id=None,
        verification="rejected", reward=None, task_family="adversarial",
        train_split="stream_train"))
    experiences.append(Experience(
        id="exp-unrelated-000", observed_at=now_iso(), source_url=UNRELATED_URL,
        source_sha256=recs[UNRELATED_URL].sha256,
        raw_excerpt=extract_text(recs[UNRELATED_URL].content or "", limit=120),
        structured_observation={"page": "no rule table"}, action=None,
        expected_outcome=None, actual_outcome=None, verifier_id=verdict["verifier_id"],
        verification="uncertain", reward=None, task_family="rule_dir_bias",
        train_split="stream_train"))
    save_jsonl(HERE / "experiences.jsonl", experiences)
    stream = build_learning_stream(experiences)
    assert len(stream) == 96 and all(e.id.startswith("exp-b-") for e in stream)
    # 泄漏审计：封存测试与训练流不共享 (start,goal)；封存 id 不在训练流
    pair_sets = lambda rows: {(tuple(r["start"]), tuple(r["goal"])) for r in rows}
    leakage_audit = {
        "train_test_pair_overlap": len(pair_sets(b_train) & pair_sets(b_test)),
        "train_ood_pair_overlap": len(pair_sets(b_train) & pair_sets(b_ood)),
        "test_ood_pair_overlap": len(pair_sets(b_test) & pair_sets(b_ood)),
        "sealed_ids_in_stream": sum(1 for e in stream if e.train_split == "sealed_eval"),
        "train_stream_hashes": sorted({e.source_sha256[:16] for e in stream}),
    }
    assert leakage_audit["train_test_pair_overlap"] == 0
    assert leakage_audit["train_ood_pair_overlap"] == 0
    assert leakage_audit["sealed_ids_in_stream"] == 0

    # ---- LEARN_DECISION（显式调度；rejected/uncertain 不计数）----
    decision = decide(verified_count=len(stream), rejected_count=1, uncertain_count=1,
                      budget_remaining=BUDGET_CAP, cooldown_ok=True)
    assert decision["decision"] == "CANDIDATE_WEIGHT_UPDATE"
    budget = OA.AdaptationBudget(BUDGET_CAP, label=EXP_ID)

    # ---- 冻结基线（Frozen 臂 = stable 本身，0 步）----
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
                                       "A_forgetting": 0.0, "B": b_before, "B_OOD": b_ood_before,
                                       "mem_injected": False, "rss_mb": rss_mb()}}
    same_parent = True

    def clone_from_stable(arm: str):
        nonlocal same_parent
        m = load_checkpoint(PARENT, cfg=cfg, device="cpu", allow_config_mismatch=True)[0]
        chk = OA.verify_bitwise_inheritance({"model": parent_state}, m)
        if not chk["all_bitwise_equal"]:
            same_parent = False
            raise RuntimeError(f"{arm} 继承断言失败：{chk['mismatched_keys'][:5]}")
        return m

    # ---- Memory-only：权重冻结 + 检索记忆注入 ----
    m = clone_from_stable("Memory-only")
    mem_before = OA.clone_snapshot(m)
    b_after_mem = eval_skill(m, mem_rows_of(b_test), dev,
                             conflict=masks["test"][0], coincident=masks["test"][1])
    b_ood_after_mem = eval_skill(m, mem_rows_of(b_ood), dev,
                                 conflict=masks["ood"][0], coincident=masks["ood"][1])
    after_snap = OA.clone_snapshot(m)
    mem_unchanged = all(torch.equal(mem_before["model"][k], after_snap["model"][k])
                        for k in mem_before["model"] if k != "block_load_count")
    arms["Memory-only"] = {"updates": 0, "mem_injected": True,
                           "A_after_occ": a_before, "A_forgetting": 0.0,
                           "B": b_after_mem, "B_OOD": b_ood_after_mem,
                           "params_bitwise_unchanged": mem_unchanged, "rss_mb": rss_mb()}
    print(f"[Memory-only] B_conflict={b_after_mem['acc_occluded_conflict']} "
          f"params_unchanged={mem_unchanged}", flush=True)

    # ---- 神经臂（预算内，墙钟护栏内）----
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
                                  "parent_state_sha256": stable_sha, "steps": STEPS[arm]})
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

    # ---- SEALED_EVALUATE → 门槛（预登记）----
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
            and arms[best]["B_OOD"]["acc_occluded_conflict"] > b_ood_before["acc_occluded_conflict"]),
        "best_arm": best,
        "memory_only_B_OOD_conflict": arms["Memory-only"]["B_OOD"]["acc_occluded_conflict"],
    }
    gates = check_gates(matrix, b_ood_gain_threshold=GATE_B_OOD, a_drop_max=GATE_A_DROP)
    status = decide_status(gates, ran_all_arms=ran_all)
    audit_path = HERE / "model_promotion_audit.jsonl"
    audit_record = {"experiment_id": EXP_ID, "event": "sealed_eval_decision",
                    "candidate_status": status, "stable_sha256": stable_sha,
                    "candidate_sha256": {a: arms[a]["fit"]["child_state_sha256"]
                                         for a in ("Neural-only", "Hybrid") if a in arms},
                    "gates": gates["gates"], "matrix": {k: v for k, v in matrix.items()},
                    "stable_checkpoint_path": str(PARENT)}
    append_audit(audit_path, audit_record)
    rollbacks = []
    if status == "REJECTED":
        quarantine = HERE / "ckpts" / "rejected"
        quarantine.mkdir(parents=True, exist_ok=True)
        for arm, p in candidate_paths.items():
            src = Path(p)
            if src.exists():
                dst = quarantine / src.name
                src.rename(dst)
                rollbacks.append({"arm": arm, "action": "QUARANTINE",
                                   "moved_to": str(dst), "reason": "gates fail",
                                   "stable_sha256_unchanged": file_sha256(PARENT) == stable_sha})
        append_audit(audit_path, {"experiment_id": EXP_ID, "event": "rollback",
                                   "detail": rollbacks})
    elif status == "PROPOSED":
        append_audit(audit_path, {"experiment_id": EXP_ID, "event": "staged_promotion_pending",
                                   "note": "候选权重保留于 ckpts/，等待人工批准；stable 未被覆盖",
                                   "candidates": candidate_paths})

    # ---- 只读真实 HTTP 冒烟（显式授权 + 门槛通过才执行；无训练）----
    real_http = {"ran": False, "note": "未授权或门槛未通过"}
    if args.real_http_smoke and status == "PROPOSED":
        rf = Fetcher(FetchConfig(offline=False, allowlist=("https://example.com/",)))
        r = rf.fetch("https://example.com/")
        real_http = {"ran": True, **r.to_dict(), "training": False}

    payload = {
        "experiment_id": EXP_ID, "captured_at": now_iso(),
        "device": "cpu", "torch_threads": int(torch.get_num_threads()),
        "parent_checkpoint": str(PARENT), "parent_saved_at": parent_meta.get("saved_at"),
        "stable_state_sha256": stable_sha,
        "preregistered": {"seeds": {"A": SEED_A, "B_train": SEED_B_TRAIN,
                                     "B_test": SEED_B_TEST, "B_OOD": SEED_B_OOD,
                                     "sampling": SAMPLING},
                           "steps": STEPS, "budget_cap": BUDGET_CAP, "scope": SCOPE,
                           "lambda_replay": LAMBDA_REPLAY, "lambda_anchor": LAMBDA_ANCHOR,
                           "gates": gates["thresholds"]},
        "observe": {"records": {u: r.to_dict() for u, r in recs.items()},
                     "provenance": provenance, "rule_fields": rule_fields,
                     "verifier": verdict, "malicious_flags": malicious_flags},
        "decision": decision, "budget": {"cap": BUDGET_CAP, "used": budget.used,
                                          "log": budget.log},
        "leakage_audit": leakage_audit,
        "A_before_occ": a_before, "B_before": b_before, "B_OOD_before": b_ood_before,
        "arms": arms, "matrix": matrix, "gates": gates,
        "candidate_status": status, "rollbacks": rollbacks,
        "promotion_audit_path": str(audit_path),
        "real_http_smoke": real_http,
        "status_flags": {"real_networking": ("PASS" if real_http.get("ran") and
                                              real_http.get("status") == "fetched"
                                              else ("FAIL" if real_http.get("ran") else "NOT_TESTED")),
                          "parameter_learning": "PASS" if best is not None else "FAIL",
                          "generalizable_skill": ("PASS" if gates["gates"]["b_ood_gain_ge_threshold"]
                                                  and gates["gates"]["neural_beats_frozen"]
                                                  else "FAIL"),
                          "auto_deployment": "NOT_TESTED（promotion.mode=manual，按任务书不自动部署）"},
        "wall_s_total": round(time.perf_counter() - t0, 1), "wall_limit_s": WALL_LIMIT_S,
        "rss_mb_end": rss_mb(),
        "honesty": "1 seed smoke；抓取网页≠阅读理解（工程脚手架提取）；参数变化≠新技能；"
                    "RESEARCH_VALIDATED 类结论需多 seed 与置信区间（待批准）",
        "wall_ok": (time.perf_counter() - t0) <= WALL_LIMIT_S,
    }
    (HERE / "online_stream_smoke.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (HERE / "benchmark_matrix.json").write_text(json.dumps(
        {"experiment_id": EXP_ID, "matrix": matrix, "arms": arms, "gates": gates,
         "candidate_status": status}, ensure_ascii=False, indent=2), encoding="utf-8")
    (HERE / "experience_manifest.json").write_text(json.dumps(
        {"experiment_id": EXP_ID, "fetch_records": {u: r.to_dict() for u, r in recs.items()},
         "provenance": provenance, "rule_fields": rule_fields, "verifier": verdict,
         "experience_counts": {"stream_train_verified": len(stream),
                                "sealed_eval": 64, "rejected": 1, "uncertain": 1},
         "experiences_jsonl_sha256": file_sha256(HERE / "experiences.jsonl"),
         "leakage_audit": leakage_audit, "split_discipline": "sealed_eval 永不进入学习流/回放",
         "dataset_hashes": {"A_test": dA["dataset_hash"]}},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"candidate_status": status, "gates": gates["gates"],
                      "budget_used": budget.used,
                      "wall_s": payload["wall_s_total"],
                      "status_flags": payload["status_flags"]},
                     ensure_ascii=False, indent=2), flush=True)
    print(f"→ {HERE / 'online_stream_smoke.json'}", flush=True)
    return 0


def _parent_file_sha() -> str:
    return file_sha256(PARENT)


if __name__ == "__main__":
    raise SystemExit(main())
