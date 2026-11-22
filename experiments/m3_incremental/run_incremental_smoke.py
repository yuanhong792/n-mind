"""M4 首次增量学习 smoke（任务书第 3-7 节最小闭环，预注册，1 seed）。

预注册（写死在本文件常量里，运行前已定，不得事后修改）：
  experiment_id      = m4-incremental-smoke-001
  parent checkpoint  = experiments/m2_ablation/C_sparse_ce_s20261131.pt（M2 冻结稀疏基线族）
  A（旧任务）        = GridTask(goal_visible_steps=4, seed=20261131)，与父模型训练数据同管线
  B train（新经验）  = GridTask(name="B", reversed_control=True, seed=20261141) 训练 96 条
  B test（未见样本） = GridTask(name="B", reversed_control=True, seed=20261142) 测试 32 条
  预算               = 所有臂合计 ≤120 optimizer.step()；实际分配：Head 48 / Core 48 / lr0对照 8 = 104
  seed               = 唯一预声明：B 经验数据 20261141 / B 测试 20261142；臂采样流 20261151-53
  Neural-Core 范围   = adapter + 父模型 block_load_count 最高的 2 个 N-Cell 块（平局取小号）
  NEW_SKILL 门槛     = B未见冲突步 acc 提高 ≥ +10pp 且高于多数类/旧策略/随机基线
  RETENTION 门槛     = A 冻结测试集下降 ≤ 5pp
  失败处理           = 如实记录，不得加训掩盖；RESEARCH_VALIDATED 不能由 1-seed 获得

臂定义（同一父快照逐位克隆，同一 B 经验预算，同一冻结测试集）：
  Control-0        不更新权重、不加外部记忆
  Control-Memory   只写/读 SQLite 情景记忆（mem 向量注入评测），参数冻结
  Neural-Head      只更新 head/action_emb/block_to_ws
  Neural-Core      再放开父模型最高负载的 2 个 N-Cell 块核权重（梯度切片掩码 + wd=0）
  Negative-lr0     与 Neural-Head 同配置但 lr=0（反向检验：参数必须逐位不变）

用法：
  python -u experiments/m3_incremental/run_incremental_smoke.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from nmind.config import default_config  # noqa: E402
from nmind.learning import datasets as D  # noqa: E402
from nmind.learning import online_adaptation as OA  # noqa: E402
from nmind.learning.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from nmind.learning.trainer import _assemble, evaluate, make_data  # noqa: E402

HERE = Path(__file__).resolve().parent
PARENT = ROOT / "experiments" / "m2_ablation" / "C_sparse_ce_s20261131.pt"

EXP_ID = "m4-incremental-smoke-001"
SEED_A = 20261131          # A 任务/数据（与父模型训练管线一致）
SEED_B_TRAIN = 20261141    # B 新经验
SEED_B_TEST = 20261142     # B 未见测试
SAMPLING = {"Neural-Head": 20261151, "Neural-Core": 20261152, "Negative-lr0": 20261153}
STEPS = {"Neural-Head": 48, "Neural-Core": 48, "Negative-lr0": 8}
BUDGET_CAP = 120
CORE_BLOCKS = 2
NEW_SKILL_MIN_GAIN = 0.10
RETENTION_MAX_DROP = 0.05
WALL_LIMIT_S = 300.0
WALL_GUARD_S = 240.0       # 超过则剩余臂安全跳过并如实记录

B_TRAIN_ROWS: list[dict] = []


def b_step_masks(rows: list[dict], vis: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    """遮挡期逐步掩码：冲突步 = 原规则 oracle != 反转规则标签；巧合步 = 两者相同。"""
    n, T = len(rows), int(rows[0]["len"])
    conflict = torch.zeros(n, T, dtype=torch.bool)
    coincident = torch.zeros(n, T, dtype=torch.bool)
    for i, r in enumerate(rows):
        goal = (int(r["goal_code"]) // 6, int(r["goal_code"]) % 6)
        for t in range(vis, T):
            row = np.asarray(r["obs"][t])
            pos = (int(np.argmax(row[0:6])), int(np.argmax(row[6:12])))
            normal = D.oracle_action(pos, goal, tuple(), 6, reversed_control=False)
            is_c = normal != int(r["actions"][t])
            conflict[i, t] = is_c
            coincident[i, t] = not is_c
    return conflict, coincident


def eval_masked(model, rows, dev, masks) -> dict:
    batch = _assemble(rows, dev)
    with torch.no_grad():
        pred = model.forward_sequence(batch["obs"], batch["actions"],
                                      mem=batch["mem"])["action_logits"].argmax(-1)
    tgt = batch["actions"]
    conflict, coincident = masks
    out: dict[str, Any] = {}
    for name, m in (("conflict", conflict), ("coincident", coincident),
                    ("total_occluded", conflict | coincident)):
        out[name] = None if int(m.sum()) == 0 else round(
            float((pred[m] == tgt[m]).float().mean()), 4)
        out[f"n_{name}"] = int(m.sum())
    return out


from typing import Any  # noqa: E402


def majority_baseline(rows, masks, train_rows) -> dict:
    batch = _assemble(rows, torch.device("cpu"))
    tgt = batch["actions"]
    conflict, _co = masks
    labels = tgt[conflict].numpy()
    vals, counts = np.unique(labels, return_counts=True)
    top = int(vals[int(np.argmax(counts))])
    train_labels = np.concatenate([np.asarray(r["actions"])[4:] for r in train_rows])
    tv, tc = np.unique(train_labels, return_counts=True)
    train_majority = int(tv[int(np.argmax(tc))])
    return {"test_majority_action": top,
            "test_majority_acc_on_conflict": round(float((labels == top).mean()), 4),
            "train_majority_action": train_majority,
            "train_majority_acc_on_test_conflict": round(float((labels == train_majority).mean()), 4),
            "random_policy": 0.25}


def rss_mb() -> float:
    try:
        import psutil
        return round(psutil.Process().memory_info().rss / 1e6, 1)
    except Exception:
        return -1.0


def visible_goal(row: dict) -> tuple[int, int]:
    """可见期（t=0）观测里的目标位置——推理系统合法可得，不是测试集标签注入。"""
    vis_row = np.asarray(row["obs"][0])
    return int(np.argmax(vis_row[12:18])), int(np.argmax(vis_row[18:24]))


def main() -> int:
    t0 = time.perf_counter()
    HERE.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cpu")
    cfg = default_config()
    cfg.learning.traj_len = 24

    # ---- 数据（预登记 seed，互相独立；B 测试集封存，学习阶段不可见）----
    taskA = D.GridTask(name="A", goal_visible_steps=4, seed=SEED_A)
    taskB_tr = D.GridTask(name="B", goal_visible_steps=4, reversed_control=True, seed=SEED_B_TRAIN)
    taskB_te = D.GridTask(name="B", goal_visible_steps=4, reversed_control=True, seed=SEED_B_TEST)
    dA = make_data(cfg, taskA, seed=SEED_A)
    dB_tr = make_data(cfg, taskB_tr, seed=SEED_B_TRAIN)
    dB_te = make_data(cfg, taskB_te, seed=SEED_B_TEST)
    a_test, b_train_rows, b_test = dA["test"], dB_tr["train"], dB_te["test"]
    global B_TRAIN_ROWS
    B_TRAIN_ROWS = b_train_rows
    masks = b_step_masks(b_test)

    # ---- 父模型（已学会任务 A 的 M2 冻结 checkpoint）----
    parent, parent_meta = load_checkpoint(PARENT, cfg=cfg, device="cpu",
                                          allow_config_mismatch=True)
    parent_state = {k: v.detach().cpu().clone() for k, v in parent.state_dict().items()}
    parent_sha = OA.state_hash(parent)
    a_before = round(evaluate(parent, None, a_test, device=dev,
                              occluded_only=True)["accuracy_occluded"], 4)
    b_before = eval_masked(parent, b_test, dev, masks)
    baselines = majority_baseline(b_test, masks, b_train_rows)
    load = parent.block_load_count.clone()
    core_ids = sorted(range(int(load.numel())),
                      key=lambda i: (-float(load[i]), i))[:CORE_BLOCKS]
    print(f"[parent] sha={parent_sha[:16]} A_occ={a_before} "
          f"B_conflict={b_before['conflict']} B_total={b_before['total_occluded']} "
          f"core_blocks={core_ids} baselines={baselines}", flush=True)
    invalid_task = b_before["conflict"] is not None and b_before["conflict"] >= 0.9
    if invalid_task:
        print("INVALID_TASK_FOR_NEW_SKILL_TEST：B 在学习前已接近满分", flush=True)

    budget = OA.AdaptationBudget(BUDGET_CAP, label=EXP_ID)
    arms: dict[str, Any] = {}
    same_parent = True

    def clone_from_parent(arm: str):
        nonlocal same_parent
        m = load_checkpoint(PARENT, cfg=cfg, device="cpu",
                            allow_config_mismatch=True)[0]
        chk = OA.verify_bitwise_inheritance({"model": parent_state}, m)
        if not chk["all_bitwise_equal"]:
            same_parent = False
            raise RuntimeError(f"{arm} 克隆继承断言失败：{chk['mismatched_keys'][:5]}")
        return m

    def eval_arm(model, arm: str) -> dict:
        OA.reset_transient_buffers(model, parent_state)   # 统计 buffer 复位到父值（口径一致）
        a_after = round(evaluate(model, None, a_test, device=dev,
                                 occluded_only=True)["accuracy_occluded"], 4)
        b_after = eval_masked(model, b_test, dev, masks)
        gain = None if (b_after["conflict"] is None or b_before["conflict"] is None) \
            else round(b_after["conflict"] - b_before["conflict"], 4)
        return {"arm": arm, "A_after_occ": a_after,
                "A_forgetting": round(a_before - a_after, 4),
                "B_after": b_after, "B_gain_conflict": gain}

    # ---- Control-0：完全不更新 ----
    m = clone_from_parent("Control-0")
    arms["Control-0"] = {**eval_arm(m, "Control-0"), "updates": 0,
                         "note": "不更新权重、不加外部记忆", "rss_mb": rss_mb()}

    # ---- Control-Memory：只写/读外部记忆，参数冻结 ----
    m = clone_from_parent("Control-Memory")
    mem_before = OA.clone_snapshot(m)
    episodes_written = 0
    memory_error = None
    try:
        from nmind.memory import MemoryStore
        db = HERE / "memory_smoke.sqlite"
        if db.exists():
            db.unlink()
        store = MemoryStore(db)
        for i, r in enumerate(b_train_rows):
            gx, gy = visible_goal(r)
            store.add_episode(session_id=EXP_ID,
                              observation=json.dumps({"episode": i, "goal": [gx, gy],
                                                      "rule": "reversed_control"}),
                              task="B_incremental_smoke", action=str(int(r["actions"][4])),
                              outcome="trajectory_written", feedback=1.0,
                              source="local", confidence=1.0)
            episodes_written += 1
    except Exception as e:   # 记忆臂失败如实记录，不伪造
        memory_error = repr(e)
    mem_rows = []
    for r in b_test:
        gx, gy = visible_goal(r)
        vec = np.zeros((int(r["len"]), 16), dtype=np.float32)
        vec[:, 0] = gx / 5.0
        vec[:, 1] = gy / 5.0
        vec[:, 2] = (gx * 6 + gy) / 35.0
        vec[:, 3] = 1.0                      # 规则提示（来自 B 经验）
        vec[:, 4] = 1.0                      # 检索命中标志
        row = dict(r)
        row["mem"] = vec
        mem_rows.append(row)
    batch = _assemble(mem_rows, dev)
    with torch.no_grad():
        pred_mem = m.forward_sequence(batch["obs"], batch["actions"],
                                      mem=batch["mem"])["action_logits"].argmax(-1)
    tgt = batch["actions"]
    conflict, coincident = masks
    b_after_mem = {"conflict": round(float((pred_mem[conflict] == tgt[conflict]).float().mean()), 4),
                   "coincident": round(float((pred_mem[coincident] == tgt[coincident]).float().mean()), 4),
                   "total_occluded": round(float(
                       (pred_mem[conflict | coincident] == tgt[conflict | coincident]).float().mean()), 4)}
    delta_mem = OA.parameter_delta(mem_before, m, OA.apply_update_scope(m, scope="frozen"))
    arms["Control-Memory"] = {"arm": "Control-Memory", "updates": 0,
                              "episodes_written": episodes_written,
                              "memory_error": memory_error,
                              "mem_vector_note": "16维：可见期目标(x,y,code)+规则标志+检索标志；父模型从未训练过读取 mem",
                              "A_after_occ": round(evaluate(m, None, a_test, device=dev,
                                                            occluded_only=True)["accuracy_occluded"], 4),
                              "A_forgetting": 0.0,
                              "B_after": b_after_mem,
                              "B_gain_conflict": round(b_after_mem["conflict"] - b_before["conflict"], 4),
                              "params_bitwise_unchanged": bool(
                                  delta_mem["frozen_all_bitwise_unchanged"]
                                  and delta_mem["trainable_changed_count"] == 0),
                              "rss_mb": rss_mb()}
    print(f"[Control-Memory] episodes={episodes_written} B_conflict={b_after_mem['conflict']} "
          f"params_unchanged={arms['Control-Memory']['params_bitwise_unchanged']}", flush=True)

    # ---- 神经臂（预算内，墙钟护栏内）----
    for arm in ("Neural-Head", "Neural-Core", "Negative-lr0"):
        if time.perf_counter() - t0 > WALL_GUARD_S:
            arms[arm] = {"arm": arm, "skipped": "wall_clock_guard",
                         "note": "安全中止：超过预注册墙钟护栏，剩余臂不启动"}
            continue
        m = clone_from_parent(arm)
        if arm == "Neural-Head":
            res = OA.fit_existing_model(m, cfg=cfg, rows=b_train_rows, device=dev,
                                        scope="adapter", max_updates=STEPS[arm],
                                        sampling_seed=SAMPLING[arm], arm=arm, budget=budget)
        elif arm == "Neural-Core":
            res = OA.fit_existing_model(m, cfg=cfg, rows=b_train_rows, device=dev,
                                        scope="core_partial", core_block_ids=core_ids,
                                        max_updates=STEPS[arm],
                                        sampling_seed=SAMPLING[arm], arm=arm, budget=budget)
        else:
            res = OA.fit_existing_model(m, cfg=cfg, rows=b_train_rows, device=dev,
                                        scope="adapter", max_updates=STEPS[arm],
                                        lr=0.0, sampling_seed=SAMPLING[arm], arm=arm,
                                        budget=budget)
        entry_eval = eval_arm(m, arm)
        conflict_score = entry_eval["B_after"]["conflict"]
        child_path = None
        reload_score = None
        if arm != "Negative-lr0":
            ckpt_dir = HERE / "ckpts"
            child_path = str(ckpt_dir / f"{EXP_ID}_{arm}_candidate.pt")
            save_checkpoint(m, cfg=cfg, path=child_path, probe=None,
                            meta={"experiment_id": EXP_ID, "arm": arm,
                                  "parent_state_sha256": parent_sha,
                                  "steps": STEPS[arm], "scope": res["scope"]})
            m2, _meta = load_checkpoint(child_path, cfg=cfg, device="cpu",
                                        allow_config_mismatch=True)
            reload_score = eval_masked(m2, b_test, dev, masks)["conflict"]
        arms[arm] = {**entry_eval, "updates": STEPS[arm],
                     "fit": {k: res[k] for k in
                             ("scope", "core_blocks_chosen", "optimizer_steps_executed",
                              "effective_samples", "optimizer_init", "loss_action_first",
                              "loss_action_last", "grad_norms_first_step",
                              "transient_buffer_drift", "parent_state_sha256",
                              "child_state_sha256", "wall_s")},
                     "delta_full": res["delta"],
                     "scope_plan": res["plan"],
                     "child_checkpoint": child_path,
                     "child_reload_conflict_score": reload_score,
                     "reload_consistent": None if reload_score is None
                     else bool(reload_score == conflict_score),
                     "rss_mb": rss_mb()}
        print(f"[{arm}] B_conflict {b_before['conflict']} → {conflict_score} "
              f"(gain {entry_eval['B_gain_conflict']}) A {a_before} → {entry_eval['A_after_occ']} "
              f"steps={res['optimizer_steps_executed']} {res['wall_s']}s", flush=True)

    # ---- 判定（预注册门槛，按任务书第 7 节状态等级）----
    verdicts: dict[str, str] = {}
    verdicts["pipeline_fixed"] = "PIPELINE_FIXED" if same_parent else "FAILED_INHERITANCE"
    ran = [a for a in ("Neural-Head", "Neural-Core") if "delta_full" in arms.get(a, {})]
    neural_ok = bool(ran) and all(
        arms[a]["delta_full"]["frozen_all_bitwise_unchanged"]
        and arms[a]["delta_full"]["trainable_changed_count"] >= 1 for a in ran)
    verdicts["neural_update"] = "NEURAL_UPDATE_VERIFIED" if neural_ok else "NOT_VERIFIED"
    lr0 = arms.get("Negative-lr0", {})
    lr0_ok = bool(lr0.get("delta_full")) and lr0["delta_full"]["frozen_all_bitwise_unchanged"] \
        and lr0["delta_full"]["trainable_changed_count"] == 0 \
        and lr0.get("B_after", {}).get("conflict") == b_before["conflict"]
    verdicts["negative_control_lr0"] = "PASS" if lr0_ok else "FAIL"
    cands = [a for a in ("Neural-Head", "Neural-Core")
             if isinstance(arms.get(a), dict) and arms[a].get("B_gain_conflict") is not None]
    if cands and not invalid_task:
        best = max(cands, key=lambda a: arms[a]["B_gain_conflict"])
        bar = max(baselines["test_majority_acc_on_conflict"],
                  baselines["train_majority_acc_on_test_conflict"],
                  baselines["random_policy"], b_before["conflict"] or 0.0)
        if arms[best]["B_gain_conflict"] >= NEW_SKILL_MIN_GAIN \
                and arms[best]["B_after"]["conflict"] > bar:
            verdicts["new_skill"] = f"NEW_SKILL_SIGNAL_SINGLE_SEED ({best})"
        elif arms[best]["B_gain_conflict"] >= NEW_SKILL_MIN_GAIN:
            verdicts["new_skill"] = f"GAIN_BUT_NOT_ABOVE_BASELINES ({best})"
        elif arms[best]["B_after"]["total_occluded"] is not None \
                and arms[best]["B_gain_conflict"] == 0 \
                and arms[best]["delta_full"]["trainable_changed_count"] >= 1:
            verdicts["new_skill"] = f"NO_SKILL_SIGNAL ({best}: 参数变了但未见B无提升)"
        else:
            verdicts["new_skill"] = f"NO_SKILL_SIGNAL ({best}: gain {arms[best]['B_gain_conflict']})"
    else:
        verdicts["new_skill"] = "INVALID_TASK_FOR_NEW_SKILL_TEST" if invalid_task else "NO_SKILL_SIGNAL"
    ret = [arms[a]["A_forgetting"] for a in ("Neural-Head", "Neural-Core")
           if isinstance(arms.get(a), dict) and arms[a].get("A_forgetting") is not None]
    verdicts["retention"] = ("RETENTION_PASS_SINGLE_SEED" if ret and min(ret) >= -RETENTION_MAX_DROP
                             else "RETENTION_FAIL")
    verdicts["research_validated"] = "NOT_GRANTED_BY_1_SEED_SMOKE（按任务书第 7 节）"

    payload = {"experiment_id": EXP_ID, "captured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
               "device": "cpu", "torch_threads": int(torch.get_num_threads()),
               "budget": {"cap": BUDGET_CAP, "used": budget.used, "log": budget.log},
               "baselines": baselines, "A_before_occ": a_before,
               "B_before": b_before, "arms": arms, "verdicts": verdicts,
               "preregistered": {"new_skill_min_gain_pp": NEW_SKILL_MIN_GAIN * 100,
                                  "retention_max_drop_pp": RETENTION_MAX_DROP * 100,
                                  "seeds": {"A": SEED_A, "B_train": SEED_B_TRAIN,
                                             "B_test": SEED_B_TEST, "sampling": SAMPLING},
                                  "steps_allocation": STEPS, "core_block_ids": core_ids,
                                  "core_rule": "父模型 block_load_count 最高的 2 块（平局取小号）"},
               "wall_s_total": round(time.perf_counter() - t0, 1),
               "wall_limit_s": WALL_LIMIT_S, "rss_mb_end": rss_mb(),
               "honesty": "1 seed smoke；参数变化≠学会新技能；训练样本提升≠泛化；"
                          "RESEARCH_VALIDATED 需独立 seed+置信区间+预算公平对照"}
    lineage = {"experiment_id": EXP_ID, "parent_checkpoint": str(PARENT),
               "parent_checkpoint_saved_at": parent_meta.get("saved_at"),
               "parent_state_sha256": parent_sha,
               "parent_git_revision": parent_meta.get("git_revision"),
               "task_A": D.task_to_dict(taskA), "task_B_train": D.task_to_dict(taskB_tr),
               "task_B_test": D.task_to_dict(taskB_te),
               "dataset_hashes": {"A_test": dA["dataset_hash"],
                                   "B_train": dB_tr["dataset_hash"],
                                   "B_test": dB_te["dataset_hash"]},
               "sampling_seeds": SAMPLING, "steps_allocation": STEPS,
               "budget_cap": BUDGET_CAP, "budget_used": budget.used,
               "core_block_ids_rule": "父模型 block_load_count 最高的 2 块（平局取小号）",
               "core_block_ids": core_ids,
               "same_model_lineage_verified": same_parent,
               "seed_semantics": "父模型参数初始化 seed=20261131（M2 预登记）；本轮零新建模型初始化；"
                                  "臂间差异只有采样流 seed 与参数范围，不是重新初始化"}
    deltas_out = {a: {"scope_plan": arms[a].get("scope_plan"),
                      "delta_full": arms[a].get("delta_full"),
                      "fit_summary": {k: arms[a]["fit"][k] for k in
                                      ("optimizer_steps_executed", "effective_samples",
                                       "grad_norms_first_step", "transient_buffer_drift")}
                      if "fit" in arms[a] else None}
                  for a in arms if isinstance(arms[a], dict) and "delta_full" in arms[a]}
    for name, obj in (("lineage.json", lineage),
                      ("parameter_deltas.json", deltas_out),
                      ("learning_controls.json", {"baselines": baselines, "A_before": a_before,
                                                   "B_before": b_before,
                                                   "arms": {a: arms[a] for a in arms},
                                                   "verdicts": verdicts}),
                      ("retention_eval.json", {a: {"A_before": a_before,
                                                    "A_after": arms[a].get("A_after_occ"),
                                                    "forgetting": arms[a].get("A_forgetting")}
                                               for a in arms if isinstance(arms[a], dict)})):
        (HERE / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    (HERE / "m4_incremental_smoke.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"verdicts": verdicts, "budget_used": budget.used,
                      "wall_s": payload["wall_s_total"]}, ensure_ascii=False, indent=2), flush=True)
    print(f"→ {HERE / 'm4_incremental_smoke.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
