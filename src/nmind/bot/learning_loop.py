"""候选神经学习循环：继承 stable 权重 → 有界答案掩码更新 → 封存自由生成评测 → 四臂对照。

口径（任务书 §4/§8/§9）：
  * 只接受 verification=verified 的经验；rejected/uncertain 一律不触发更新（内部再过滤）；
  * 候选从 stable checkpoint 逐位克隆并断言继承，stable 文件永不写回；
  * 更新走 `language.lm_update.masked_stream_update`（语言侧唯一正确口径；
    `bounded_stream_update` 是 NModel 网格行的实现，喂不进 token 序列）；
  * 评测前 `seal_eval_state` 复位训练态 buffer，漂移单独记录，不冒充参数学习；
  * 四臂分开计量：frozen（同权重不更新）/ memory（只查外部记忆，零更新）/
    neural（真实更新）/ negative（标签置换对照）；缺任何一臂都如实标 NOT_RUN；
  * 通过门槛只标 PROPOSED，晋升必须人工批准（evolution_manager）。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch

from ..learning.online_adaptation import (AdaptationBudget, clone_snapshot,
                                           seal_eval_state, state_hash,
                                           verify_bitwise_inheritance)
from ..language.decoder import generate
from ..language.lm_update import encode_rows, masked_stream_update, parameter_audit
from ..language.neural_dialogue import load_lm, save_lm
from ..language.tokenizer import CharTokenizer
from .experience_bridge import VerifiedExperience, to_learning_rows

NEW_SKILL_GAIN_THRESHOLD = 0.10          # 预登记阈值：不可为过门槛而调整
FORGETTING_LIMIT_PP = 3.0


def free_gen_em(model, tok: CharTokenizer, prompts: list[str], answers: list[str],
                *, max_new_tokens: int = 24) -> float | None:
    """真正自由生成（无 teacher forcing、不读标签）的精确匹配率。"""
    if not prompts:
        return None
    model.eval()
    hits = 0
    with torch.no_grad():
        for p, a in zip(prompts, answers):
            ids = tok.encode(p)
            _logits, state = model(torch.tensor([ids], dtype=torch.long))
            rec, _st = generate(model, tok, tok.encode("助手："), state=state,
                                max_new_tokens=max_new_tokens, greedy=True)
            hits += int(rec.text.strip() == a.strip())
    return round(hits / len(answers), 4)


def _norm(text: str) -> str:
    return "".join(ch for ch in str(text) if not ch.isspace())


def memory_retrieval(question: str,
                     index: tuple[list[str], list[str]] | None) -> str | None:
    """memory 臂的真口径：只允许在**学习流已经存下的** (题面 → 答案) 里精确检索，命不中就弃权。

    老实现 `_memory_only_answer()` 直接把评测集标签原样返回，EM 按构造恒为 1.0；
    那只能当"如果字典里恰好有考卷答案"的上界，不能当能力证据（任务书 §6.2 明确要求
    memory 臂要与 neural 臂可比，可比的前提是它拿不到 unseen 行的标签）。
    """
    if not index:
        return None
    q = _norm(question)
    for prompt, answer in zip(index[0], index[1]):
        if _norm(prompt) == q:
            return answer
    return None


def memory_arm_em(prompts: list[str], answers: list[str],
                  index: tuple[list[str], list[str]] | None) -> dict[str, Any]:
    """检索臂：命中/弃权分开计数，弃权算错（外部记忆不是无限好的兜底）。"""
    if not prompts:
        return {"sealed_em": None, "steps": 0, "note": "NOT_RUN：无评测行"}
    if not index:
        return {"sealed_em": None, "steps": 0,
                 "note": "NOT_RUN：未提供检索索引（memory 臂只能读学习流，不许读标签）"}
    hit = abstain = 0
    for p, a in zip(prompts, answers):
        got = memory_retrieval(p, index)
        if got is None:
            abstain += 1
        elif str(got).strip() == str(a).strip():
            hit += 1
    n = len(prompts)
    return {"sealed_em": round(hit / n, 4), "steps": 0, "hit": hit, "abstain": abstain,
            "n": n, "index_size": len(index[0]),
            "upper_bound_if_labels_were_memoized": 1.0,
            "note": "索引只含学习流（train）已存条目。未见测试集按构造题面不同，"
                    "所以命中率低/为 0 是**预期结果**——这一臂的存在就是为了证明"
                    "'背过的题'与'参数学到的能力'是两件事"}



def run_candidate_update(*, stable_checkpoint: str | Path, tokenizer_path: str | Path,
                         verified: list[VerifiedExperience], sealed_prompts: list[str],
                         sealed_answers: list[str], dev: torch.device,
                         updates: int = 48, lr: float = 3e-3, seed: int = 20261601,
                         replay_pairs: tuple[list[str], list[str]] | None = None,
                         retention_pairs: tuple[list[str], list[str]] | None = None,
                         memory_index: tuple[list[str], list[str]] | None = None,
                         replay_ratio: float = 0.15, trainable: str = "all",
                         adapter_prefixes: tuple[str, ...] | None = None,
                         arms: tuple[str, ...] = ("frozen", "neural"),
                         wall_limit_s: float = 60.0,
                         candidate_path: str | Path | None = None,
                         budget: AdaptationBudget | None = None,
                         max_new_tokens: int = 24) -> dict[str, Any]:
    """在硬预算内做一次候选更新并给出四臂对照报告。默认调用方只在获得授权后才进来。

    `updates` 与 `wall_limit_s` 都是**跨所有会训练的臂合计**的硬上限（不是每条臂各一份）：
    neural 与 negative 对照各拿一半步数、一半墙钟。预算用 AdaptationBudget 强制，超限时
    spend() 直接抛 BudgetExceeded，而不是悄悄少跑。"""
    tok = CharTokenizer.load(tokenizer_path)
    verified = [e for e in verified if e.verification_status == "verified"]
    if not verified:
        return {"executed": False, "status": "INCONCLUSIVE",
                 "reason": "无 verified 经验：rejected/uncertain 不得触发神经更新（任务书 §9.7）",
                 "arms": {a: "NOT_RUN" for a in arms}}
    prompts, answers, skipped = to_learning_rows(verified)
    if not prompts:
        # R1 的真实卡点在这里：verified 一堆，但没有一条是"拿来训神经"的。
        # 必须把原因吐出来，而不是静默少训或直接开训。
        return {"executed": False, "status": "NO_NEURAL_ELIGIBLE_EXPERIENCE",
                 "reason": f"{len(verified)} 条 verified 经验里，0 条 learning_target="
                            "neural_generation；被过滤明细=" + json.dumps(skipped, ensure_ascii=False),
                 "n_verified_in": len(verified), "skipped_reasons": skipped,
                 "arms": {a: "NOT_RUN" for a in arms}}
    rows = encode_rows(tok, prompts, answers)
    replay_rows = encode_rows(tok, *replay_pairs) if replay_pairs else None

    parent, parent_meta = load_lm(stable_checkpoint, tok)
    parent_sha = state_hash(parent)
    child, _ = load_lm(stable_checkpoint, tok)
    chk = verify_bitwise_inheritance(clone_snapshot(parent), child)
    if not chk["all_bitwise_equal"]:
        return {"executed": False, "status": "FAILED",
                 "reason": "候选逐位继承断言失败",
                 "detail": chk["mismatched_keys"][:5], "arms": {}}

    base = free_gen_em(child, tok, sealed_prompts, sealed_answers,
                        max_new_tokens=max_new_tokens)
    old_prompts = [p for p, _a in list(zip(prompts, answers))[:8]]
    old_answers = [a for _p, a in list(zip(prompts, answers))[:8]]
    base_old = free_gen_em(child, tok, old_prompts, old_answers, max_new_tokens=max_new_tokens)
    # 抗遗忘锚点必须是"父模型已经会做的事"（A 组封存集），不是训练集自己：
    # R1 用训练行前 8 条当"旧任务"，那测的是背没背住，测不出旧技能有没有被冲掉。
    ret_prompts, ret_answers = (retention_pairs if retention_pairs
                                 else (old_prompts, old_answers))
    base_ret = free_gen_em(child, tok, ret_prompts, ret_answers,
                            max_new_tokens=max_new_tokens)

    budget = budget or AdaptationBudget(updates, label="bot-candidate")
    arm_reports: dict[str, Any] = {"frozen": {"sealed_em": base, "steps": 0,
                                               "note": "同权重基线，零更新"}}
    if "memory" in arms:
        arm_reports["memory"] = memory_arm_em(sealed_prompts, sealed_answers, memory_index)
    else:
        arm_reports["memory"] = "NOT_RUN"

    before_state = {k: v.detach().clone() for k, v in child.state_dict().items()}
    # 共享预算是**跨臂硬总闸**，所以每条会训练的臂只能拿到其中的一份；
    # 否则 neural 臂会把预算吃干，negative（标签置换对照）实际 0 步，对照形同不存在。
    # 墙钟同理：policy.max_wall_s 是"单轮候选训练"的上限，必须跨臂分配，
    # 不能被两条臂各用一次而实际翻倍。
    train_arms = [a for a in ("neural", "negative") if a in arms]
    per_arm = max(1, int(updates) // max(1, len(train_arms))) if train_arms else 0
    arm_wall = (float(wall_limit_s) / len(train_arms)) if train_arms else float(wall_limit_s)
    if "neural" in arms:
        upd = masked_stream_update(child, rows, updates=per_arm, dev=dev, lr=lr, seed=seed,
                                    label="bot-neural", wall_limit_s=arm_wall,
                                    replay_rows=replay_rows,
                                    replay_ratio=(replay_ratio if replay_rows else 0.0),
                                    trainable=trainable, budget=budget,
                                    adapter_prefixes=adapter_prefixes)
    else:
        upd = {"optimizer_steps_executed": 0, "stopped_by": "arm_not_selected"}
    after_state = {k: v.detach().clone() for k, v in child.state_dict().items()}
    audit = seal_eval_state(child, {k: v.clone() for k, v in parent.state_dict().items()})
    sealed_after = free_gen_em(child, tok, sealed_prompts, sealed_answers,
                                max_new_tokens=max_new_tokens)
    # 遗忘指标必须与基线同口径（R2 §4 修复的完成，Gate-B 001 轮抓出）：训练后重测的
    # 必须是 retention_pairs（A 组封存集）本身。此前 old_after 仍测训练行前 8 条，
    # 那是"记忆化"不是"保持"——001 轮实测：记忆化 +50pp 掩盖了 A 组 -31.25pp 真遗忘，
    # 旧闸给出假阳性 PASS。训练行前 8 条保留为独立诊断列，不参与闸判定。
    ret_after = free_gen_em(child, tok, ret_prompts, ret_answers,
                             max_new_tokens=max_new_tokens)
    memorize_after = free_gen_em(child, tok, old_prompts, old_answers,
                                  max_new_tokens=max_new_tokens)
    # before/after 快照来自 state_dict()，里面既有可训练参数也有路由计数 buffer。
    # 必须拆开报：block_load_count 在一次前向里就能跳到几千，混在"参数变化"里
    # 会让"权重到底动了多少"这个取证问题失去意义（M4 虚假增益同款陷阱的取证版本）。
    p_names = {n for n, _ in child.named_parameters()}
    b_names = {n for n, _ in child.named_buffers()}
    param_change = parameter_audit({k: v for k, v in before_state.items() if k in p_names},
                                   {k: v for k, v in after_state.items() if k in p_names})
    buffer_change = parameter_audit({k: v for k, v in before_state.items() if k in b_names},
                                    {k: v for k, v in after_state.items() if k in b_names})
    arm_reports["neural"] = {"update": {k: upd[k] for k in
                                         ("optimizer_steps_executed", "loss_first", "loss_last",
                                          "wall_s", "stopped_by", "non_finite_steps_skipped",
                                          "trainable", "trainable_params")},
                              "steps_cap_this_arm": per_arm,
                              "total_steps_cap_all_arms": int(updates),
                              "sealed_em_after": sealed_after,
                              "old_task_em_after": ret_after}

    if "negative" in arms:
        neg_model, _ = load_lm(stable_checkpoint, tok)
        # 负对照臂与主臂共用同一个 AdaptationBudget ⇒ 授权预算是**跨臂硬总闸**，
        # 不会因为多跑一条对照臂就把步数翻倍（老实现各建各的预算）。
        neg = masked_stream_update(neg_model, rows, updates=per_arm, dev=dev, lr=lr, seed=seed,
                                    label="bot-negative", wall_limit_s=arm_wall,
                                    trainable=trainable, budget=budget,
                                    adapter_prefixes=adapter_prefixes,
                                    label_permutation_seed=seed + 1)
        neg_em = free_gen_em(neg_model, tok, sealed_prompts, sealed_answers,
                              max_new_tokens=max_new_tokens)
        arm_reports["negative"] = {"update": {k: neg[k] for k in
                                               ("optimizer_steps_executed", "stopped_by")},
                                    "steps_cap_this_arm": per_arm,
                                    "sealed_em_after": neg_em}
    else:
        arm_reports["negative"] = "NOT_RUN"

    report: dict[str, Any] = {
        "executed": True, "status": "RUNNING",
        "parent_sha256": parent_sha,
        "parent_checkpoint_meta": parent_meta,
        "child_sha256": state_hash(child),
        "inheritance_verified": True,
        "verified_experience_count": len(verified),
        "trainable_scope": trainable,
        "adapter_prefixes": (list(adapter_prefixes) if adapter_prefixes is not None
                              else ["head.", "block_to_ws."]),
        "arms": arm_reports,
        "sealed_baseline_em": base,
        "sealed_after_em": sealed_after,
        "new_skill_gain": (round(sealed_after - base, 4)
                            if base is not None and sealed_after is not None else None),
        "old_task_baseline_em": base_ret,
        "old_task_after_em": ret_after,
        "old_task_forgetting_pp": (round((ret_after - base_ret) * 100, 2)
                                    if base_ret is not None and ret_after is not None else None),
        "train_rows_memorization": {"baseline_em": base_old, "after_em": memorize_after,
                                     "note": "训练行前8条记忆化诊断：上升只证明背题，"
                                             "不参与遗忘闸判定（Gate-B 001 轮教训）"},
        "parameter_change": param_change,
        "buffer_change_during_training": buffer_change,
        "audit_scope_note": "parameter_change 只含 named_parameters()；路由计数 buffer 的单列在 "
                             "buffer_change_during_training，评测前由 transient_buffer_drift 复位",
        "transient_buffer_drift": audit,
        "budget_used_steps": budget.used,
        "budget_max_steps": updates,
        "budget_plan": {"authorized_total_steps": int(updates),
                        "training_arms": list(train_arms),
                        "per_arm_step_cap": per_arm,
                        "wall_cap_total_s": round(float(wall_limit_s), 2),
                        "wall_cap_per_arm_s": round(arm_wall, 2),
                        "steps_used_total": int(budget.used),
                        "budget_log": budget.log},
        "thresholds": {"new_skill_gain_min": NEW_SKILL_GAIN_THRESHOLD,
                        "old_task_forgetting_max_pp": FORGETTING_LIMIT_PP},
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    gain = report["new_skill_gain"]
    forget = report["old_task_forgetting_pp"]
    gates = {"new_skill_gain": gain is not None and gain >= NEW_SKILL_GAIN_THRESHOLD,
              "old_task_retention": forget is None or forget >= -FORGETTING_LIMIT_PP,
              "weight_changed": param_change["tensors_changed"] > 0,
              "inheritance": True}
    report["gates"] = gates
    report["status"] = "PROPOSED" if all(gates.values()) else "REJECTED"
    if candidate_path is not None:
        save_lm(candidate_path, model=child, tokenizer=tok, cfg=child.cfg,
                steps=int(upd.get("optimizer_steps_executed") or 0), seed=seed,
                parent_sha=parent_sha, manifest_hash="bot-practice-experiences")
        report["candidate_path"] = str(candidate_path)
    return report


def load_verified_from_jsonl(path: str | Path) -> list[VerifiedExperience]:
    """从审计 JSONL 复原 verified 经验（重启后继续学习用，绝不读 sealed）。"""
    out: list[VerifiedExperience] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("verification_status") != "verified":
            continue
        out.append(VerifiedExperience(**rec))
    return out
