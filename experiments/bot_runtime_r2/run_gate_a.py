"""Gate-A 驱动脚本（任务书 §3 / P1 产物）：只测量、零 optimizer.step。

真正的测量逻辑在 `nmind.bot.task_capability_bridge.run_gate_a()` 里（复用运行时同一套
TaskSpec/attempt/judge 口径，避免"实验脚本一套口径、Bot 另一套口径"）；本脚本只做四件事：

  1. 用 stable checkpoint 构造 binding（A/B/C/X 绑定族）+ tools（R1 三个工具练习族）两组 TaskSpec；
  2. 调 run_gate_a 得到逐族 EM / CI / 词表 OOV / oracle 独立性 / 反事实矩阵；
  3. 叠上 P2 选择器的**资格硬门槛**（缺口阈值 + 学习流与未见测试集输入指纹是否互斥），
     把"有缺口"收敛成"有可验证的缺口"；
  4. 落三个 Gate-A 产物 + 打印结论。

三个产物（任务书 A.4）：
  results/bot_r2/baseline_capability_matrix.json   —— 能力矩阵（Bot 空闲循环只读这个）
  results/bot_r2/task_alignment_audit.json         —— 任务适配审计（逐族拒绝原因/泄露/反事实）
  results/bot_r2/skill_registry.json               —— 技能注册表 + 选择器权重（阈值快照）
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


from nmind.bot import task_capability_bridge as bridge  # noqa: E402
from nmind.bot.curriculum import (CurriculumConfig, SkillSelector,  # noqa: E402
                                  dataset_stats)
from nmind.bot.skill_registry import default_registry  # noqa: E402
from nmind.language import binding_data as bd  # noqa: E402
from nmind.language.neural_dialogue import load_lm  # noqa: E402
from nmind.language.tokenizer import CharTokenizer  # noqa: E402
from nmind.learning.online_adaptation import state_hash  # noqa: E402

OUT = ROOT / "experiments" / "bot_runtime_r2"          # 任务书 §10 交付目录
MIRROR = ROOT / "results" / "bot_r2"                   # Bot 空闲循环的默认只读路径
STABLE = ROOT / "experiments" / "m5_language" / "m52_s0_checkpoint.pt"
TOK = ROOT / "experiments" / "m5_language" / "tokenizer_m52.json"
# M5.2 s0 的 state_hash ——血统锚点。出处：experiments/bot_runtime_r1/lineage_and_buffer_audit.json
# :state_hash_parent 与 results/bot_r1_e2e/status.json:stable_model_sha256（两处一致）。
EXPECTED_STATE_SHA = "6404f453b0aa722a660f16d534ffe7543e2f015e25c7e6a2bb9fafc042fdd529"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    tok = CharTokenizer.load(str(TOK))
    model, lm_meta = load_lm(str(STABLE), tok)   # 本脚本一次 backward 都不做（Gate-A 零训练）
    stable_sha = state_hash(model)
    file_sha = __import__("hashlib").sha256(STABLE.read_bytes()).hexdigest()
    print(f"stable file_sha256={file_sha}\nstate_hash      ={stable_sha}\n"
          f"lineage_expected={EXPECTED_STATE_SHA}\nlineage_ok={stable_sha == EXPECTED_STATE_SHA}")
    if stable_sha != EXPECTED_STATE_SHA:
        print("!! 权重血统对不上，Gate-A 不许开工（先修 checkpoint 出处）")
        return 2

    reg = default_registry()
    binding = bridge.binding_specs(
        tok, splits=("train", "sealed", "unseen_combo", "counterfactual"), registry=reg)
    tools = bridge.practice_specs(tok, n=24, seed=20261699, registry=reg)
    print(f"TaskSpec：binding={len(binding)} tools={len(tools)} "
          f"（train 行仅进矩阵，判定用 SPLIT_PRIORITY={bridge.SPLIT_PRIORITY}）")

    gate = bridge.run_gate_a(model, tok, binding=binding, tools=tools,
                             stable_sha=stable_sha, stable_file_sha=file_sha)

    # ---- P2 资格硬门槛叠加：有缺口 ≠ 有可验证的缺口 ----
    per_item = round(float(gate["wall_s"]) / max(1, int(gate["n_measured"])), 5)
    ds = dataset_stats(tok, gate, per_item_s=per_item or 0.06)
    selector = SkillSelector(CurriculumConfig(),
                             allowed_skill_ids=tuple(
                                 s.skill_id for s in reg.trainable()) if reg.trainable() else ())
    assessments = selector.assess(gate_a=gate, registry=reg, dataset_stats=ds,
                                  recent_families=())
    bridge_eligible = [a for a in assessments
                       if a.gate_a_verdict == "ELIGIBLE_NEW_SKILL"]
    passed = [a.skill_id for a in assessments if a.eligible]
    blocked = [{"skill_id": a.skill_id, "family": a.family, "tier": a.tier,
                "gate_a_verdict": a.gate_a_verdict, "neural_em": a.neural_em,
                "gap_pp": a.gap_pp, "score": a.score, "rejections": a.rejections}
               for a in bridge_eligible if not a.eligible]
    gate["gate_a_eligibility"] = {
        "note": ("bridge 的 ELIGIBLE_NEW_SKILL 只看缺口与 oracle；P2 选择器再查"
                 "“学习流与未见测试集在输入指纹层面是否互斥”。两者都过才算可进 Gate-B。"),
        "bridge_eligible_families": gate["eligible_families"],
        "bridge_eligible_skills": [a.skill_id for a in bridge_eligible],
        "eligible_gap_set": passed,
        "blocked_by_selector": blocked,
        "all_assessments": [a.to_dict() for a in assessments],
        "dataset_stats": ds, "per_item_s": per_item,
        "thresholds_preregistered": {"min_gap_pp": selector.cfg.min_gap_pp,
                                      "min_dataset_rows": selector.cfg.min_dataset_rows,
                                      "exploration_fraction": selector.cfg.exploration_fraction,
                                      "max_attempts_per_skill":
                                      selector.cfg.max_attempts_per_skill}}
    if gate["verdict"] == "TASK_CAPABILITY_ALIGNED" and not passed:
        gate["verdict"] = "NO_ELIGIBLE_LEARNING_TASK"
        gate["summary"] = ("A 组已复现，但选择器把一个可训技能都没放过（详见 "
                           "gate_a_eligibility.blocked_by_selector）→ 本轮不进 Gate-B，"
                           "Bot 走 NO_ELIGIBLE_TASK 休眠路径")
        print("!! 选择器把所有 ELIGIBLE_NEW_SKILL 都拦下了，Gate-A 结论降级为 "
              "NO_ELIGIBLE_LEARNING_TASK")

    # ---- 三个产物 ----
    gate["lm_manifest"] = lm_meta
    (OUT / "baseline_capability_matrix.json").write_text(
        json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")

    audit = {
        "gate": "GATE_A", "no_training": True, "verdict": gate["verdict"],
        "summary": gate["summary"],
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "generated_by": "experiments/bot_runtime_r2/run_gate_a.py",
        "stable_sha256_state": stable_sha, "stable_file_sha256": file_sha,
        "expected_state_sha256": EXPECTED_STATE_SHA,
        "lineage_verified": stable_sha == EXPECTED_STATE_SHA,
        "thresholds": gate["thresholds"],
        "n_specs": gate["n_specs"], "n_measured": gate["n_measured"],
        "wall_s": gate["wall_s"],
        "families": gate["families"], "matrix": gate["matrix"],
        "leakage_and_vocab": gate["leakage_and_vocab"],
        "tool_ledger": gate["tool_ledger"],
        "counterfactual": gate["counterfactual"],
        "baselines": gate["baselines"],
        "gate_a_eligibility": gate["gate_a_eligibility"],
        "conclusion_points": [
            "A 组（位置问答）复现 = 神经基线可用，Gate-B 的抗遗忘锚点成立",
            "R1 三个练习族按 TOOL 记账：题面 OOV 字符率高，不是“练得不够”，是“喂错了”",
            "B 组里只有独立 oracle + 学习流/测试集输入指纹互斥的族才允许进 Gate-B",
            "C 组（两实体绑定/组合泛化）仅诊断，未经单独许可不训练",
        ]}
    (OUT / "task_alignment_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    reg_doc = {"gate": "GATE_A", "generated_at": audit["generated_at"],
               "stable_sha256_state": stable_sha,
               **reg.to_json(),
               "trainable_skill_ids": [s.skill_id for s in reg.trainable()],
               "selector_config": selector.cfg.to_dict(),
               "capability_states_observed": {a.skill_id: a.gate_a_verdict
                                               for a in assessments}}
    if reg.rejected:
        print("!! 注册表有被拒绝的 SkillSpec（契约不合法）:", reg.rejected)
    (OUT / "skill_registry.json").write_text(
        json.dumps(reg_doc, ensure_ascii=False, indent=2), encoding="utf-8")

    # 交付目录与 Bot 只读目录各留一份，并断言字节一致（防止"报告引 A 文件、Bot 读 B 文件"）
    import hashlib
    import shutil
    MIRROR.mkdir(parents=True, exist_ok=True)
    for fn in ("baseline_capability_matrix.json", "task_alignment_audit.json",
               "skill_registry.json"):
        shutil.copyfile(OUT / fn, MIRROR / fn)
        a = hashlib.sha256((OUT / fn).read_bytes()).hexdigest()
        b = hashlib.sha256((MIRROR / fn).read_bytes()).hexdigest()
        if a != b:
            raise SystemExit(f"镜像不一致：{fn}")

    print("\n=== GATE-A verdict ===")
    print(gate["verdict"], "|", gate["summary"])
    print(f"A 组复现 EM={gate['a_sealed_em']}  eligible_families={gate['eligible_families']}")
    print("\n--- 逐族（判定 EM 用 sealed/unseen_combo/counterfactual；train 列只看背没背过） ---")
    for a in assessments:
        cell = gate["matrix"].get(a.family, {}).get("splits", {})
        tr = next((c for s, c in cell.items() if s == "train"), {})
        print(f"{a.skill_id:<34} {a.tier:<4} em={a.neural_em} "
              f"ci={cell.get('sealed', {}).get('ci95')} train_em={tr.get('em')} "
              f"gap={a.gap_pp:>6} score={a.score:>6} "
              f"{'ELIGIBLE' if a.eligible else 'REJECT: ' + ' | '.join(a.rejections)[:110]}")
    print("\nblocked_by_selector:", json.dumps(blocked, ensure_ascii=False)[:600])
    print("\nartifacts:", [p.name for p in sorted(OUT.glob("*.json"))])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
