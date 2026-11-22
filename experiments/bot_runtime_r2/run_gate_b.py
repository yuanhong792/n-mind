"""Gate-B 执行脚本（gateb-unseen-entity-001，任务书 §6）。

预登记：experiments/bot_runtime_r2/gateb_pre_registration.json（训练前落盘，status=PENDING_APPROVAL）。
运行前提：用户明确批准后手动运行 `python experiments/bot_runtime_r2/run_gate_b.py`；
本脚本不自动解锁、不自动重试，REJECTED 后不追加预算（预登记 no_retry 条款）。

流程（全部走 runtime 既定口径，不另造实现）：
  1. 校验 stable checkpoint state_hash == 预登记值（不符直接退出，不训练）；
  2. 数据按预登记 seed 构造：学习流 48（位置1/2）、封存 40（位置3/4）、
     A 组锚点 80（M5.2 s0 sealed 同源）、A 组回放 48；
  3. 学习行包装成 VerifiedExperience（learning_target=neural_generation，
     经验来源如实标注为预登记数据集而非自主练习产物）；
  4. run_candidate_update 四臂对照（frozen/memory/neural/negative，预算跨臂硬总闸 96 步）；
  5. 候选按判定落位 proposed/ 或 rejected/，再做 per-position 分解评测；
  6. stable 文件 sha 前后核验 + 结果双写（experiments/ 留档 + results/ 运行时镜像）。

判定口径（预登记 6 闸，代码常量 NEW_SKILL_GAIN_THRESHOLD=0.10 / FORGETTING_LIMIT_PP=3.0）：
  全过 → PROPOSED（等 --i-approve 晋升）；任一不过 → REJECTED（隔离，不重跑）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from nmind.bot.experience_bridge import VerifiedExperience  # noqa: E402
from nmind.bot.learning_loop import (free_gen_em, run_candidate_update)  # noqa: E402
from nmind.language import binding_data as bd  # noqa: E402
from nmind.language.neural_dialogue import load_lm  # noqa: E402
from nmind.language.tokenizer import CharTokenizer  # noqa: E402
from nmind.learning.online_adaptation import state_hash  # noqa: E402

STABLE = ROOT / "experiments" / "m5_language" / "m52_s0_checkpoint.pt"
TOK = ROOT / "experiments" / "m5_language" / "tokenizer_m52.json"
PREREG = ROOT / "experiments" / "bot_runtime_r2" / "gateb_pre_registration.json"
OUT = ROOT / "experiments" / "bot_runtime_r2"
MIRROR = ROOT / "results" / "bot_r2"
EXPECTED_STATE_SHA = "6404f453b0aa722a660f16d534ffe7543e2f015e25c7e6a2bb9fafc042fdd529"


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(tasks):
    return [t.prompt for t in tasks], [t.target for t in tasks]


def _pos_of(task) -> str:
    for k, v in task.slots:
        if "位置" in v:
            return v
    return "?"


def main() -> int:
    ap = argparse.ArgumentParser(description="Gate-B 受控候选训练（预登记驱动，防误跑）")
    ap.add_argument("--run", default="", help="轮号（如 001/002），必填；不指定不启动任何训练")
    args = ap.parse_args()
    if not args.run:
        print("[GATE-B] 用法：python run_gate_b.py --run <轮号>（如 --run 002）。")
        print("[GATE-B] 不指定轮号不启动任何训练（001 预登记为 APPROVED 历史口径，防误重放）。")
        return 2
    pre_path = PREREG if args.run == "001" else OUT / f"gateb_pre_registration_{args.run}.json"
    if not pre_path.exists():
        print(f"[GATE-B] 预登记不存在：{pre_path}")
        return 3
    pre = json.loads(pre_path.read_text(encoding="utf-8"))
    if pre["status"] != "APPROVED":
        print(f"[GATE-B] 预登记状态={pre['status']}，尚未获用户批准，不启动训练。")
        print(f"[GATE-B] 批准方式：把 {pre_path.name} 的 status 改为 APPROVED 后重跑本脚本。")
        return 2
    run_id = args.run
    result_file = OUT / f"gateb_result_{run_id}.json"
    if result_file.exists():
        print(f"[GATE-B] {result_file.name} 已存在——该轮已完成，禁止重跑覆盖既有证据。")
        return 4
    t0 = time.time()
    torch.set_num_threads(8)
    dev = torch.device("cpu")

    # 1. 父模型指纹校验：不符预登记值说明 stable 已被别处改动，直接退出
    tok = CharTokenizer.load(str(TOK))
    model, meta = load_lm(str(STABLE), tok)
    parent_sha = state_hash(model)
    if parent_sha != EXPECTED_STATE_SHA:
        print(f"[GATE-B] stable state_hash 不符：{parent_sha} != {EXPECTED_STATE_SHA}，退出。")
        return 3
    stable_sha_before = _file_sha(STABLE)

    # 2. 数据按预登记 seed 构造（build_family 的 split→(n, seed) 表已在模块内锁定）
    train = bd.build_family("unseen_entity_qa", "train")
    sealed = bd.build_family("unseen_entity_qa", "sealed")
    anchor = bd.build_family("position_qa", "sealed")          # A 组封存 80 条
    # position_qa 在 SPLIT_TABLE 中没有预登记 train 行（父模型训练行走 M5.2 直调口径），
    # 这里直调 builder：n=48 / seed=20261401 与预登记一致，分布与父模型 s0 训练同源
    replay = bd.build_position_qa(48, bd.M52_SEED, "train")    # A 组回放 48 条
    assert len(train) == 48 and len(sealed) == 40 and len(anchor) == 80, \
        f"数据规模与预登记不符：{len(train)}/{len(sealed)}/{len(anchor)}"
    train_p, train_a = _rows(train)
    sealed_p, sealed_a = _rows(sealed)
    anchor_p, anchor_a = _rows(anchor)
    replay_p, replay_a = _rows(replay)

    # 3. 预登记学习行 → verified 经验（如实标注数据集来源；oracle 独立判分）
    model_sha_short = parent_sha[:16]
    verified = [
        VerifiedExperience(
            experience_id=f"{run_id}-{t.fingerprint}",
            task_family=t.family,
            observation={"question": t.question, "source": "preregistered_dataset"},
            action={"reference": "oracle_target_only"},
            feedback={"oracle": t.target, "feedback_source": "deterministic_oracle"},
            verification_status="verified",
            validator_id=bd.ORACLE_ID,
            source_id=f"preregistered:{t.family}:{t.split}",
            content_sha256=t.fingerprint,
            model_checkpoint_sha256=model_sha_short,
            skill_id="skill.binding.unseen_entity",
            response_source="rule",
            feedback_source="deterministic_oracle",
            learning_target="neural_generation",
        ) for t in train
    ]

    # 4. 四臂候选更新（预算全部读自预登记：002 起 trainable/lr/steps/seed 均由预登记锁定，
    #    脚本本身冻结——预登记差异即实验差异，001(all)/002(adapter) 走同一代码路径）
    b = pre["budget"]
    # 003 起预登记可声明 adapter_prefixes（进一步收窄 adapter 范围，如冻结输出头只训
    # block_to_ws）；未声明沿用 set_trainable 默认（head.+block_to_ws.），001/002 口径不变
    apx = b.get("adapter_prefixes")
    cand_tmp = OUT / "candidates" / f"_pending_{run_id}.pt"
    cand_tmp.parent.mkdir(parents=True, exist_ok=True)
    report = run_candidate_update(
        stable_checkpoint=str(STABLE), tokenizer_path=str(TOK),
        verified=verified, sealed_prompts=sealed_p, sealed_answers=sealed_a,
        dev=dev, updates=int(b["total_gradient_steps"]), lr=float(b["lr"]),
        seed=int(b["init_seed"]),
        replay_pairs=(replay_p, replay_a), retention_pairs=(anchor_p, anchor_a),
        memory_index=(train_p, train_a), replay_ratio=float(b["replay_ratio"]),
        trainable=str(b["trainable"]),
        adapter_prefixes=(tuple(apx) if apx else None),
        arms=("frozen", "memory", "neural", "negative"),
        wall_limit_s=float(b["wall_limit_s_total"]), candidate_path=str(cand_tmp),
        max_new_tokens=int(b["max_new_tokens"]))

    # 5. 候选按判定落位（PROPOSED→proposed/，REJECTED→rejected/），再做 per-position 分解
    verdict = report.get("status", "UNKNOWN")
    final_dir = OUT / "candidates" / ("proposed" if verdict == "PROPOSED" else "rejected")
    final_dir.mkdir(parents=True, exist_ok=True)
    final_path = final_dir / f"{run_id}.pt"
    if cand_tmp.exists():
        shutil.move(str(cand_tmp), str(final_path))
        report["candidate_path"] = str(final_path)

    per_position: dict[str, Any] = {}
    if final_path.exists():
        cand_model, _ = load_lm(str(final_path), tok)
        for pos in ("3号位置", "4号位置"):
            idx = [i for i, t in enumerate(sealed) if _pos_of(t) == pos]
            per_position[pos] = {
                "n": len(idx),
                "em": free_gen_em(cand_model, tok,
                                  [sealed_p[i] for i in idx],
                                  [sealed_a[i] for i in idx]),
            }
        base_model, _ = load_lm(str(STABLE), tok)
        per_position["_baseline"] = {
            pos: free_gen_em(base_model, tok,
                             [sealed_p[i] for i in idx2],
                             [sealed_a[i] for i in idx2])
            for pos in ("3号位置", "4号位置")
            for idx2 in [[i for i, t in enumerate(sealed) if _pos_of(t) == pos]]
        }

    # 6. stable 完整性核验 + 结果落盘（experiments/ 留档 + results/ 镜像）
    stable_sha_after = _file_sha(STABLE)
    result: dict = {
        "experiment": pre["experiment"],
        "preregistration": str(PREREG),
        "verdict": verdict,
        "gates": report.get("gates"),
        "report": report,
        "per_position_sealed": per_position,
        "stable_file_sha256_before": stable_sha_before,
        "stable_file_sha256_after": stable_sha_after,
        "stable_untouched": stable_sha_before == stable_sha_after,
        "parent_state_sha256": parent_sha,
        "data_fingerprints": {
            "train_sha16": hashlib.sha256("".join(train_p).encode("utf-8")).hexdigest()[:16],
            "sealed_sha16": hashlib.sha256("".join(sealed_p).encode("utf-8")).hexdigest()[:16],
        },
        "wall_s": round(time.time() - t0, 2),
        "device": "cpu", "torch_threads": 8,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    OUT.joinpath(f"gateb_result_{run_id}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    MIRROR.mkdir(parents=True, exist_ok=True)
    (MIRROR / f"gateb_result_{run_id}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    # 7. 判定打印（诚实口径：过/不过都如实写，不粉饰）
    gates = result["gates"] or {}
    print(f"[GATE-B] 判定 = {verdict}")
    print(f"[GATE-B] 基线 sealed EM = {report.get('sealed_baseline_em')}  "
          f"训练后 = {report.get('sealed_after_em')}  gain = {report.get('new_skill_gain')}")
    print(f"[GATE-B] A组保持 old_task_forgetting_pp = {report.get('old_task_forgetting_pp')}")
    for name, ok in gates.items():
        print(f"[GATE-B]   {name}: {'PASS' if ok else 'FAIL'}")
    print(f"[GATE-B] per-position: {json.dumps(per_position, ensure_ascii=False)}")
    print(f"[GATE-B] stable 未被改动 = {result['stable_untouched']}  wall = {result['wall_s']}s")
    print(f"[GATE-B] 结果已落盘 experiments/bot_runtime_r2/gateb_result_{run_id}.json")
    return 0 if verdict == "PROPOSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
