"""Gate-B 004 机制诊断：无训练探针 + 受控因果干预（第三/四阶段，零 optimizer.step）。

对 stable 与 001/002/003 三个 REJECTED 候选做：
  第三阶段（无训练分析）：
    * trace_forward 抽样：路由分布 / 槽占用 / 读取注意力熵 / head 输入范数 / EOS 偏置
    * 答案 token 排序统计：B 组 sealed 全量，正确答案首字符的 rank/prob 分布
    * 线性探针：位置 4 类 / 实体 2 类（head 输入与 Workspace 内容两个特征源，
      独立 train/test 划分）——只作表示诊断，不计入自主能力成绩
    * 反事实表示变化：同上下文不同查询实体的 head 输入余弦
    * Workspace 内容漂移：同 prompt 下 stable vs 候选的逐槽余弦
  第四阶段（受控因果干预，不改任何原始 checkpoint）：
    * 参数替换矩阵：stable 骨架 + {002.head, 002.block_to_ws, 002.both, 003.block_to_ws}；
      候选骨架 + stable 对应模块（反向）
    * 激活替换：同 prompt 下把候选末态的 Workspace 内容 / 全部块状态 (h) 换成
      stable 的（或反向），从该状态继续自由生成
    * 记录 A 正确率 / B 正确率 / 反事实跟随率 / 输出分布变化（答案首字符 logit）
  方法论红线：模块组合存在非线性相互作用，单次替换的效果不解释为唯一因果来源；
  所有替换仅在内存副本上进行，磁盘 checkpoint 逐位不碰。

产出：experiments/bot_runtime_r2/GATEB004_DIAGNOSTICS.json（+ results/bot_r2/ 镜像）。
运行：python experiments/bot_runtime_r2/diagnose_gateb004.py
"""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from nmind.bot.learning_loop import free_gen_em  # noqa: E402
from nmind.language import binding_data as bd  # noqa: E402
from nmind.language.decoder import generate  # noqa: E402
from nmind.language.neural_dialogue import LMState, load_lm  # noqa: E402
from nmind.language.state_probe import (counterfactual_representation_shift,  # noqa: E402
                                         extract_head_input, extract_ws_content,
                                         linear_probe, trace_forward)
from nmind.language.tokenizer import CharTokenizer  # noqa: E402
from nmind.language.workspace_diagnostics import ws_content_drift  # noqa: E402

STABLE = ROOT / "experiments" / "m5_language" / "m52_s0_checkpoint.pt"
TOK = ROOT / "experiments" / "m5_language" / "tokenizer_m52.json"
CANDS = {"001": ROOT / "experiments/bot_runtime_r2/candidates/rejected/gateb001.pt",
          "002": ROOT / "experiments/bot_runtime_r2/candidates/rejected/002.pt",
          "003": ROOT / "experiments/bot_runtime_r2/candidates/rejected/003.pt"}
OUT = ROOT / "experiments" / "bot_runtime_r2"
MIRROR = ROOT / "results" / "bot_r2"
EXPECTED_STATE_SHA = "6404f453b0aa722a660f16d534ffe7543e2f015e25c7e6a2bb9fafc042fdd529"
N_A_EVAL = 40          # A 组（位置问答封存）评测条数采样
N_TRACE = 6            # trace_forward 抽样条数（全量 JSON 过大）


def _rows(tasks):
    return [t.prompt for t in tasks], [t.target for t in tasks]


def _norm(s: str) -> str:
    return "".join(ch for ch in s if not ch.isspace())


def _em(model, tok, prompts, answers):
    return free_gen_em(model, tok, prompts, answers, max_new_tokens=24)


def _cf_follow(model, tok, items) -> float:
    """反事实跟随率：items = [{"fingerprint","prompt","target","group"}]（v1 口径）。"""
    from nmind.language.decoder import free_generation
    preds: dict[str, str] = {}
    model.eval()
    with torch.no_grad():
        for it in items:
            rec = free_generation(model, tok, it["prompt"], max_new_tokens=24)
            preds[it["fingerprint"]] = rec.text.strip()
    groups: dict[str, list[bool]] = {}
    for it in items:
        groups.setdefault(it["group"], []).append(
            _norm(preds.get(it["fingerprint"], "")) == _norm(it["target"]))
    return round(sum(all(v) for v in groups.values()) / len(groups), 4) if groups else None


def _answer_rank_stats(model, tok, prompts, targets) -> dict[str, Any]:
    """B 组 sealed 全量：正确答案首字符的 rank/prob 统计（v1 口径生成条件前向）。"""
    ranks: list[int] = []
    probs: list[float] = []
    eos_gaps: list[float] = []
    model.eval()
    with torch.no_grad():
        for p, t in zip(prompts, targets):
            rec = trace_forward(model, tok, p, target=t)
            info = rec.tokens_trace[-1].target_token_info
            if info and info.get("rank") is not None:
                ranks.append(info["rank"])
                probs.append(info["prob"])
                eos_gaps.append(info["eos_minus_top"])
    return {"n": len(ranks),
             "rank_mean": round(sum(ranks) / len(ranks), 2) if ranks else None,
             "rank_top1_rate": round(sum(1 for r in ranks if r == 1) / len(ranks), 4) if ranks else None,
             "prob_mean": round(sum(probs) / len(probs), 6) if probs else None,
             "eos_minus_top_mean": round(sum(eos_gaps) / len(eos_gaps), 4) if eos_gaps else None}


def _swap_modules(base_model, donor_model, modules: list[str]):
    """内存副本上的模块参数替换（磁盘 checkpoint 不碰）。modules ⊆ {"head","block_to_ws"}。"""
    m = copy.deepcopy(base_model)
    donor_sd = donor_model.state_dict()
    keep = {k: v for k, v in donor_sd.items() if any(k.startswith(mod + ".") for mod in modules)}
    missing, unexpected = m.load_state_dict(keep, strict=False)
    return m, sorted(keep.keys())


def _route_histogram(model, tok, prompts) -> dict[str, Any]:
    """路由块选择频率（B 组抽样 trace）。"""
    counts = [0] * model.cfg.n_blocks
    for p in prompts[:N_TRACE]:
        tr = trace_forward(model, tok, p)
        for tt in tr.tokens_trace:
            for i in tt.route_indices:
                counts[i] += 1
    total = sum(counts) or 1
    return {"counts": counts, "freq": [round(c / total, 4) for c in counts]}


def main() -> int:
    t0 = time.time()
    torch.set_num_threads(8)
    tok = CharTokenizer.load(str(TOK))
    models: dict[str, Any] = {"stable": load_lm(str(STABLE), tok)[0]}
    for tag, path in CANDS.items():
        models[tag] = load_lm(str(path), tok)[0]
    # stable 指纹校验 + 候选 parent 指纹
    diag: dict[str, Any] = {"experiment": "gateb004-diagnostics", "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                             "zero_training": True, "protocols": {"freegen": "freegen-v1（与 001-003 历史口径一致）"},
                             "models": {}, "probes": {}, "interventions": {}, "wall_s": None}

    a_tasks = bd.build_family("position_qa", "sealed")[:N_A_EVAL]
    a_p, a_a = _rows(a_tasks)
    b_tasks = bd.build_family("unseen_entity_qa", "sealed")
    b_p, b_a = _rows(b_tasks)
    anchor_all = bd.build_family("position_qa", "sealed")
    ap_all, aa_all = _rows(anchor_all)
    b_train = bd.build_family("unseen_entity_qa", "train")
    a_train = bd.build_position_qa(48, 20261401, "train")
    cf_items = []
    for i, t in enumerate(bd.build_family("binding_swap", "counterfactual")[:24]):
        cf_items.append({"fingerprint": t.fingerprint, "prompt": t.prompt,
                          "target": t.target, "group": f"grp{i // 2}"})

    # ---------- 第三阶段：无训练分析 ----------
    for tag, m in models.items():
        d: dict[str, Any] = {}
        d["a_retention_em"] = _em(m, tok, ap_all, aa_all)          # A 组封存 80 条
        d["b_sealed_em"] = _em(m, tok, b_p, b_a)                    # B 组封存 40 条
        d["cf_follow_rate"] = _cf_follow(m, tok, cf_items)          # 反事实跟随率
        d["answer_rank_stats_B"] = _answer_rank_stats(m, tok, b_p, b_a)
        d["route_histogram_B"] = _route_histogram(m, tok, b_p)
        tr_sample = trace_forward(m, tok, b_p[0], target=b_a[0])
        d["trace_sample"] = {
            "prompt": b_p[0],
            "logits_match_original": tr_sample.logits_match_original,
            "final_occupied": tr_sample.tokens_trace[-1].ws_occupied,
            "final_ws_content_norm": tr_sample.final_ws_content_norm,
            "final_block_states_norm": tr_sample.final_block_states_norm,
            "read_attn_entropy_last": tr_sample.tokens_trace[-1].read_attn_entropy,
            "eos_logit_last": tr_sample.tokens_trace[-1].eos_logit,
        }
        # 线性探针：位置 4 类（A 组训练分布，两个特征源）
        pos_labels = []
        for t in a_train:
            for k, v in t.slots:
                if "位置" in v:
                    pos_labels.append(int(v[0]) - 1)
        hi_pos = [extract_head_input(m, tok, t.prompt) for t in a_train]
        wc_pos = [extract_ws_content(m, tok, t.prompt) for t in a_train]
        d["probe_position_head_input"] = linear_probe(hi_pos, pos_labels, n_classes=4)
        d["probe_position_ws_content"] = linear_probe(wc_pos, pos_labels, n_classes=4)
        # 线性探针：实体 2 类（同位置 1/2 号，小明 vs 小红）
        ent_feats, ent_labels, ent_pos = [], [], []
        for t in a_train:
            for k, v in t.slots:
                if "位置" in v and v in ("1号位置", "2号位置"):
                    ent_feats.append(extract_head_input(m, tok, t.prompt))
                    ent_labels.append(0)
                    ent_pos.append(v)
        for t in b_train:
            for k, v in t.slots:
                if "位置" in v and v in ("1号位置", "2号位置"):
                    ent_feats.append(extract_head_input(m, tok, t.prompt))
                    ent_labels.append(1)
                    ent_pos.append(v)
        d["probe_entity_head_input"] = linear_probe(ent_feats, ent_labels, n_classes=2)
        d["probe_entity_pos_balance"] = {"pos1": ent_pos.count("1号位置"),
                                          "pos2": ent_pos.count("2号位置")}
        # Workspace 内容漂移（stable vs 候选，B 组首条）
        if tag != "stable":
            d["ws_content_drift_vs_stable"] = ws_content_drift(models["stable"], m, tok, b_p[0])
            d["ws_content_drift_B_mean_cosine"] = round(
                sum(ws_content_drift(models["stable"], m, tok, p)["slots"][s]["cosine"]
                    for p in b_p[:6] for s in range(2)) / 12, 4)
        # 反事实表示变化：同上下文不同查询实体（手工配对，B 组域）
        shift_a = counterfactual_representation_shift(
            m, tok, "用户：小红在1号位置。\n用户：小红在哪里？\n助手：",
            "用户：小明在1号位置。\n用户：小红在哪里？\n助手：")
        shift_b = counterfactual_representation_shift(
            m, tok, "用户：小红在1号位置。\n用户：小明在哪里？\n助手：",
            "用户：小红在1号位置。\n用户：小红在哪里？\n助手：")
        d["counterfactual_rep_shift"] = {"query_entity_changed": shift_a,
                                          "query_entity_same_control": shift_b}
        diag["models"][tag] = d
        print(f"[diag] {tag}: A={d['a_retention_em']} B={d['b_sealed_em']} "
              f"cf={d['cf_follow_rate']} 位置探针={d['probe_position_head_input']['test_accuracy']} "
              f"实体探针={d['probe_entity_head_input']['test_accuracy']}")

    # ---------- 第四阶段：受控因果干预 ----------
    stable_m = models["stable"]

    def eval_mixed(m) -> dict[str, Any]:
        return {"A_em": _em(m, tok, ap_all, aa_all), "B_em": _em(m, tok, b_p, b_a),
                 "cf_follow": _cf_follow(m, tok, cf_items)}

    interventions: dict[str, Any] = {}
    # 参数替换矩阵（stable 骨架 + 候选模块；候选骨架 + stable 模块）
    plans = [
        ("stable+head(002)", stable_m, models["002"], ["head"]),
        ("stable+b2w(002)", stable_m, models["002"], ["block_to_ws"]),
        ("stable+head+b2w(002)", stable_m, models["002"], ["head", "block_to_ws"]),
        ("stable+b2w(003)", stable_m, models["003"], ["block_to_ws"]),
        ("001+stable_head", models["001"], stable_m, ["head"]),
        ("002+stable_head", models["002"], stable_m, ["head"]),
        ("002+stable_b2w", models["002"], stable_m, ["block_to_ws"]),
        ("002+stable_head+b2w", models["002"], stable_m, ["head", "block_to_ws"]),
        ("003+stable_head", models["003"], stable_m, ["head"]),
    ]
    for name, base, donor, mods in plans:
        m, swapped = _swap_modules(base, donor, mods)
        interventions[name] = {"swapped_params": swapped, **eval_mixed(m)}
        print(f"[swap] {name}: {interventions[name]}")
    # 激活替换：同 prompt 下候选末态换 stable 的 Workspace 内容 / 全部块状态，继续生成
    def _continue_with_replaced(model, tok, prompts, answers, replace_from, what: str) -> float:
        hits = 0
        model.eval()
        with torch.no_grad():
            for p, a in zip(prompts, answers):
                _lg, st_cand = model(torch.tensor([tok.encode(p)], dtype=torch.long))
                _lg2, st_ref = replace_from(torch.tensor([tok.encode(p)], dtype=torch.long))
                if what == "ws":
                    st_cand.ws = copy.deepcopy(st_ref.ws)
                elif what == "h":
                    st_cand.h = st_ref.h.clone()
                rec, _st = generate(model, tok, [], primed=(_lg, st_cand),
                                     max_new_tokens=24, greedy=True)
                hits += int(_norm(rec.text) == _norm(a))
        return round(hits / len(answers), 4)

    for cand_tag in ("002", "003"):
        interventions[f"{cand_tag}+stable_ws"] = {
            "B_em_ws_replaced": _continue_with_replaced(
                models[cand_tag], tok, b_p, b_a, stable_m, "ws"),
            "A_em_ws_replaced": _continue_with_replaced(
                models[cand_tag], tok, ap_all, a_a, stable_m, "ws")}
        interventions[f"{cand_tag}+stable_h"] = {
            "B_em_h_replaced": _continue_with_replaced(
                models[cand_tag], tok, b_p, b_a, stable_m, "h"),
            "A_em_h_replaced": _continue_with_replaced(
                models[cand_tag], tok, ap_all, a_a, stable_m, "h")}
    interventions["stable+002_ws"] = {
        "B_em_ws_replaced": _continue_with_replaced(
            stable_m, tok, b_p, b_a, models["002"], "ws")}
    interventions["stable+002_h"] = {
        "B_em_h_replaced": _continue_with_replaced(
            stable_m, tok, b_p, b_a, models["002"], "h")}
    diag["interventions"] = interventions
    for k, v in interventions.items():
        print(f"[act] {k}: {v}")

    diag["wall_s"] = round(time.time() - t0, 2)
    diag["integrity"] = {
        "stable_untouched": "全程内存副本，磁盘 checkpoint 零写入",
        "methodology_note": "模块组合存在非线性相互作用，单次替换效果不解释为唯一因果来源；"
                             "探针监督只作表示诊断，不计入自主语言能力成绩",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "GATEB004_DIAGNOSTICS.json").write_text(
        json.dumps(diag, ensure_ascii=False, indent=2), encoding="utf-8")
    MIRROR.mkdir(parents=True, exist_ok=True)
    (MIRROR / "GATEB004_DIAGNOSTICS.json").write_text(
        json.dumps(diag, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] GATEB004_DIAGNOSTICS.json 落盘，wall={diag['wall_s']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
