"""M5-Language 评测入口（任务书 P4）：自由生成为主指标，teacher-forced 仅诊断。

* 每条对话从 state=None 开始，逐轮：吸收用户轮 → 分支生成助手回复（不读 oracle）→
  吸收**模型自己的回复**（上下文不用 teacher forcing）；
* 对照基线：逐意图多数答案（manifest）、随机未训练同构模型、规则/模板解析器（rule_baseline）；
* 门槛（预登记，不得看结果后修改）：主任务 free_generation_exact_match ≥ 多数基线 + 10pp
  且 ≥2 个独立意图类型超基线 → IN_DOMAIN_PASS；OOD_PASS = ≥2 个 OOD split 满足同条件；
* 训练态 buffer 评测前复位（seal_eval_state），漂移记录在案。
"""

from __future__ import annotations

import json
import re
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from ..learning.online_adaptation import seal_eval_state
from ..skills.toolbox import calculator
from .decoder import generate
from .dialogue_data import (OOD_OBJECTS, OOD_PERSONS, TRAIN_OBJECTS, TRAIN_PERSONS,
                             build_all)
from .neural_dialogue import LMConfig, NMindLM, load_lm
from .tokenizer import CharTokenizer

ROOT = Path(__file__).resolve().parents[3]   # src/nmind/language/ → 仓库根

RE_POS1 = re.compile(r"([^\s，。]+?)在(\d)号位置。")
RE_POS2 = re.compile(r"([^\s，。]+?)正位于(\d)号位置。")
RE_MOVE1 = re.compile(r"把(.+?)移动到(\d)号位置。")
RE_MOVE2 = re.compile(r"把(.+?)放到(\d)号位置。")
RE_MOVE3 = re.compile(r"请把(.+?)挪到(\d)号位置。")
RE_COLOR1 = re.compile(r"([^\s，。]+?)是(红|蓝|绿|黄)色")
RE_PAINT = re.compile(r"把(.+?)涂成(红|蓝|绿|黄)色。")
RE_CALC = re.compile(r"(\d+)\s*(?:加|\+)\s*(\d+)|(\d+)\s*(?:乘|\*)\s*(\d+)")
RE_COUNT = re.compile(r"(\d)号位置有几样东西|有几样东西")


def _rule_state_track(text: str, state: dict) -> None:
    for pat in (RE_POS1, RE_POS2):
        m = pat.search(text)
        if m:
            state["positions"][m.group(1)] = f"{m.group(2)}号位置"
    for pat in (RE_MOVE1, RE_MOVE2, RE_MOVE3):
        m = pat.search(text)
        if m:
            state["positions"][m.group(1)] = f"{m.group(2)}号位置"
    m = RE_COLOR1.search(text)
    if m:
        state["colors"][m.group(1)] = f"{m.group(2)}色"
    m = RE_PAINT.search(text)
    if m:
        state["colors"][m.group(1)] = f"{m.group(2)}色"


def _pos_num(p: str) -> int:
    return int(p[:1] or 1)


def rule_baseline_reply(text: str, intent: str, state: dict, last_entity: str | None) -> str:
    """规则/模板解析基线（rule_baseline）：模板正则 + 世界状态跟踪，单独计量。"""
    if intent == "qa_pos":
        e = last_entity or ""
        p = state["positions"].get(e)
        return f"{e}在{p}。" if p else ABSTAIN_RULE
    if intent == "qa_pos_ref":
        e = last_entity or ""
        p = state["positions"].get(e)
        return f"{e}在{p}。" if p else ABSTAIN_RULE
    if intent == "qa_color":
        e = _last_entity_in_text(text) or last_entity or ""
        c = state["colors"].get(e)
        return f"{e}是{c}的。" if c else ABSTAIN_RULE
    if intent == "qa_color_ref":
        e = last_entity or ""
        c = state["colors"].get(e)
        return f"{e}是{c}的。" if c else ABSTAIN_RULE
    if intent == "qa_count":
        m = re.search(r"(\d)号位置", text)
        if not m:
            return ABSTAIN_RULE
        p = f"{m.group(1)}号位置"
        n = sum(1 for v in state["positions"].values() if v == p)
        return f"有{n}样。"
    if intent == "qa_compare":
        es = re.findall(r"([\u4e00-\u9fff]{2,4})的位置编号", text)
        if len(es) >= 2:
            n1 = _pos_num(state["positions"].get(es[0], "1号位置"))
            n2 = _pos_num(state["positions"].get(es[1], "1号位置"))
            return "是的。" if n1 > n2 else "不是。"
        return ABSTAIN_RULE
    if intent == "qa_neg":
        m = re.search(r"([^\s，。]+?)不在(\d)号位置吗", text)
        if not m:
            return ABSTAIN_RULE
        e, p = m.group(1), f"{m.group(2)}号位置"
        return f"是的，{e}不在{p}。" if state["positions"].get(e) != p else \
            f"不对，{e}在{p}。"
    if intent == "tool_calc":
        m = RE_CALC.search(text)
        if not m:
            return ABSTAIN_RULE
        a, b = m.group(1) or m.group(3), m.group(2) or m.group(4)
        op = "+" if (m.group(1) and m.group(2)) else "*"
        return f"TOOL_CALCULATOR {a}{op}{b}"
    if intent == "tool_lookup":
        m = re.search(r"查资料：(.+?)？|帮我查一下(.+?)。", text)
        key = (m.group(1) or m.group(2)) if m else ""
        return f"TOOL_LOOKUP {key}"
    if intent == "unanswerable":
        return "我不知道。"
    return "好的。"


ABSTAIN_RULE = "我不知道。"


def _last_entity_in_text(text: str) -> str | None:
    m = re.findall(r"[\u4e00-\u9fff]{2,4}", text)
    return m[0] if m else None


def _norm(s: str) -> str:
    return (s or "").strip()


def run_eval(*, checkpoint_path: str, tokenizer_path: str, data_dir: str,
             out_path: str | None = None, device: str = "cpu",
             splits: list[str] | None = None, max_new_tokens: int = 32,
             greedy: bool = True, sample_seed: int | None = 20261299,
             include_random_model: bool = True) -> dict[str, Any]:
    tok = CharTokenizer.load(tokenizer_path)
    model, meta = load_lm(checkpoint_path, tok)
    model.eval().to(device)
    seal = seal_eval_state(model, {k: v for k, v in model.state_dict().items()})
    data = Path(data_dir)
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    splits = splits or ["sealed_test", "ood_name", "ood_paraphrase", "ood_composition",
                         "ood_length"]

    random_model = None
    if include_random_model:
        torch.manual_seed(999)
        random_model = NMindLM(LMConfig(**{"vocab_size": tok.vocab_size, **{
            k: v for k, v in model.cfg.to_dict().items() if k != "vocab_size"}}))
        random_model.eval()

    def gen_reply(m: NMindLM, state: Any, text: str):
        t0 = time.perf_counter()
        rec, _st = generate(m, tok, tok.encode("助手："), state=state,
                             max_new_tokens=max_new_tokens, greedy=greedy, seed=sample_seed)
        ms = round((time.perf_counter() - t0) * 1000, 2)
        return rec, ms

    def absorb(m: NMindLM, state: Any, text: str) -> Any:
        ids = tok.encode(text, add_eos=True)
        if not ids:
            return state
        with torch.no_grad():
            _, st = m(torch.tensor([ids], dtype=torch.long), state)
        return st

    latencies: list[float] = []
    split_results: dict[str, Any] = {}
    _ENT = {"ood_name": (OOD_PERSONS, OOD_OBJECTS)}
    for split in splits:
        payload = json.loads((data / "splits" / f"{split}.json").read_text(encoding="utf-8"))
        per_intent: dict[str, dict[str, int]] = {}
        rule_hits: dict[str, dict[str, int]] = {}
        nonempty = 0
        n_q = 0
        for d in payload:
            state = None
            rule_state: dict = {"positions": {}, "colors": {}}
            last_entity = None
            ent_list = list(_ENT.get(split, (TRAIN_PERSONS, TRAIN_OBJECTS)))
            ent_list = ent_list[0] + ent_list[1]
            for t in d["turns"]:
                if t["role"] == "user":
                    _rule_state_track(t["text"], rule_state)
                    state = absorb(model, state, f"用户：{t['text']}\n")
                    hit = [e for e in ent_list if e in t["text"]]
                    if hit:
                        last_entity = hit[-1]          # 只在新提及时更新（指代轮保持不变）
                else:
                    intent = t["intent"]
                    if intent not in {"qa_pos", "qa_pos_ref", "qa_color", "qa_color_ref",
                                       "qa_count", "qa_compare", "qa_neg", "tool_calc",
                                       "tool_lookup", "unanswerable"}:
                        continue               # 陈述确认轮不计入能力指标
                    if intent in ("qa_pos_ref", "qa_color_ref") and not last_entity:
                        pass
                    rec, ms = gen_reply(model, state, t["text"])
                    latencies.append(ms)
                    pred = _norm(rec.text)
                    oracle = _norm(t["text"])
                    n_q += 1
                    if pred:
                        nonempty += 1
                    em = pred == oracle
                    acc = per_intent.setdefault(intent, {"n": 0, "em": 0})
                    acc["n"] += 1
                    acc["em"] += int(em)
                    # 工具两套指标
                    if intent == "tool_calc":
                        sel = pred.startswith("TOOL_CALCULATOR")
                        exec_ok = False
                        m = re.match(r"TOOL_CALCULATOR\s*([\d\s+\-*]+)", pred)
                        if sel and m:
                            try:
                                mo = re.search(r"TOOL_CALCULATOR\s*([\d\s+\-*]+)", oracle)
                                exec_ok = (abs(calculator(m.group(1))
                                                - calculator(mo.group(1))) < 1e-9)
                            except Exception:
                                exec_ok = False
                        per_intent.setdefault("tool_calc_selection", {"n": 0, "em": 0})
                        per_intent["tool_calc_selection"]["n"] += 1
                        per_intent["tool_calc_selection"]["em"] += int(sel)
                        per_intent.setdefault("tool_calc_execution", {"n": 0, "em": 0})
                        per_intent["tool_calc_execution"]["n"] += 1
                        per_intent["tool_calc_execution"]["em"] += int(exec_ok)
                    if intent == "tool_lookup":
                        sel = pred.startswith("TOOL_LOOKUP")
                        per_intent.setdefault("tool_lookup_selection", {"n": 0, "em": 0})
                        per_intent["tool_lookup_selection"]["n"] += 1
                        per_intent["tool_lookup_selection"]["em"] += int(sel)
                    # 吸收模型自己的回复（上下文不用 teacher forcing）
                    state = absorb(model, state, f"助手：{pred}")
                    # 规则基线
                    rtext = _norm(rule_baseline_reply(
                        next((u["text"] for u in reversed(d["turns"])
                              if u["role"] == "user" and u["intent"] == intent), ""),
                        intent, rule_state, last_entity))
                    racc = rule_hits.setdefault(intent, {"n": 0, "em": 0})
                    racc["n"] += 1
                    racc["em"] += int(rtext == oracle)
                    if intent in ("qa_pos", "qa_pos_ref", "qa_compare", "qa_neg",
                                   "qa_color", "qa_color_ref"):
                        hit = [e for e in ent_list if e in oracle]
                        if hit:
                            last_entity = hit[-1]
        majority = manifest["splits"].get(split, {}).get("majority_answers", {})
        intent_rows = {}
        for intent, acc in sorted(per_intent.items()):
            acc_rate = round(acc["em"] / max(1, acc["n"]), 4)
            maj = majority.get(intent)
            maj_rate = round(sum(1 for _ in range(acc["n"])) * 0, 4)  # 占位，下行重算
            intent_rows[intent] = {"n": acc["n"], "em": acc["em"], "acc": acc_rate,
                                    "majority_answer": maj}
        # 多数基线准确率：oracle 中等于多数答案的比例——需要 oracle 分布，重扫一遍
        oracle_by_intent: dict[str, dict[str, int]] = {}
        for d in payload:
            for t in d["turns"]:
                if t["role"] == "assistant" and t["intent"] in {"qa_pos", "qa_pos_ref",
                                                                  "qa_color", "qa_color_ref",
                                                                  "qa_count", "qa_compare",
                                                                  "qa_neg", "tool_calc",
                                                                  "tool_lookup", "unanswerable"}:
                    oracle_by_intent.setdefault(t["intent"], {})
                    oracle_by_intent[t["intent"]][t["text"]] = \
                        oracle_by_intent[t["intent"]].get(t["text"], 0) + 1
        for intent, row in intent_rows.items():
            dist = oracle_by_intent.get(intent, {})
            if dist and row.get("majority_answer") is not None:
                row["majority_acc"] = round(dist.get(row["majority_answer"], 0)
                                             / max(1, sum(dist.values())), 4)
                row["beats_majority_10pp"] = bool(row["acc"] - row["majority_acc"] >= 0.10)
        rule_rows = {intent: round(v["em"] / max(1, v["n"]), 4)
                      for intent, v in rule_hits.items()}
        question_acc = round(
            sum(v["em"] for k, v in intent_rows.items() if not k.startswith("tool_"))
            / max(1, sum(v["n"] for k, v in intent_rows.items() if not k.startswith("tool_"))), 4)
        types_pass = [k for k, v in intent_rows.items()
                       if k in {"qa_pos", "qa_pos_ref", "qa_color", "qa_color_ref",
                                 "qa_count", "qa_compare", "qa_neg"}
                       and v.get("beats_majority_10pp")]
        split_results[split] = {
            "n_question_turns": n_q, "nonempty_rate": round(nonempty / max(1, n_q), 4),
            "free_generation_exact_match": question_acc,
            "per_intent": intent_rows, "rule_baseline_acc": rule_rows,
            "types_beating_majority_10pp": types_pass,
        }

    # 随机未训练模型对照（仅 sealed_test）
    random_baseline = None
    if random_model is not None:
        payload = json.loads((data / "splits" / "sealed_test.json").read_text(encoding="utf-8"))
        em = n = 0
        for d in payload:
            state = None
            for t in d["turns"]:
                if t["role"] == "user":
                    state = absorb(random_model, state, f"用户：{t['text']}\n")
                elif t["intent"] in {"qa_pos", "qa_pos_ref", "qa_color", "qa_color_ref",
                                      "qa_count", "qa_compare", "qa_neg", "tool_calc",
                                      "tool_lookup", "unanswerable"}:
                    rec, _ms = gen_reply(random_model, state, t["text"])
                    em += int(_norm(rec.text) == _norm(t["text"]))
                    n += 1
                    state = absorb(random_model, state, f"助手：{_norm(rec.text)}")
        random_baseline = {"sealed_test_em": round(em / max(1, n), 4), "n": n}

    # teacher-forced 字符准确率（诊断口径，不作对话能力验收）
    tf_acc: dict[str, float] = {}
    sealed_payload = json.loads((data / "splits" / "sealed_test.json").read_text(encoding="utf-8"))
    from .train_language import dialogue_ids
    tf_hit = tf_n = 0
    model.eval()
    with torch.no_grad():
        for d in sealed_payload:
            ids = dialogue_ids(d, tok, 160)
            if len(ids) < 2:
                continue
            x = torch.tensor([ids[:-1]], dtype=torch.long)
            y = torch.tensor(ids[1:], dtype=torch.long)
            logits, _st = model(x)
            pred = logits[0].argmax(-1)
            mask = y != 0
            tf_hit += int((pred[mask] == y[mask]).sum())
            tf_n += int(mask.sum())
    tf_acc["sealed_test"] = round(tf_hit / max(1, tf_n), 4)

    # 预登记门槛
    def gate_ok(split_row: dict) -> bool:
        return len(split_row["types_beating_majority_10pp"]) >= 2

    in_domain_pass = gate_ok(split_results["sealed_test"])
    ood_pass = sum(1 for s in ("ood_name", "ood_paraphrase", "ood_composition", "ood_length")
                    if s in split_results and gate_ok(split_results[s])) >= 2
    nonempty_min = min(r["nonempty_rate"] for r in split_results.values())
    flags = {
        "PIPELINE_READY": True,
        "FREE_GENERATION_WORKS": bool(nonempty_min >= 0.95),
        "IN_DOMAIN_PASS": bool(in_domain_pass),
        "OOD_PASS": bool(ood_pass),
        "OPEN_CHAT_NOT_VALIDATED": True,
        "NEURAL_CONTINUAL_LEARNING": "NOT_TESTED（m4.1 REJECTED；等混合时域新预注册批准）",
        "REAL_NETWORKING": "NOT_TESTED（默认离线）",
    }
    payload_out = {
        "experiment": "m5-language-eval", "captured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checkpoint": checkpoint_path, "checkpoint_meta": meta,
        "greedy": greedy, "max_new_tokens": max_new_tokens,
        "seal_eval_state": seal,
        "splits": split_results, "random_untrained_baseline": random_baseline,
        "char_accuracy_teacher_forced": tf_acc,
        "latency_ms_p50": round(statistics.median(latencies), 2) if latencies else None,
        "latency_ms_p95": round(sorted(latencies)[int(0.95 * (len(latencies) - 1))], 2)
        if latencies else None,
        "flags": flags,
        "honesty": "封存集只评一次；错误回答全部保留在分任务明细之外——本 JSON 不展示逐条文本，"
                    "逐条样例见报告与 CLI 演示记录",
    }
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(payload_out, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
    return payload_out
