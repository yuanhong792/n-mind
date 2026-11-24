"""M41 严格能力指标审计（无训练、无 optimizer.step）。

验证：
- 原始生成 token 序列必须等于 answer_ids + EOS；
- 自然 EOS、非法特殊 token 不得被 decode 过滤后伪装为正确；
- hard_gate 默认启用；
- ISO 必须连续两个独立验证点，成本取第二个确认点；
- 单 seed 独立判断，关键任务下限/硬门/严格指标缺失都会阻断达标。

输出 M41_METRIC_AUDIT.json。
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import torch

from evaluate_m4 import (EOS, SEP, PAD, eval_one, eval_rows, generate, iso_record,
                         load_critical_minimums)
from tokenizer_m4 import M4Tokenizer

HERE = Path(__file__).resolve().parent
V = 102


class ProgrammedModel:
    """只按预设 token 序列输出 logits，用于精确验证指标函数，不训练。"""

    def __init__(self, ids):
        self.ids = list(ids)
        self.calls = []

    def initial_state(self, batch, device):
        return {"ptr": 0}

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, tokens, state=None, valid_mask=None, hard_gate=False):
        self.calls.append({"T": int(tokens.shape[1]), "hard_gate": bool(hard_gate)})
        ptr = state["ptr"]
        tid = self.ids[ptr % len(self.ids)]
        state["ptr"] = ptr + 1
        logits = torch.full((tokens.shape[0], tokens.shape[1], V), -100.0)
        logits[:, :, tid] = 100.0
        return logits, state


def point(step, strict_macro, cost, crit=None, hard_gate=True):
    mins = load_critical_minimums()
    c = {k: 1.0 for k in mins}
    if crit:
        c.update(crit)
    return {"step": step, "strict_macro": strict_macro, "hard_gate": hard_gate,
            "critical": c, "cumulative_macs": cost}


def audit():
    tok = M4Tokenizer()
    row = {"task": "A2_copy2", "prompt": "COPY 7", "answer": "7"}
    a7 = tok.encode("7")[0]
    checks = {}

    m_ok = ProgrammedModel([a7, EOS])
    e_ok = eval_one(m_ok, tok, row)
    checks["STRICT_TOKEN_EOS_OK"] = {
        "pass": bool(e_ok["strict_ok"] and e_ok["finish_reason"] == "eos"),
        "observed": {"raw_ids": e_ok["raw_ids"], "expected_ids": e_ok["expected_ids"],
                     "strict_ok": e_ok["strict_ok"],
                     "finish_reason": e_ok["finish_reason"],
                     "forward_calls": m_ok.calls},
        "required": {"strict_ok": True, "finish_reason": "eos",
                     "ids_equal": True},
    }

    m_hidden = ProgrammedModel([a7, SEP, EOS])
    e_hidden = eval_one(m_hidden, tok, row)
    hidden_reject = (e_hidden["decoded_content_ok"] and not e_hidden["strict_ok"]
                     and 1 in e_hidden["invalid_special_tokens"])
    checks["DECODE_FILTER_CANNOT_HIDE_SPECIAL_TOKEN"] = {
        "pass": bool(hidden_reject),
        "observed": {"raw_ids": e_hidden["raw_ids"], "expected_ids": e_hidden["expected_ids"],
                     "decoded_content_ok": e_hidden["decoded_content_ok"],
                     "strict_ok": e_hidden["strict_ok"],
                     "invalid_special_tokens": e_hidden["invalid_special_tokens"]},
        "required": "decode 文本相同也必须因 SEP/PAD/BOS/UNK 原始 token 判失败",
    }

    m_pad = ProgrammedModel([a7, PAD, EOS])
    e_pad = eval_one(m_pad, tok, row)
    checks["PAD_CANNOT_BE_FILTERED"] = {
        "pass": bool(e_pad["decoded_content_ok"] and not e_pad["strict_ok"]),
        "observed": {"raw_ids": e_pad["raw_ids"],
                     "strict_ok": e_pad["strict_ok"],
                     "invalid_special_tokens": e_pad["invalid_special_tokens"]},
        "required": {"strict_ok": False},
    }

    m_long = ProgrammedModel([a7, tok.encode("8")[0]])
    g_long = generate(m_long, tok, row["prompt"], gen_limit=2)
    checks["NO_NATURAL_EOS_REJECTED"] = {
        "pass": bool(g_long["finish_reason"] == "length" and g_long["ids"] != [a7, EOS]),
        "observed": {"raw_ids": g_long["ids"], "finish_reason": g_long["finish_reason"]},
        "required": {"finish_reason": "length", "strict_ok": False},
    }

    m_gate = ProgrammedModel([a7, EOS])
    ev_gate = eval_rows(m_gate, tok, [row], hard_gate=True)
    checks["HARD_GATE_DEFAULT"] = {
        "pass": bool(m_gate.calls and all(c["hard_gate"] for c in m_gate.calls)
                     and inspect.signature(eval_rows).parameters["hard_gate"].default is True),
        "observed": {"forward_calls": m_gate.calls,
                     "eval_rows_default": str(inspect.signature(
                         eval_rows).parameters["hard_gate"].default),
                     "strict_macro": ev_gate["strict_macro"]},
        "required": {"hard_gate": True, "eval_rows_default": True},
    }

    two = [point(64, 0.80, 1000), point(128, 0.82, 1600)]
    iso_two = iso_record(two, seed=1)
    checks["ISO_TWO_CONSECUTIVE_POINTS"] = {
        "pass": iso_two["80%"]["status"] == "REACHED_STABLE"
                and iso_two["80%"]["confirmed_step"] == 128
                and iso_two["80%"]["cost"] == 1600,
        "observed": iso_two["80%"],
        "required": {"status": "REACHED_STABLE", "confirmed_step": 128,
                     "cost": 1600},
    }

    final_only = [point(64, 0.20, 1000), point(128, 0.20, 1600), point(192, 0.85, 2400)]
    iso_final = iso_record(final_only, seed=1)
    checks["ISO_FINAL_SINGLE_POINT_REJECTED"] = {
        "pass": iso_final["80%"]["status"] == "TARGET_NOT_REACHED",
        "observed": iso_final["80%"],
        "required": {"status": "TARGET_NOT_REACHED"},
    }

    crit_bad = [point(64, 0.85, 1000, {"C_reassign": 0.0}),
                point(128, 0.85, 1600, {"C_reassign": 0.0})]
    iso_crit = iso_record(crit_bad, seed=1)
    checks["ISO_CRITICAL_FLOOR_REQUIRED"] = {
        "pass": iso_crit["80%"]["status"] == "TARGET_NOT_REACHED",
        "observed": iso_crit["80%"],
        "required": "C_reassign 低于预登记下限不得达标",
    }

    gate_bad = [point(64, 0.85, 1000, hard_gate=False),
                point(128, 0.85, 1600, hard_gate=False)]
    iso_gate = iso_record(gate_bad, seed=1)
    checks["ISO_HARD_GATE_REQUIRED"] = {
        "pass": iso_gate["80%"]["status"] == "TARGET_NOT_REACHED",
        "observed": iso_gate["80%"],
        "required": "软门结果不得计入部署能力",
    }

    cost_missing = [{"step": 64, "strict_macro": 0.85, "hard_gate": True,
                     "critical": {k: 1.0 for k in load_critical_minimums()}},
                    {"step": 128, "strict_macro": 0.85, "hard_gate": True,
                     "critical": {k: 1.0 for k in load_critical_minimums()}}]
    iso_cost = iso_record(cost_missing, seed=1)
    checks["ISO_NO_FICTIONAL_COST"] = {
        "pass": (iso_cost["80%"]["status"] == "REACHED_STABLE_COST_UNAVAILABLE"
                 and iso_cost["80%"]["cost"] is None),
        "observed": iso_cost["80%"],
        "required": {"cost": None, "status": "REACHED_STABLE_COST_UNAVAILABLE"},
    }

    seed1 = iso_record(two, seed=1)["80%"]
    seed2 = iso_record(two, seed=2)["80%"]
    checks["ISO_SEED_INDEPENDENT"] = {
        "pass": seed1["seed"] == 1 and seed2["seed"] == 2
                and seed1["cost"] == seed2["cost"],
        "observed": {"seed1": seed1["seed"], "seed2": seed2["seed"],
                     "status1": seed1["status"], "status2": seed2["status"]},
        "required": "每个 seed 单独记录，不做平均值判定",
    }

    status = "PASS" if all(v["pass"] for v in checks.values()) else "FAILED"
    out = {"status": status, "checks": checks,
           "critical_minimums": load_critical_minimums(),
           "note": "指标测试使用预设 logits 的桩模型，不执行训练，不代表任务能力。"}
    p = HERE / "M41_METRIC_AUDIT.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[m41-metric] status={status} 写出 {p}")
    return out


if __name__ == "__main__":
    audit()
