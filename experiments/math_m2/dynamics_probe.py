"""M2 P5/P14：动力学探针 —— 诊断重复生成与状态循环。

对若干 prompt 逐 token 记录隐藏状态与输出分布，计算：
  - cos_adjacent（相邻状态余弦，趋近 1 = 不动点征兆）
  - fixed_point_ratio（状态变化低于阈值的比例）
  - token_entropy（输出塌缩）
  - unique_token_ratio（重复生成直接证据）
并对比不同架构（尤其 M2-C/M2-E 的序列记忆推进是否抑制循环）。
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from tokenizer_m2 import M2Tokenizer
from candidate_registry import build, candidate_names
from evaluate_m2 import load_rows
from dynamics_analysis import analyze, cos_adjacent

HERE = Path(__file__).resolve().parent
CHECK = HERE / "checkpoints"


def load_best(name):
    """尝试加载某架构 S2 初筛 best；不存在则返回随机初始化对照。"""
    tok = M2Tokenizer()
    model = build(name, tok.vocab_size)
    p = CHECK / f"S2_{name}_best.pt"
    if p.exists():
        blob = torch.load(p, map_location="cpu", weights_only=False)
        model.load_state_dict(blob["state_dict"])
        source = str(p.relative_to(HERE))
    else:
        source = "random_init"
    model.eval()
    return model, source


@torch.inference_mode()
def trace(model, tok, prompt, max_new=14):
    inp = tok.encode_inference(prompt)
    logits, state = model(torch.tensor([inp]))
    states = []
    logits_seq = []
    out = []
    from evaluate_m2 import allowed_ids
    allowed = torch.tensor(allowed_ids())
    for _ in range(max_new):
        tail = logits[0, -1].index_select(0, allowed)
        tid = int(allowed[int(torch.argmax(tail))])
        if tid == 2:  # EOS
            break
        out.append(tid)
        # 取各架构的代表状态向量：GRU/双记忆用 h，Transformer 用 mem 的均值池化
        if "h" in state:
            states.append(state["h"][0].detach().clone())
        elif "mem" in state:
            states.append(state["mem"][0].float().mean(0).detach().clone())
        else:
            first = next(iter(state.values()))
            states.append(first[0].detach().float().reshape(-1).clone())
        logits_seq.append(logits[0, -1].detach().clone())
        logits, state = model(torch.tensor([[tid]]), state)
    st = torch.stack(states) if states else torch.zeros(1, 1)
    lg = torch.stack(logits_seq) if logits_seq else torch.zeros(1, tok.vocab_size)
    return {"text": tok.decode(out), "token_ids": out, "n": len(out),
            "unique_ratio": round(len(set(out)) / max(1, len(out)), 4),
            "dynamics": analyze(st, lg)}


def main():
    torch.set_num_threads(8)
    tok = M2Tokenizer()
    val = load_rows("val")
    # 选 COPY 长样本，重复生成最容易暴露
    copy_rows = [r for r in val if r["task"] in ("S2_copyN", "S3_reverse", "S5_repeat")][:6]
    prompts = [r["prompt"] for r in copy_rows]
    report = {"status": "M2_DYNAMICS_PROBE_DONE", "probes": [], "summary": {}}
    for name in candidate_names():
        model, source = load_best(name)
        traces = [trace(model, tok, p) for p in prompts]
        avg_unique = sum(t["unique_ratio"] for t in traces) / max(1, len(traces))
        avg_fixed = sum(t["dynamics"]["fixed_point_ratio"] for t in traces) / max(1, len(traces))
        max_cos = max((max(t["dynamics"]["cos_adjacent"]) if t["dynamics"]["cos_adjacent"] else 0.0)
                      for t in traces)
        report["probes"].append({"architecture": name, "checkpoint_source": source,
                                 "avg_unique_token_ratio": round(avg_unique, 4),
                                 "avg_fixed_point_ratio": round(avg_fixed, 4),
                                 "max_adjacent_cos": round(max_cos, 4),
                                 "traces": traces[:3]})
        report["summary"][name] = {"avg_unique": round(avg_unique, 4),
                                   "avg_fixed": round(avg_fixed, 4)}
    # 重复生成判定：unique_ratio 低且 fixed_point 高
    report["interpretation"] = (
        "avg_unique_token_ratio 越低、avg_fixed_point_ratio 与 max_adjacent_cos 越高，"
        "越接近不动点循环（重复生成）。序列记忆推进应降低 fixed_ratio、提高 unique。")
    (HERE / "M2_DYNAMICS_PROBE.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
