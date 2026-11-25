"""阶段 2.4 匹配计时重测（2026-11-25，AutoClaw）：条件与 20261124 完全一致，另存 env 元数据。

条件：参数匹配基线（GRU 414,316 / Transformer 437,408 / Delta-rule 424,732）、
batch=16、序列长度=32（与 20261124 版一致）、CPU 单线程、20 步、AdamW、CE；
6 次重复丢弃首次预热，报告中位数与 min/max；另测 dev（组合留出 600 条）评估耗时。
输出：timing_matched_20261125.json（不覆盖 20261124 历史工件）。
"""
from __future__ import annotations

import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from models import GRUBaseline, CausalTransformerBaseline, DeltaRuleLinearBaseline  # noqa: E402
from nmind.language.large_binding_task import make_train_test  # noqa: E402
from nmind.language.tokenizer import CharTokenizer  # noqa: E402

OUT = ROOT / "experiments" / "phase2_baselines" / "timing_matched_20261125.json"
TOK = ROOT / "experiments" / "m5_language" / "tokenizer_m52.json"


def env_metadata() -> dict:
    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT),
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        rev = ""
    return {"python": platform.python_version(), "torch": torch.__version__,
             "platform": platform.platform(), "torch_threads": torch.get_num_threads(),
             "git_rev": rev, "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
             "logical_cores": os.cpu_count() if (os := __import__("os")) else None}


def main() -> int:
    torch.set_num_threads(1)
    t_wall = time.perf_counter()
    tok = CharTokenizer.load(str(TOK))
    _, dev = make_train_test(20000, 600, str(TOK))
    data_build_s = time.perf_counter() - t_wall
    V = tok.vocab_size
    x = torch.randint(0, V, (16, 32))
    y = torch.randint(0, V, (16, 32))
    ce = torch.nn.CrossEntropyLoss()
    out = []
    for name, cls in {"gru": GRUBaseline, "transformer": CausalTransformerBaseline,
                       "delta": DeltaRuleLinearBaseline}.items():
        samples = []
        m = None
        for rep in range(6):
            m = cls(V)
            opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
            t = time.perf_counter()
            for _ in range(20):
                opt.zero_grad()
                z = m(x)
                z = z[0] if isinstance(z, tuple) else z
                loss = ce(z.reshape(-1, V), y.reshape(-1))
                loss.backward()
                opt.step()
            samples.append((time.perf_counter() - t) * 1000 / 20)
        ds = []
        t = time.perf_counter()
        with torch.no_grad():
            for r in dev:
                z = torch.tensor(tok.encode(r["prompt"] + r["target"]))
                q = m(z[:-1].unsqueeze(0))
                q = q[0] if isinstance(q, tuple) else q
                ds.append(float(ce(q.reshape(-1, V), z[1:].unsqueeze(0).reshape(-1))))
        out.append({"model": name, "params": sum(p.numel() for p in m.parameters()),
                     "step_ms_median": statistics.median(samples[1:]),
                     "step_ms_min": min(samples[1:]), "step_ms_max": max(samples[1:]),
                     "warmup_discarded": samples[0],
                     "dev_eval_s": time.perf_counter() - t, "dev_n": len(dev),
                     "dev_loss": sum(ds) / len(ds)})
    payload = {"protocol": "20 steps, batch16 x seq32, 6 reps discard first, torch 1 thread",
                "data_build_s": round(data_build_s, 2), "rows": out, "env": env_metadata()}
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
