"""阶段 2.4 步骤 5（阶梯完成后执行）：匹配基线 1/4/8/16/20 线程计时重测。

条件与 timing_matched_20261125 一致（batch=16、seq=32、20 步、AdamW+CE、V=31），
每个线程档重建模型、3 次重复取中位数（丢弃首次预热）；不与阶梯训练并行。
输出：timing_threads_20261125.json。不改任何模型代码。
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from models import GRUBaseline, CausalTransformerBaseline, DeltaRuleLinearBaseline  # noqa: E402

OUT = ROOT / "experiments" / "phase2_baselines" / "timing_threads_20261125.json"
THREADS = (1, 4, 8, 16, 20)


def main() -> int:
    V = 31
    x = torch.randint(0, V, (16, 32))
    y = torch.randint(0, V, (16, 32))
    ce = torch.nn.CrossEntropyLoss()
    rows = []
    for nthread in THREADS:
        torch.set_num_threads(nthread)
        for name, cls in {"gru": GRUBaseline, "transformer": CausalTransformerBaseline,
                           "delta": DeltaRuleLinearBaseline}.items():
            samples = []
            m = None
            for rep in range(4):
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
            rows.append({"threads": nthread, "model": name,
                          "params": sum(p.numel() for p in m.parameters()),
                          "step_ms_median": round(statistics.median(samples[1:]), 3),
                          "step_ms_min": round(min(samples[1:]), 3),
                          "step_ms_max": round(max(samples[1:]), 3)})
            print(rows[-1])
    torch.set_num_threads(1)
    OUT.write_text(json.dumps({"protocol": "20 steps batch16xseq32, 4 reps discard first, V=31",
                                "rows": rows, "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z")},
                               indent=2), encoding="utf-8")
    print("saved", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
