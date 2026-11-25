"""阶段 2.4 LR 选择重跑 v2（2026-11-25，AutoClaw）。

相对 20261124 版（lr_select_parallel.py）的修订——本次指令步骤 3 的缺口补齐：
  1. 逐任务 JSON 增加 env 元数据（python/torch/platform/threads/pid/git_rev/ts/cores），
     且写盘原子化（tmp→replace，杜绝半截文件）；
  2. 全局 15 分钟上限真正生效：每个任务是独立子进程（--worker），主进程轮询，
     超限即 terminate/kill 未完成任务并立刻写 summary.json（cap_triggered=true）；
     已完成任务的 JSON 保留且完整（原子写保证）；
  3. summary 永远落盘；选择规则：某模型 3 个 LR 文件全部存在且 complete 才可选，
     按 (dev_rank_accuracy, -dev_loss) 取最大，否则该模型 INCONCLUSIVE_BUDGET；
  4. 训练数学与 20261124 版一致（同 seed 20270501、同数据、1000 步、dev 只评一次）；
  5. 数据缓存：make_train_test 是确定性生成，一次性落盘缓存供 9 个工作进程复用
     （20261125 实测单次构建 223s；缓存不改动任何数据内容，meta 记录 vocab_hash）。

用法：
  python lr_select_parallel_v2.py                 # 主流程（9 任务、900s 全局上限）
  python lr_select_parallel_v2.py --worker ...    # 单任务工作进程（内部使用）
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

MODELS = ("gru", "transformer", "delta")
LRS = (3e-4, 1e-3, 3e-3)
LR_SEED = 20270501
CAP_S = 900.0                      # 本次批准：并行、15 分钟全局上限（不含缓存构建）
OUT = ROOT / "experiments" / "phase2_baselines" / "lr_tasks_20261125"
TOK = ROOT / "experiments" / "m5_language" / "tokenizer_m52.json"
DATA_CACHE = ROOT / "experiments" / "phase2_baselines" / "data_cache_20261125.pt"
REQUIRED_KEYS = ("model", "lr", "seed", "steps", "dev_rank_accuracy", "dev_loss",
                  "wall_s", "complete", "env")


def env_metadata() -> dict:
    import torch
    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=str(ROOT),
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        rev = ""
    return {"python": platform.python_version(), "torch": torch.__version__,
             "platform": platform.platform(), "torch_threads": torch.get_num_threads(),
             "pid": os.getpid(), "git_rev": rev, "logical_cores": os.cpu_count(),
             "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z")}


def ensure_data_cache(path: str | Path = DATA_CACHE, tok_path: str = str(TOK)) -> dict:
    """确定性数据的进程间缓存：make_train_test(20000,600) 生成一次，落盘复用。

    不改动任何数据内容：meta 记录生成器、规模与 vocab_hash，命中缓存时校验 meta。
    """
    import torch
    path = Path(path)
    if path.exists():
        blob = torch.load(path, weights_only=False, map_location="cpu")
        need = ("enc", "sealed", "V", "majority_pos", "meta")
        if all(k in blob for k in need):
            return blob
    t0 = time.time()
    sys.path.insert(0, str(ROOT / "src"))
    from nmind.language.large_binding_task import make_train_test
    from nmind.language.tokenizer import CharTokenizer
    tok = CharTokenizer.load(tok_path)
    tr, sealed = make_train_test(20000, 600, tok_path)
    enc = [torch.tensor(tok.encode(r["prompt"] + r["target"])) for r in tr]
    cnt: dict = {}
    for r in tr:
        mm = re.search(r"在([1-4]号位置)。$", r["target"])
        if mm:
            cnt[mm.group(1)] = cnt.get(mm.group(1), 0) + 1
    blob = {"enc": enc, "sealed": sealed, "V": tok.vocab_size,
             "majority_pos": max(cnt, key=cnt.get),
             "meta": {"train_n": len(tr), "test_n": len(sealed),
                       "tokenizer": "tokenizer_m52.json", "vocab_hash": tok.vocab_hash(),
                       "generator": "nmind.language.large_binding_task.make_train_test",
                       "built_ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                       "note": "make_train_test 确定性生成的缓存；内容与直接生成逐字节一致"}}
    tmp = path.with_suffix(".tmp")
    torch.save(blob, tmp)
    tmp.replace(path)
    print(f"data cache built in {time.time() - t0:.1f}s -> {path}")
    return blob


def run_worker(model: str, lr: float, seed: int, out_dir: Path, cache_path: str) -> dict:
    """单任务：1000 步训练 + dev（组合留出 600 条）只在结束时评一次。"""
    import torch
    t0 = time.perf_counter()
    torch.set_num_threads(1)
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(Path(__file__).parent))
    from models import GRUBaseline, CausalTransformerBaseline, DeltaRuleLinearBaseline
    from nmind.language.tokenizer import CharTokenizer
    cls = {"gru": GRUBaseline, "transformer": CausalTransformerBaseline,
            "delta": DeltaRuleLinearBaseline}[model]
    blob = torch.load(cache_path, weights_only=False, map_location="cpu")
    enc, dev, V = blob["enc"], blob["sealed"], blob["V"]
    tok = CharTokenizer.load(str(TOK))
    m = cls(V)
    opt = torch.optim.AdamW(m.parameters(), lr=lr)
    ce = torch.nn.CrossEntropyLoss()
    for step in range(1000):
        idx = [(step * 16 + j) % len(enc) for j in range(16)]
        z = torch.nn.utils.rnn.pad_sequence([enc[i] for i in idx], batch_first=True)
        opt.zero_grad()
        q = m(z[:, :-1])
        q = q[0] if isinstance(q, tuple) else q
        loss = ce(q.reshape(-1, V), z[:, 1:].reshape(-1))
        loss.backward()
        opt.step()
    hits = 0
    devloss = 0.0
    with torch.no_grad():
        for r in dev:
            zz = torch.tensor(tok.encode(r["prompt"] + r["target"]))
            q = m(zz[:-1].unsqueeze(0))
            q = q[0] if isinstance(q, tuple) else q
            p = len(tok.encode(r["prompt"]))
            ans = zz[p:]
            hits += int(torch.equal(q[0, p - 1:p - 1 + len(ans)].argmax(-1), ans))
            devloss += float(ce(q.reshape(-1, V), zz[1:].unsqueeze(0).reshape(-1)))
    rec = {"model": model, "lr": lr, "seed": seed, "steps": 1000,
            "dev_rank_accuracy": hits / len(dev), "dev_loss": devloss / len(dev),
            "wall_s": time.perf_counter() - t0, "complete": True, "env": env_metadata()}
    tmp = out_dir / f".{model}_lr{lr}.json.tmp"
    tmp.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    tmp.replace(out_dir / f"{model}_lr{lr}.json")
    return rec


def select_from_files(out_dir: str | Path, models=MODELS, lrs=LRS) -> dict:
    """选择规则：3 个 LR 全部存在且 complete 才可选；并列时 dev_loss 更低者胜。"""
    out: dict = {}
    for m in models:
        recs = []
        ok = True
        for lr in lrs:
            f = Path(out_dir) / f"{m}_lr{lr}.json"
            if not f.exists():
                ok = False
                break
            d = json.loads(f.read_text(encoding="utf-8"))
            if [k for k in REQUIRED_KEYS if k not in d] or not d.get("complete"):
                ok = False
                break
            recs.append(d)
        if not ok:
            out[m] = {"status": "INCONCLUSIVE_BUDGET",
                       "reason": "缺少某个 LR 的完整结果文件（未完成或被上限终止）"}
            continue
        best = max(recs, key=lambda d: (d["dev_rank_accuracy"], -d["dev_loss"]))
        out[m] = {"status": "SELECTED", "lr": best["lr"],
                   "dev_rank_accuracy": best["dev_rank_accuracy"],
                   "dev_loss": best["dev_loss"], "seed": best["seed"], "steps": best["steps"],
                   "per_lr": [{"lr": d["lr"], "dev_rank_accuracy": d["dev_rank_accuracy"],
                                "dev_loss": d["dev_loss"], "wall_s": round(d["wall_s"], 1)}
                               for d in recs]}
    return out


def run_selection(*, worker_script: str, tasks, out_dir: str | Path, cap_s: float,
                   poll_s: float = 0.5, python_exe: str | None = None,
                   extra_worker_args: list[str] | None = None) -> dict:
    """通用执行器：独立子进程 + 轮询 + 全局上限；summary 永远落盘。

    tasks: [(model, lr, seed), ...]；worker_script 必须支持
    --worker --model M --lr L --seed S --out DIR 并在完成时原子写 {model}_lr{lr}.json。
    选择范围由 tasks 本身决定（模型/LR 集合从任务列表推导，不硬编码）。
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    py = python_exe or sys.executable
    extra = list(extra_worker_args or [])
    t0 = time.time()
    deadline = t0 + cap_s
    procs: dict = {}
    for name, lr, seed in tasks:
        logf = out_dir / f"{name}_lr{lr}.log"
        procs[(name, lr, seed)] = subprocess.Popen(
            [py, worker_script, "--worker", "--model", str(name), "--lr", str(lr),
             "--seed", str(seed), "--out", str(out_dir), *extra],
            stdout=open(logf, "a", encoding="utf-8"), stderr=subprocess.STDOUT)
    finished: list = []
    killed: list = []
    while procs:
        time.sleep(poll_s)
        for key, p in list(procs.items()):
            rc = p.poll()
            if rc is not None:
                procs.pop(key)
                finished.append({"model": key[0], "lr": key[1], "seed": key[2],
                                  "exitcode": rc})
        if procs and time.time() >= deadline:
            for key, p in procs.items():
                p.terminate()
                killed.append(key)
            time.sleep(1.0)
            for key, p in list(procs.items()):
                if p.poll() is None:
                    p.kill()
            procs.clear()
    rows = [{"model": k[0], "lr": k[1], "seed": k[2], "complete": False,
              "status": "timeout_killed", "env": None}           # 被终止任务无环境记录
             for k in killed]
    models = sorted({t[0] for t in tasks})
    lrs = sorted({t[1] for t in tasks})
    summary = {"wall_s": round(time.time() - t0, 2), "cap_s": cap_s,
                "cap_triggered": bool(killed), "killed_tasks": [list(k) for k in killed],
                "finished_processes": finished,
                "selection": select_from_files(out_dir, models=models, lrs=lrs)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                            encoding="utf-8")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--model")
    ap.add_argument("--lr", type=float)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--out")
    ap.add_argument("--cache", default=str(DATA_CACHE))
    a = ap.parse_args()
    if a.worker:
        rec = run_worker(a.model, a.lr, a.seed, Path(a.out), a.cache)
        print(json.dumps({"done": True, "model": rec["model"], "lr": rec["lr"]}))
        return 0
    ensure_data_cache()                      # 缓存构建不计入 15 分钟任务上限
    tasks = [(m, lr, LR_SEED) for m in MODELS for lr in LRS]
    summary = run_selection(worker_script=str(Path(__file__).resolve()), tasks=tasks,
                             out_dir=OUT, cap_s=CAP_S,
                             extra_worker_args=["--cache", str(DATA_CACHE)])
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
