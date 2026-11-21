"""权重保存/加载与回滚。

保存的不只是 state_dict，还带上**配置哈希、数据哈希、随机种子、git 版本**：
"同一权重 + 同一输入 → 同一结果"这条验收（指令 M1）必须能被别人复算，
缺任何一项都只能算"我这边跑出来是这样"。

局部更新（M6）要求"保存每次更新前的模型版本并允许回滚"，
这里用带序号的快照目录实现：更新前 save_snapshot，坏了 load_snapshot 回去。
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from ..config import Config, default_config
from ..core.architecture import NModel


def config_hash(cfg: Config) -> str:
    payload = json.dumps(cfg.model_dump(mode="json"), sort_keys=True, ensure_ascii=False,
                         default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def git_revision(cwd: Path | None = None) -> str:
    """有 git 就记 commit；没有就明确记 'no-git'，绝不写成看似真实的哈希。"""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=str(cwd or Path.cwd()),
                             capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            return out.stdout.strip() or "no-git"
    except (OSError, subprocess.SubprocessError):
        pass
    return "no-git"


def _model_kwargs(cfg: Config, model: NModel) -> dict[str, Any]:
    return {"n_actions": model.n_actions, "mem_dim": model.mem_dim,
            "execution": model.execution, "use_router": model.use_router,
            "use_error_gating": model.use_error_gating, "use_workspace": model.use_workspace,
            "top_k": model.top_k}


def save_checkpoint(model: NModel, *, cfg: Config | None = None, path: str | Path,
                    probe: torch.nn.Module | None = None, meta: dict[str, Any] | None = None) -> str:
    cfg = cfg or default_config()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "format": "nmind-ckpt/1",
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": cfg.model_dump(mode="json"),
        "config_hash": config_hash(cfg),
        "model_kwargs": _model_kwargs(cfg, model),
        "git_revision": git_revision(p.parent.parent if p.parent.name == "results" else None),
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "probe": None if probe is None else {k: v.detach().cpu() for k, v in probe.state_dict().items()},
        "meta": meta or {},
    }
    torch.save(blob, p)
    return str(p)


def _resolve_device(device: str | torch.device) -> torch.device:
    """把配置里的 "auto" 变成真设备。

    torch.load 的 map_location 不认识 "auto"，会把 "auto" 当成序列化的设备标签
    直接报 "don't know how to restore data location"。这个坑每台机器都会踩一次，
    所以在唯一入口处收口，而不是要求每个调用方先自己探测。
    """
    if isinstance(device, torch.device):
        return device
    if device in ("auto", "", None):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_checkpoint(path: str | Path, *, cfg: Config | None = None,
                    device: str | torch.device = "cpu", strict: bool = True,
                    allow_config_mismatch: bool = False) -> tuple[NModel, dict[str, Any]]:
    """读回权重并**按保存时的装配参数重建结构**。

    结构不一致时报错而不是静默插值：加载错结构的权重会得到一个看起来很正常
    但毫无意义的模型，那比直接失败糟糕得多。
    """
    dev = _resolve_device(device)
    p = Path(path)
    blob = torch.load(p, map_location=dev, weights_only=True)
    if blob.get("format") != "nmind-ckpt/1":
        raise ValueError(f"不是 nmind checkpoint（format={blob.get('format')!r}）")
    saved_cfg = Config(**blob["config"])
    if cfg is not None and not allow_config_mismatch:
        if config_hash(cfg) != blob["config_hash"]:
            raise ValueError("配置文件与 checkpoint 不一致：model 形状可能不同。"
                             "确认要强行加载请用 allow_config_mismatch=True")
    model = NModel(saved_cfg.model, **blob["model_kwargs"])
    missing, unexpected = _load_state(model, blob["state_dict"], strict=strict)
    model.to(dev).eval()
    meta = dict(blob.get("meta") or {})
    meta.update({"config_hash": blob["config_hash"], "saved_at": blob["saved_at"],
                 "git_revision": blob.get("git_revision", "unknown"),
                 "missing_keys": list(missing), "unexpected_keys": list(unexpected)})
    return model, meta


def _load_state(model: NModel, state: dict[str, Any], *, strict: bool):
    try:
        model.load_state_dict(state, strict=strict)
        return [], []
    except RuntimeError as e:
        if strict:
            raise
        ms = model.load_state_dict(state, strict=False)
        return ms.missing_keys, ms.unexpected_keys


def probe_state(probe: torch.nn.Module) -> dict[str, Any]:
    return {k: v.detach().cpu() for k, v in probe.state_dict().items()}


def load_probe(probe: torch.nn.Module, state: dict[str, Any] | None, *, device="cpu") -> bool:
    if not state:
        return False
    probe.load_state_dict(state, strict=True)
    probe.to(device).eval()
    return True


# ---------------------------------------------------------------------- 回滚快照

def save_snapshot(model: NModel, directory: str | Path, *, tag: str = "",
                  meta: dict[str, Any] | None = None) -> str:
    """更新前拍一张快照；序号递增，便于 load_snapshot(rollback_to_previous)。"""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    existing = sorted(d.glob("snapshot-*.pt"))
    n = len(existing) + 1
    name = f"snapshot-{n:04d}{('-' + tag) if tag else ''}.pt"
    path = d / name
    torch.save({"index": n, "tag": tag, "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "meta": meta or {}}, path)
    return str(path)


def list_snapshots(directory: str | Path) -> list[dict[str, Any]]:
    d = Path(directory)
    out = []
    for f in sorted(d.glob("snapshot-*.pt")):
        blob = torch.load(f, map_location="cpu", weights_only=True)
        out.append({"path": str(f), "index": blob["index"], "tag": blob.get("tag", ""),
                    "meta": blob.get("meta", {})})
    return out


def load_snapshot(model: NModel, path: str | Path) -> dict[str, Any]:
    blob = torch.load(Path(path), map_location="cpu", weights_only=True)
    model.load_state_dict(blob["state_dict"], strict=True)
    return {"index": blob.get("index"), "tag": blob.get("tag", ""), "meta": blob.get("meta", {})}
