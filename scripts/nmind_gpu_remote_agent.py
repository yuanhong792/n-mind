"""Opt-in, pull-only N-MIND Windows GPU trainer. No remote shell or open ports."""
from __future__ import annotations

import argparse
import base64
import json
import platform
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPOSITORY = "yuanhong792/n-mind"
BRANCH = "experiment/ncell-architecture-v37-20261123"
REMOTE_PATH = "experiments/v37_ncell_architecture/REMOTE_GPU_REQUEST.json"
URL = (f"https://api.github.com/repos/{REPOSITORY}/contents/{REMOTE_PATH}"
       f"?ref={urllib.parse.quote(BRANCH, safe='')}")
ARMS = frozenset(("legacy", "router_ste", "attention", "router_attention", "reset", "all"))
ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
SHA_PATTERN = re.compile(r"^[a-f0-9]{40}$")


def fetch_request() -> dict | None:
    req = urllib.request.Request(URL, headers={
        "Accept": "application/vnd.github+json",
        "User-Agent": "N-MIND-GPU-Agent", "Cache-Control": "no-cache"})
    try:
        with urllib.request.urlopen(req, timeout=20) as reply:
            envelope = json.load(reply)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise
    if envelope.get("type") != "file" or envelope.get("encoding") != "base64":
        raise ValueError("unexpected GitHub response")
    body = base64.b64decode(envelope["content"])
    if len(body) > 8192:
        raise ValueError("request too large")
    doc = json.loads(body.decode("utf-8"))
    if not isinstance(doc, dict):
        raise ValueError("request must be an object")
    return doc


def _integer(doc: dict, key: str, lo: int, hi: int) -> int:
    value = doc.get(key)
    if type(value) is not int or not lo <= value <= hi:
        raise ValueError(f"{key} must be integer in {lo}..{hi}")
    return value


def validate_request(doc: dict | None) -> dict | None:
    if doc is None or doc.get("enabled") is not True:
        return None
    if doc.get("schema") != "nmind-remote-gpu/1":
        raise ValueError("invalid request schema")
    if doc.get("action") not in ("train", "cancel"):
        raise ValueError("only train or cancel allowed")
    rid = doc.get("request_id")
    if not isinstance(rid, str) or not ID_PATTERN.fullmatch(rid):
        raise ValueError("unsafe request id")
    if doc["action"] == "cancel":
        target = doc.get("target_request_id")
        if not isinstance(target, str) or not ID_PATTERN.fullmatch(target):
            raise ValueError("unsafe cancel target")
        if set(doc) != {"schema", "enabled", "action", "request_id", "target_request_id"}:
            raise ValueError("unrecognized cancel parameters")
        return doc
    expected = {"schema", "enabled", "action", "request_id", "source_revision",
                "steps", "arms", "seed", "batch_size", "eval_samples",
                "max_seconds_per_arm", "min_vram_gb", "requires_gpu"}
    if set(doc) != expected:
        raise ValueError("unexpected or missing training parameters")
    if not isinstance(doc.get("source_revision"), str) or not SHA_PATTERN.fullmatch(doc["source_revision"]):
        raise ValueError("source_revision must be exact commit SHA")
    steps = _integer(doc, "steps", 1, 2048)
    _integer(doc, "batch_size", 1, 120)
    _integer(doc, "eval_samples", 1, 40)
    _integer(doc, "seed", 0, 2147483647)
    _integer(doc, "max_seconds_per_arm", 10, 3600)
    arms = doc.get("arms")
    if not isinstance(arms, list) or not arms or len(arms) > 6:
        raise ValueError("invalid arm count")
    if any(not isinstance(arm, str) or arm not in ARMS for arm in arms) or len(set(arms)) != len(arms):
        raise ValueError("unknown or repeated arm")
    vram = doc.get("min_vram_gb")
    if type(vram) not in (int, float) or not 1 <= vram <= 32:
        raise ValueError("invalid GPU memory minimum")
    if doc.get("requires_gpu") is not True or type(doc.get("requires_gpu")) is not bool:
        raise ValueError("remote training requires explicit CUDA")
    if steps * len(arms) > 12288:
        raise ValueError("aggregate step limit exceeded")
    return doc


def training_command(job: dict, root: Path, out: Path) -> list[str]:
    validate_request(job)
    return [sys.executable, str(root / "experiments/v37_ncell_architecture/run_ablation.py"),
            "--steps", str(job["steps"]), "--batch-size", str(job["batch_size"]),
            "--eval-samples", str(job["eval_samples"]), "--arms", *job["arms"],
            "--seed", str(job["seed"]), "--threads", "4",
            "--device", "cuda", "--max-steps-per-arm", "2048",
            "--max-seconds-per-arm", str(job["max_seconds_per_arm"]),
            "--save-candidates", "--output-dir", str(out)]


def check_checkout(root: Path, sha: str) -> None:
    r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                       capture_output=True, text=True, check=True)
    if r.stdout.strip() != sha:
        raise RuntimeError("local source differs from approved job revision")
    r = subprocess.run(["git", "diff", "--quiet", "HEAD", "--"], cwd=root)
    if r.returncode != 0:
        raise RuntimeError("local tracked source modified; training refused")


def cuda_preflight(min_gib: float) -> dict:
    if platform.system() != "Windows":
        raise RuntimeError("only Windows machines may execute this agent")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA missing; no CPU fallback")
    prop = torch.cuda.get_device_properties(0)
    gib = prop.total_memory / 1024 ** 3
    if gib < min_gib:
        raise RuntimeError(f"insufficient VRAM: {gib:.2f} GiB")
    return {"name": prop.name, "vram_gib": round(gib, 2),
            "cuda": torch.version.cuda, "torch": torch.__version__}


def save_status(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def run_job(job: dict, root: Path, output_root: Path) -> None:
    check_checkout(root, job["source_revision"])
    gpu = cuda_preflight(job["min_vram_gb"])
    out = output_root / job["request_id"]
    if out.exists():
        raise RuntimeError("job id already used; will not run again")
    out.mkdir(parents=True)
    status = out / "REMOTE_GPU_STATUS.json"
    save_status(status, {"state": "STARTING", "request_id": job["request_id"], "gpu": gpu})
    command = training_command(job, root, out)
    cancelled = timed_out = False
    with (out / "train.log").open("w", encoding="utf-8") as log:
        proc = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
        save_status(status, {"state": "RUNNING", "request_id": job["request_id"],
                             "gpu": gpu, "pid": proc.pid})
        start = time.monotonic()
        last_check = 0.0
        hard_limit = len(job["arms"]) * job["max_seconds_per_arm"] + 300
        while proc.poll() is None:
            if time.monotonic() - start > hard_limit:
                timed_out = True
                break
            if (out / "STOP").exists():
                cancelled = True
                break
            if time.monotonic() - last_check >= 180:
                last_check = time.monotonic()
                try:
                    update = validate_request(fetch_request())
                    if update and update["action"] == "cancel" and update["target_request_id"] == job["request_id"]:
                        cancelled = True
                        break
                except Exception as exc:
                    print(f"cancel poll failed: {exc}", file=sys.stderr)
            time.sleep(3)
        if cancelled or timed_out:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        rc = proc.wait()
    state = "CANCELLED" if cancelled else "TIMED_OUT" if timed_out else (
        "COMPLETED" if rc == 0 else "FAILED")
    save_status(status, {"state": state, "returncode": rc,
                         "request_id": job["request_id"], "gpu": gpu,
                         "output_dir": str(out), "stable_modified": False,
                         "candidate_uploaded": False})
    print(f"{job['request_id']}: {state}; results at {out}")


def main() -> int:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--once", action="store_true")
    cli.add_argument("--poll-seconds", type=int, default=600)
    cli.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    cli.add_argument("--output-root", type=Path, default=Path.home() / "N-MIND-remote-runs")
    args = cli.parse_args()
    if args.poll_seconds < 300:
        cli.error("poll-seconds must be >= 300")
    args.output_root.mkdir(parents=True, exist_ok=True)
    print("N-MIND GPU remote agent: pull-only, training-only; no inbound network port")
    while True:
        try:
            job = validate_request(fetch_request())
            if job and job["action"] == "train":
                run_job(job, args.project_root.resolve(), args.output_root.resolve())
        except Exception as exc:
            print(f"agent refused job: {exc}", file=sys.stderr)
        if args.once:
            return 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
