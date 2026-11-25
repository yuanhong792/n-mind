"""列出 2.7 有效臂的停止点 checkpoint 与封存 logits（供 git add -f）。"""
import glob
import json
from pathlib import Path

ARMS = Path("experiments/phase2_baselines/mqar_20261125/arms")
files = []
for p in glob.glob(str(ARMS / "p27_*.json")):
    if p.endswith(".cfg.json") or "_per_sample" in p:
        continue
    d = json.loads(Path(p).read_text(encoding="utf-8"))
    if not str(d.get("status", "")).startswith("COMPLETE"):
        continue
    ck = ARMS / f"{d['arm_id']}_step{d.get('steps_done')}.pt"
    if ck.exists():
        files.append(str(ck))
    lg = ARMS / f"{d['arm_id']}_seal_logits.pt"
    if lg.exists():
        files.append(str(lg))
Path(".git/_p27_pt.txt").write_text("\n".join(files), encoding="utf-8")
tot = sum(Path(f).stat().st_size for f in files) / 1e6
print(f"{len(files)} files, {tot:.1f} MB")
