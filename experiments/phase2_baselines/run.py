"""Phase 2 runner scaffold. --dry-run is the only permitted mode in this phase."""
from __future__ import annotations
import argparse, json
from pathlib import Path
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--dry-run',action='store_true'); a=ap.parse_args()
    if not a.dry_run: raise SystemExit('formal training is disabled until explicit budget approval')
    p=Path(__file__).with_name('preregister.json'); print(json.dumps(json.loads(p.read_text(encoding='utf8')),ensure_ascii=False,indent=2))
if __name__=='__main__': main()
