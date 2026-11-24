from __future__ import annotations
from pathlib import Path
import json

REQUIRED = ('checkpoint', 'sample_logits', 'metadata', 'timing_only')
def validate_arm_artifact(payload: dict) -> None:
    missing=[k for k in REQUIRED if k not in payload]
    if missing: raise ValueError('missing arm artifacts: '+','.join(missing))
    if not payload['timing_only'] and (not payload['checkpoint'] or not payload['sample_logits']):
        raise ValueError('non-timing arm must save checkpoint and sample logits')
def write_arm_manifest(path: str|Path, payload: dict):
    validate_arm_artifact(payload); Path(path).write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf8')
