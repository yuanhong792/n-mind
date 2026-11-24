"""Pre-training task validity gate; thresholds are fixed, not tuned per task."""
from __future__ import annotations

def task_validity_check(task: dict) -> dict:
    random_level=float(task.get('random_level', 0.0)); shortcuts=task.get('shortcut_ems', {})
    checks={
      'oracle_em': float(task.get('oracle_em', 0.0)) == 1.0,
      'shortcuts_le_random_plus_005': all(float(v) <= random_level + 0.05 for v in shortcuts.values()),
      'effective_samples_ge_500': int(task.get('effective_samples', 0)) >= 500,
      'combination_overlap_zero': int(task.get('combination_overlap', 1)) == 0,
      'counterfactual_pairs_present': bool(task.get('counterfactual_pairs_present', False)),
      'unk_rate_zero': float(task.get('unk_rate', 1.0)) == 0.0,
    }
    return {'task':task.get('name','unnamed'),'checks':checks,'valid':all(checks.values()),
            'failed': [k for k,v in checks.items() if not v]}
