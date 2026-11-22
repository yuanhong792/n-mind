"""Training gate and diagnostics; formal training is intentionally unavailable by default."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
from .neural_language_v2 import LanguageV2Model

class TrainingPermissionError(PermissionError): pass

def require_training_authorization(*, authorized: bool, max_steps: int) -> None:
    if not authorized or max_steps <= 0: raise TrainingPermissionError("formal training requires explicit positive budget authorization")

def zero_step_diagnostic(model: LanguageV2Model) -> dict:
    state = {k: v.detach().cpu().numpy().tobytes() for k, v in model.state_dict().items()}
    digest = hashlib.sha256(b"".join(state.values())).hexdigest()
    return {"status": "NOT_RUN", "optimizer_step_count": 0, "parameter_sha256_before_after": [digest, digest], "training_authorized": False}
