"""四个预登记候选架构配置。参数量必须在实现后实测，不用无效参数填充。"""
from __future__ import annotations

CANDIDATES = {
    "M0": {
        "model_type": "ncell_reference",
        "role": "historical_reference",
        "config": {"d_model": 64, "n_blocks": 4, "block_hidden": 64,
                   "workspace_dim": 32, "workspace_slots": 8, "top_k": 2},
        "trainable_params_target": 106410,
        "gate": "n/a", "conv_kernel": 0, "uncertainty": False,
    },
    "M1-A": {
        "model_type": "gru_reference",
        "role": "plain_control",
        "config": {"d_model": 64, "hidden": 128, "layers": 1},
        "trainable_params_target": None,
        "gate": "n/a", "conv_kernel": 0, "uncertainty": False,
    },
    "M1-B": {
        "model_type": "delta_memory",
        "role": "new_architecture_shared_gate",
        "config": {"d_model": 64, "hidden": 96, "dk": 48, "dv": 48,
                   "gate": "shared", "conv_kernel": 1, "uncertainty": False},
        "trainable_params_target": None,
        "gate": "shared", "conv_kernel": 1, "uncertainty": False,
    },
    "M1-C": {
        "model_type": "decoupled_memory",
        "role": "new_architecture_independent_gates",
        "config": {"d_model": 64, "hidden": 96, "dk": 48, "dv": 48,
                   "gate": "independent", "conv_kernel": 1, "uncertainty": False},
        "trainable_params_target": None,
        "gate": "independent", "conv_kernel": 1, "uncertainty": False,
    },
}


def build_model(name: str, vocab_size: int):
    spec = CANDIDATES[name]
    mt = spec["model_type"]
    if mt == "ncell_reference":
        from models.ncell_reference import NCellReference
        return NCellReference(vocab_size, **spec["config"])
    if mt == "gru_reference":
        from models.gru_reference import GRUReferenceM1
        return GRUReferenceM1(vocab_size, **spec["config"])
    if mt == "delta_memory":
        from models.delta_memory import DeltaMemoryLM
        return DeltaMemoryLM(vocab_size, **spec["config"])
    if mt == "decoupled_memory":
        from models.decoupled_memory import DecoupledMemoryLM
        return DecoupledMemoryLM(vocab_size, **spec["config"])
    raise ValueError(f"未知模型类型: {mt}")


def candidate_names():
    return list(CANDIDATES)
