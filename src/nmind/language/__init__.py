"""N-Language：受控模板映射（默认）与可选本地预训练适配器。"""

from .interface import capability_note, describe_event, parse_rule, verbalize_decision
from .local_adapter import AdapterStatus, LocalTextAdapter

__all__ = ["capability_note", "describe_event", "parse_rule", "verbalize_decision",
           "LocalTextAdapter", "AdapterStatus"]
