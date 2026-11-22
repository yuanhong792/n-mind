"""M5-Language：中文受限领域神经语言交互（自回归自由生成，权重全部随机初始化）。"""

from .decoder import GenerationRecord, generate
from .dialogue_data import (Dialogue, SPLITS, Turn, build_all, build_dialogue, build_split,
                             dialogue_to_dict, majority_answers)
from .neural_dialogue import LMConfig, LMState, NMindLM, load_lm, save_lm
from .session import ChatReply, ChatSession
from .tokenizer import CharTokenizer

__all__ = ["GenerationRecord", "generate", "Dialogue", "SPLITS", "Turn", "build_all",
            "build_dialogue", "build_split", "dialogue_to_dict", "majority_answers",
            "LMConfig", "LMState", "NMindLM", "load_lm", "save_lm",
            "ChatReply", "ChatSession", "CharTokenizer"]
