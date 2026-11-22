"""Web 工作台组装：加载本地神经模型 + 知识库 + 探索器（与 Tkinter 模式同栈）。

浏览器端不直接触碰模型——一切推理/检索/抓取经 Python 后端完成。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .chat_service import ChatService
from .idle_explorer import IdleWebExplorer
from .knowledge_search import KnowledgeSearch
from .source_registry import SourceRegistry


def build_web_stack(policy, offline: bool, registry: SourceRegistry,
                    knowledge: KnowledgeSearch, root_dir: Path):
    """返回 (ChatService, IdleWebExplorer)。

    stable 权重解析与 CLI run / Tkinter 完全相同：bot 根目录副本优先，
    词表哈希校验拒绝加载不兼容 checkpoint；缺失时模型侧如实标缺失。
    """
    from ..language.neural_dialogue import load_lm
    from ..language.tokenizer import CharTokenizer
    from ..online.promotion import file_sha256

    from ..memory import MemoryService, MemoryStore
    memory_service = MemoryService(MemoryStore(root_dir / "memory.db"))
    chat_service = ChatService(knowledge=knowledge, memory_service=memory_service)
    stable = root_dir / "stable_model.pt"
    tok_path = root_dir / "tokenizer.json"
    if stable.exists() and tok_path.exists():
        try:
            tokenizer = CharTokenizer.load(tok_path)
            model, _meta = load_lm(stable, tokenizer)
            chat_service = ChatService(model=model, tokenizer=tokenizer,
                                       model_sha=file_sha256(stable),
                                       knowledge=knowledge, memory_service=memory_service)
        except Exception as exc:                  # noqa: BLE001 - 权重损坏如实报缺
            print(f"[web] 本地语言权重加载失败（神经侧将标注不可用）：{exc}")
    explorer = IdleWebExplorer(registry=registry, knowledge_store=knowledge.store,
                               policy=policy, offline=offline)
    return chat_service, explorer


def load_stable_pair(args) -> tuple[Path | None, Path | None]:
    """与 CLI 的 stable 权重解析同口径（供 Web 服务单独调用时复用）。"""
    from ..cli import _root, _stable_pair
    return _stable_pair(args, _root(args))


def stack_summary(chat_service: ChatService, explorer: IdleWebExplorer,
                  offline: bool) -> dict[str, Any]:
    return {"model_loaded": chat_service.model_loaded,
            "model_sha": chat_service.model_sha,
            "offline": offline,
            "explorer_enabled": explorer.enabled,
            "allowed_hosts": explorer.registry.enabled_hosts()}
