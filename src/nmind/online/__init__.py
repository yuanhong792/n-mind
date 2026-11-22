"""N-MIND 在线经验通路：只读抓取 → 结构化提取 → 程序化验证 → 经验流 → 调度。

M4 任务书第 3/4 节实现。安全口径：
  * 默认 offline，真实 HTTP 默认关且需 allowlist；
  * 网页内容永远是低信任数据——命令式文本只被标记隔离，绝不执行、不改配置；
  * 只有 verified 经验能进入学习流；sealed_eval 永远进不了 replay/训练流；
  * candidate 权重独立存放，失败回滚，stable 不被原地覆盖。
"""

from .experiences import Experience, build_learning_stream, load_jsonl, now_iso, save_jsonl
from .explorer import pick_task_family
from .extractor import detect_instructions, extract_rule_table, extract_text, strip_scripts
from .fetcher import FetchConfig, Fetcher, FetchRecord, OfflineModeError
from .fixtures import (MALICIOUS_URL, RULE_MIRROR_URL, RULE_PAGE_URL, SIM_WEB,
                       UNRELATED_URL, fixture_fetcher, fixture_transport)
from .promotion import (append_audit, check_gates, decide_status, file_sha256,
                        rollback_candidate)
from .provenance import Provenance, build_provenance
from .scheduler import SchedulerConfig, decide
from .verifier import build_rule_oracle, verify_rule

__all__ = [
    "Experience", "build_learning_stream", "load_jsonl", "now_iso", "save_jsonl",
    "pick_task_family", "detect_instructions", "extract_rule_table", "extract_text",
    "strip_scripts", "FetchConfig", "Fetcher", "FetchRecord", "OfflineModeError",
    "MALICIOUS_URL", "RULE_MIRROR_URL", "RULE_PAGE_URL", "SIM_WEB", "UNRELATED_URL",
    "fixture_fetcher", "fixture_transport", "append_audit", "check_gates",
    "decide_status", "file_sha256", "rollback_candidate", "Provenance",
    "build_provenance", "SchedulerConfig", "decide", "build_rule_oracle", "verify_rule",
]
