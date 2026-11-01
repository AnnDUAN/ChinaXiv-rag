"""Agent 运行时：问题分析、批量检索、证据判断、可选联网和回答。

先分析意图并拆查询，再批量检索本地论文；证据不够时最多联网一次，
最后基于本地和网页证据回答。网页正文只放在本次运行的内存里。
"""

from .analyze import (
    INTENT_LABELS,
    QueryAnalysis,
    analyze_question,
    clean_analysis,
    infer_intent_hint,
)
from .assess import EvidenceAssessment, assess_evidence
from .checkpoint import CheckpointStore, PostgresCheckpointStore
from .runner import AgentError, AgentRunner
from .state import AgentState, AgentStatus

__all__ = [
    "AgentError",
    "AgentRunner",
    "AgentState",
    "AgentStatus",
    "CheckpointStore",
    "EvidenceAssessment",
    "INTENT_LABELS",
    "PostgresCheckpointStore",
    "QueryAnalysis",
    "analyze_question",
    "assess_evidence",
    "clean_analysis",
    "infer_intent_hint",
]
