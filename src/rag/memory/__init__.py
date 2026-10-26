"""小型长期记忆系统。

写入：候选抽取 → 去重/更新判断 → PostgreSQL + pgvector
召回：向量 + 重要度 + 新鲜度 + 置信度 → 预算打包
"""

from .models import MemoryCandidate, MemoryItem, MemoryKind, MemoryStatus
from .recall import format_memories, recall_memories
from .service import MemoryService

__all__ = [
    "MemoryCandidate",
    "MemoryItem",
    "MemoryKind",
    "MemoryStatus",
    "MemoryService",
    "recall_memories",
    "format_memories",
]
