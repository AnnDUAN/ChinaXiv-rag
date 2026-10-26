"""记忆的数据结构、枚举和指纹。"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any


class MemoryKind(str, Enum):
    FACT = "fact"
    PREFERENCE = "preference"
    EVENT = "event"
    TASK = "task"
    CORRECTION = "correction"


class MemoryStatus(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    ARCHIVED = "archived"
    REJECTED = "rejected"


KIND_ZH: dict[MemoryKind, str] = {
    MemoryKind.FACT: "事实",
    MemoryKind.PREFERENCE: "偏好",
    MemoryKind.EVENT: "事件",
    MemoryKind.TASK: "任务",
    MemoryKind.CORRECTION: "纠正",
}


@dataclass
class MemoryCandidate:
    """还没写入数据库的记忆候选。"""

    kind: MemoryKind
    content: str
    subject: str = "user:self"
    importance: int = 5
    confidence: float = 0.7
    is_speculation: bool = False
    valid_from: date | None = None
    source: str = "manual"
    session_id: str | None = None
    structured: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None


@dataclass
class MemoryItem:
    """数据库里的一条记忆。"""

    id: int
    namespace: str
    subject: str
    kind: MemoryKind
    content: str
    importance: int
    confidence: float
    status: MemoryStatus
    is_speculation: bool
    fingerprint: str
    t_valid_from: date
    t_valid_to: date | None = None
    source: str | None = None
    session_id: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    last_used_at: datetime | None = None
    use_count: int = 0
    superseded_by: int | None = None
    structured: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "namespace": self.namespace,
            "subject": self.subject,
            "kind": self.kind.value,
            "kind_zh": KIND_ZH.get(self.kind, self.kind.value),
            "content": self.content,
            "importance": self.importance,
            "confidence": round(self.confidence, 3),
            "status": self.status.value,
            "is_speculation": self.is_speculation,
            "source": self.source,
            "session_id": self.session_id,
            "t_valid_from": self.t_valid_from.isoformat(),
            "t_valid_to": self.t_valid_to.isoformat() if self.t_valid_to else None,
            "use_count": self.use_count,
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else None,
            "superseded_by": self.superseded_by,
        }


_WS = re.compile(r"\s+")


def normalize_content(text: str) -> str:
    return _WS.sub(" ", text or "").strip()


def fingerprint(kind: MemoryKind, subject: str, content: str) -> str:
    raw = f"{kind.value}|{subject}|{normalize_content(content)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def estimate_tokens(text: str) -> int:
    """粗略估算中文 token 数。"""
    if not text:
        return 0
    return max(1, math.ceil(len(text) * 0.9))
