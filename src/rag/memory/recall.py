"""记忆召回、打分和上下文打包。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Sequence

from ..db.engine import connect
from .models import KIND_ZH, MemoryItem, MemoryKind, MemoryStatus, estimate_tokens
from .store import MemoryStore

_DECAY_DAYS: dict[MemoryKind, float] = {
    MemoryKind.PREFERENCE: 180.0,
    MemoryKind.FACT: 90.0,
    MemoryKind.CORRECTION: 90.0,
    MemoryKind.TASK: 60.0,
    MemoryKind.EVENT: 14.0,
}


@dataclass(frozen=True)
class RecalledMemory:
    item: MemoryItem
    score: float
    relevance: float


def recency(item: MemoryItem, today: date) -> float:
    days = max(0, (today - item.t_valid_from).days)
    tau = _DECAY_DAYS.get(item.kind, 90.0)
    return math.exp(-days / tau)


def score_item(item: MemoryItem, relevance: float, today: date) -> float:
    return (
        0.55 * max(0.0, min(1.0, relevance))
        + 0.20 * (item.importance / 10.0)
        + 0.15 * recency(item, today)
        + 0.10 * max(0.0, min(1.0, item.confidence))
    )


def pack(
    recalled: Sequence[RecalledMemory],
    *,
    budget_chars: int = 800,
) -> list[RecalledMemory]:
    """按分数贪心装入，避免重复内容。"""
    out: list[RecalledMemory] = []
    seen: set[str] = set()
    used = 0
    for row in sorted(recalled, key=lambda r: r.score, reverse=True):
        key = row.item.content.strip()
        if key in seen:
            continue
        size = len(key)
        if out and used + size > budget_chars:
            continue
        out.append(row)
        seen.add(key)
        used += size
    return out


def recall_memories(
    query: str,
    *,
    namespace: str = "default",
    subject: str | None = "user:self",
    top_k: int = 5,
    budget_chars: int = 800,
    store: MemoryStore | None = None,
) -> list[RecalledMemory]:
    """按语义召回记忆；向量不可用时退回最近记忆。"""
    if store is None:
        with connect() as conn:
            return recall_memories(
                query,
                namespace=namespace,
                subject=subject,
                top_k=top_k,
                budget_chars=budget_chars,
                store=MemoryStore(conn),
            )

    today = date.today()
    hits: list[tuple[MemoryItem, float]] = []
    query_vector = None
    try:
        from ..indexing.embedder import LocalEmbedder

        query_vector = LocalEmbedder.get().embed_query(query)
    except Exception:
        query_vector = None

    if query_vector is not None:
        try:
            hits = store.search_vector(
                namespace=namespace,
                query_vector=query_vector,
                limit=max(top_k * 4, 20),
                subject=subject,
            )
        except Exception:
            hits = []

    if not hits:
        recent = store.recent(
            namespace=namespace,
            limit=max(top_k * 4, 20),
            subject=subject,
        )
        hits = [(item, 0.0) for item in recent]

    scored = [
        RecalledMemory(
            item=item,
            relevance=relevance,
            score=score_item(item, relevance, today),
        )
        for item, relevance in hits
    ]
    selected = pack(scored, budget_chars=budget_chars)[:top_k]
    store.touch([r.item.id for r in selected])
    return selected


def format_memories(
    recalled: Sequence[RecalledMemory],
    *,
    show_score: bool = False,
) -> str:
    if not recalled:
        return ""
    lines = ["【长期记忆】"]
    for row in recalled:
        item = row.item
        prefix = f"[{KIND_ZH.get(item.kind, item.kind.value)}]"
        suffix = f"（重要度 {item.importance}，置信度 {item.confidence:.2f}）"
        if show_score:
            suffix += f"（召回分 {row.score:.2f}）"
        lines.append(f"- {prefix} {item.content}{suffix}")
    return "\n".join(lines)


def estimate_memory_tokens(recalled: Sequence[RecalledMemory]) -> int:
    return sum(estimate_tokens(r.item.content) for r in recalled)
