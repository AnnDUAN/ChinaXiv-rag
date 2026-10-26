"""写入前的简单闸门：精确去重、重要度过滤、近邻更新判断。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from .models import MemoryCandidate, MemoryItem, MemoryKind, fingerprint


class GateDecision(str, Enum):
    DUPLICATE = "duplicate"
    UPDATE = "update"
    NEW = "new"
    REJECT = "reject"


@dataclass(frozen=True)
class GateConfig:
    min_importance: int = 5
    duplicate_cosine: float = 0.93
    update_cosine: float = 0.82
    speculation_confidence: float = 0.6

    def __post_init__(self) -> None:
        if not 1 <= self.min_importance <= 10:
            raise ValueError("min_importance 必须在 1-10")
        if not 0 <= self.duplicate_cosine <= 1:
            raise ValueError("duplicate_cosine 必须在 0-1")
        if not 0 <= self.update_cosine <= self.duplicate_cosine:
            raise ValueError("update_cosine 必须 <= duplicate_cosine")
        if not 0 <= self.speculation_confidence <= 0.6:
            raise ValueError("speculation_confidence 必须在 0-0.6")


@dataclass(frozen=True)
class GateResult:
    decision: GateDecision
    neighbor: MemoryItem | None = None
    similarity: float = 0.0
    reason: str = ""


def cosine(a: Sequence[float] | None, b: Sequence[float] | None) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def decide(
    candidate: MemoryCandidate,
    neighbors: Sequence[MemoryItem],
    *,
    config: GateConfig | None = None,
) -> GateResult:
    cfg = config or GateConfig()
    if candidate.importance < cfg.min_importance:
        return GateResult(
            GateDecision.REJECT,
            reason=f"重要度 {candidate.importance} 低于阈值 {cfg.min_importance}",
        )

    fp = fingerprint(candidate.kind, candidate.subject, candidate.content)
    for item in neighbors:
        if item.fingerprint == fp:
            return GateResult(
                GateDecision.DUPLICATE,
                neighbor=item,
                similarity=1.0,
                reason="内容指纹完全相同",
            )

    if not neighbors:
        return GateResult(GateDecision.NEW, reason="没有相近的已有记忆")

    best = max(
        neighbors,
        key=lambda item: cosine(candidate.embedding, item.embedding)
        if candidate.embedding and item.embedding
        else 0.0,
    )
    similarity = cosine(candidate.embedding, best.embedding)

    if similarity >= cfg.duplicate_cosine:
        return GateResult(
            GateDecision.DUPLICATE,
            neighbor=best,
            similarity=similarity,
            reason=f"相似度 {similarity:.2f}，视为重复",
        )

    if candidate.kind is MemoryKind.CORRECTION:
        return GateResult(
            GateDecision.UPDATE,
            neighbor=best,
            similarity=similarity,
            reason="用户纠正，覆盖最近的相关记忆",
        )

    if best.kind is candidate.kind and similarity >= cfg.update_cosine:
        return GateResult(
            GateDecision.UPDATE,
            neighbor=best,
            similarity=similarity,
            reason=f"同类别且相似度 {similarity:.2f}，视为更新",
        )

    return GateResult(
        GateDecision.NEW,
        neighbor=best,
        similarity=similarity,
        reason="可以同时存在",
    )
