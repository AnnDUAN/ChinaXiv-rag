"""长期记忆的对外服务层。"""

from __future__ import annotations

from datetime import date
from typing import Any, Sequence

from ..db.engine import connect
from ..llm import LLMClient
from .extract import extract_candidates
from .gate import GateConfig, GateDecision, decide
from .models import MemoryCandidate, MemoryKind, MemoryStatus
from .recall import RecalledMemory, format_memories, recall_memories
from .store import MemoryStore


class MemoryService:
    """一个小型长期记忆服务。

    namespace 用来隔离不同用户/项目；subject 默认是 user:self。
    """

    def __init__(
        self,
        *,
        namespace: str = "default",
        subject: str = "user:self",
        gate_config: GateConfig | None = None,
    ) -> None:
        self.namespace = namespace
        self.subject = subject
        self.gate_config = gate_config or GateConfig()

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------
    def add(
        self,
        content: str,
        *,
        kind: MemoryKind = MemoryKind.FACT,
        importance: int = 5,
        confidence: float = 0.7,
        is_speculation: bool = False,
        source: str = "manual",
        session_id: str | None = None,
        valid_from: date | None = None,
    ) -> dict[str, Any]:
        candidate = MemoryCandidate(
            kind=kind,
            content=content,
            subject=self.subject,
            importance=importance,
            confidence=confidence,
            is_speculation=is_speculation,
            valid_from=valid_from or date.today(),
            source=source,
            session_id=session_id,
        )
        return self.remember(candidate)

    def remember(self, candidate: MemoryCandidate) -> dict[str, Any]:
        if not candidate.content.strip():
            raise ValueError("记忆内容不能为空")
        candidate.subject = self.subject
        candidate.valid_from = candidate.valid_from or date.today()
        if candidate.is_speculation:
            candidate.confidence = min(candidate.confidence, 0.6)
        candidate.embedding = self._embed(candidate.content)

        with connect() as conn:
            store = MemoryStore(conn)
            neighbors = store.neighbors(
                namespace=self.namespace,
                subject=self.subject,
            )
            result = decide(
                candidate, neighbors, config=self.gate_config
            )

            if result.decision is GateDecision.REJECT:
                store.audit(
                    None,
                    "reject",
                    {"content": candidate.content, "reason": result.reason},
                )
                return {
                    "action": "reject",
                    "memory_id": None,
                    "reason": result.reason,
                }

            if result.decision is GateDecision.DUPLICATE:
                old = result.neighbor
                if old is not None:
                    store.touch([old.id])
                    store.audit(
                        old.id,
                        "duplicate",
                        {"content": candidate.content},
                    )
                return {
                    "action": "duplicate",
                    "memory_id": old.id if old else None,
                    "reason": result.reason,
                }

            new_id = store.insert(
                namespace=self.namespace,
                subject=self.subject,
                kind=candidate.kind,
                content=candidate.content,
                importance=candidate.importance,
                confidence=candidate.confidence,
                is_speculation=candidate.is_speculation,
                source=candidate.source,
                session_id=candidate.session_id,
                valid_from=candidate.valid_from or date.today(),
                embedding=candidate.embedding,
                structured=candidate.structured,
            )
            if new_id is None:
                return {
                    "action": "duplicate",
                    "memory_id": None,
                    "reason": "数据库唯一约束判定为重复",
                }

            if (
                result.decision is GateDecision.UPDATE
                and result.neighbor is not None
            ):
                store.supersede(
                    result.neighbor.id,
                    new_id,
                    valid_from=candidate.valid_from,
                )
                store.audit(
                    new_id,
                    "update",
                    {
                        "content": candidate.content,
                        "supersedes": result.neighbor.id,
                        "similarity": round(result.similarity, 3),
                    },
                )
                action = "update"
            else:
                store.audit(
                    new_id,
                    "create",
                    {"content": candidate.content, "kind": candidate.kind.value},
                )
                action = "create"

            return {
                "action": action,
                "memory_id": new_id,
                "reason": result.reason,
            }

    def extract(
        self,
        text: str,
        *,
        source: str = "conversation",
        session_id: str | None = None,
        client: LLMClient | None = None,
        dry_run: bool = False,
    ) -> list[dict[str, Any]]:
        """从对话文本抽取候选并写入。"""
        candidates = extract_candidates(
            text,
            subject=self.subject,
            source=source,
            session_id=session_id,
            client=client,
        )
        if dry_run:
            return [
                {
                    "kind": c.kind.value,
                    "content": c.content,
                    "importance": c.importance,
                    "confidence": c.confidence,
                    "is_speculation": c.is_speculation,
                    # dry-run 只展示抽取结果，不执行去重/更新判断。
                    "decision": "candidate",
                }
                for c in candidates
            ]
        return [self.remember(c) for c in candidates]

    # ------------------------------------------------------------------
    # 读取
    # ------------------------------------------------------------------
    def recall(
        self,
        query: str,
        *,
        top_k: int = 5,
        budget_chars: int = 800,
    ) -> list[RecalledMemory]:
        with connect() as conn:
            return recall_memories(
                query,
                namespace=self.namespace,
                subject=self.subject,
                top_k=top_k,
                budget_chars=budget_chars,
                store=MemoryStore(conn),
            )

    def format_for_prompt(
        self,
        query: str,
        *,
        top_k: int = 5,
        budget_chars: int = 800,
    ) -> str:
        return format_memories(
            self.recall(query, top_k=top_k, budget_chars=budget_chars)
        )

    def list(
        self,
        *,
        limit: int = 20,
        status: MemoryStatus | None = MemoryStatus.ACTIVE,
        kind: MemoryKind | None = None,
    ) -> list[Any]:
        with connect() as conn:
            return MemoryStore(conn).list(
                namespace=self.namespace,
                subject=self.subject,
                status=status,
                kind=kind,
                limit=limit,
            )

    def forget(self, memory_id: int) -> bool:
        with connect() as conn:
            store = MemoryStore(conn)
            item = store.get(memory_id)
            if item is None or item.namespace != self.namespace:
                return False
            ok = store.archive(memory_id)
            if ok:
                store.audit(memory_id, "archive", {"reason": "manual forget"})
            return ok

    def stats(self) -> dict[str, int]:
        with connect() as conn:
            return MemoryStore(conn).stats(self.namespace)

    def events(self, limit: int = 20) -> list[dict[str, Any]]:
        with connect() as conn:
            return MemoryStore(conn).events(
                limit, namespace=self.namespace
            )

    # ------------------------------------------------------------------
    @staticmethod
    def _embed(content: str) -> list[float] | None:
        try:
            from ..indexing.embedder import LocalEmbedder

            vector = LocalEmbedder.get().embed_passages([content])[0]
            return [float(x) for x in vector]
        except Exception:
            return None
