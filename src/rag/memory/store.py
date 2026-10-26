"""记忆的 PostgreSQL + pgvector 存储。"""

from __future__ import annotations

from datetime import date
from typing import Any, Sequence

import psycopg
from pgvector import Vector
from psycopg.types.json import Jsonb

from .models import (
    MemoryItem,
    MemoryKind,
    MemoryStatus,
    fingerprint as make_fingerprint,
)

_ITEM_COLS = """
    id, namespace, subject, kind, content, structured, importance, confidence,
    status, is_speculation, fingerprint, source, session_id,
    t_valid_from, t_valid_to, created_at, updated_at, last_used_at,
    use_count, superseded_by, embedding
"""


def _vec(value: Any) -> list[float] | None:
    if value is None:
        return None
    to_list = getattr(value, "to_list", None)
    if callable(to_list):
        return [float(x) for x in to_list()]
    return [float(x) for x in value]


def to_item(row: dict[str, Any]) -> MemoryItem:
    return MemoryItem(
        id=int(row["id"]),
        namespace=row["namespace"],
        subject=row["subject"],
        kind=MemoryKind(row["kind"]),
        content=row["content"],
        structured=row.get("structured") or {},
        importance=int(row["importance"]),
        confidence=float(row["confidence"]),
        status=MemoryStatus(row["status"]),
        is_speculation=bool(row["is_speculation"]),
        fingerprint=row["fingerprint"],
        source=row.get("source"),
        session_id=row.get("session_id"),
        t_valid_from=row["t_valid_from"],
        t_valid_to=row.get("t_valid_to"),
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at"),
        last_used_at=row.get("last_used_at"),
        use_count=int(row.get("use_count") or 0),
        superseded_by=row.get("superseded_by"),
        embedding=_vec(row.get("embedding")),
    )


class MemoryStore:
    def __init__(self, conn: psycopg.Connection) -> None:
        self.conn = conn

    # ------------------------------------------------------------------
    # 写入
    # ------------------------------------------------------------------
    def insert(
        self,
        *,
        namespace: str,
        subject: str,
        kind: MemoryKind,
        content: str,
        importance: int,
        confidence: float,
        is_speculation: bool,
        source: str | None,
        session_id: str | None,
        valid_from: date,
        embedding: Sequence[float] | None,
        structured: dict[str, Any] | None = None,
        status: MemoryStatus = MemoryStatus.ACTIVE,
    ) -> int | None:
        fp = make_fingerprint(kind, subject, content)
        row = self.conn.execute(
            f"""
            INSERT INTO memories
                (namespace, subject, kind, content, structured, importance,
                 confidence, status, is_speculation, fingerprint, source,
                 session_id, t_valid_from, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (namespace, fingerprint) DO NOTHING
            RETURNING id
            """,
            (
                namespace,
                subject,
                kind.value,
                content,
                Jsonb(structured or {}),
                importance,
                confidence,
                status.value,
                is_speculation,
                fp,
                source,
                session_id,
                valid_from,
                Vector(list(embedding)) if embedding is not None else None,
            ),
        ).fetchone()
        return int(row["id"]) if row else None

    def supersede(
        self,
        old_id: int,
        new_id: int,
        *,
        valid_from: date | None = None,
    ) -> bool:
        row = self.conn.execute(
            """
            UPDATE memories
               SET status = 'superseded',
                   superseded_by = %s,
                   updated_at = now(),
                   t_valid_to = CASE
                       WHEN %s::date IS NOT NULL AND %s::date > t_valid_from
                       THEN %s::date
                       ELSE t_valid_to
                   END
             WHERE id = %s AND status = 'active'
            RETURNING id
            """,
            (new_id, valid_from, valid_from, valid_from, old_id),
        ).fetchone()
        return row is not None

    def archive(self, memory_id: int) -> bool:
        row = self.conn.execute(
            """
            UPDATE memories
               SET status = 'archived', embedding = NULL, updated_at = now()
             WHERE id = %s AND status IN ('active', 'superseded')
            RETURNING id
            """,
            (memory_id,),
        ).fetchone()
        return row is not None

    def touch(self, ids: Sequence[int]) -> int:
        if not ids:
            return 0
        row = self.conn.execute(
            """
            UPDATE memories
               SET use_count = use_count + 1,
                   last_used_at = now()
             WHERE id = ANY(%s)
            RETURNING id
            """,
            (list(ids),),
        ).fetchall()
        return len(row)

    def audit(
        self,
        memory_id: int | None,
        action: str,
        detail: dict[str, Any] | None = None,
    ) -> None:
        self.conn.execute(
            """
            INSERT INTO memory_events (memory_id, action, detail)
            VALUES (%s, %s, %s)
            """,
            (memory_id, action, Jsonb(detail or {})),
        )

    # ------------------------------------------------------------------
    # 读取
    # ------------------------------------------------------------------
    def get(self, memory_id: int) -> MemoryItem | None:
        row = self.conn.execute(
            f"SELECT {_ITEM_COLS} FROM memories WHERE id = %s",
            (memory_id,),
        ).fetchone()
        return to_item(row) if row else None

    def list(
        self,
        *,
        namespace: str = "default",
        subject: str | None = None,
        status: MemoryStatus | None = MemoryStatus.ACTIVE,
        kind: MemoryKind | None = None,
        limit: int = 20,
    ) -> list[MemoryItem]:
        clauses = ["namespace = %s"]
        params: list[Any] = [namespace]
        if subject:
            clauses.append("subject = %s")
            params.append(subject)
        if status:
            clauses.append("status = %s")
            params.append(status.value)
        if kind:
            clauses.append("kind = %s")
            params.append(kind.value)
        params.append(limit)
        rows = self.conn.execute(
            f"""
            SELECT {_ITEM_COLS} FROM memories
            WHERE {' AND '.join(clauses)}
            ORDER BY importance DESC, updated_at DESC
            LIMIT %s
            """,
            params,
        ).fetchall()
        return [to_item(r) for r in rows]

    def neighbors(
        self,
        *,
        namespace: str,
        subject: str,
        limit: int = 50,
    ) -> list[MemoryItem]:
        rows = self.conn.execute(
            f"""
            SELECT {_ITEM_COLS} FROM memories
            WHERE namespace = %s
              AND subject = %s
              AND status = 'active'
              AND t_valid_from <= CURRENT_DATE
              AND (t_valid_to IS NULL OR CURRENT_DATE < t_valid_to)
              AND embedding IS NOT NULL
            ORDER BY updated_at DESC
            LIMIT %s
            """,
            (namespace, subject, limit),
        ).fetchall()
        return [to_item(r) for r in rows]

    def search_vector(
        self,
        *,
        namespace: str,
        query_vector: Sequence[float],
        limit: int = 20,
        subject: str | None = None,
        kind: MemoryKind | None = None,
        exact: bool = True,
    ) -> list[tuple[MemoryItem, float]]:
        clauses = [
            "namespace = %s",
            "status = 'active'",
            "t_valid_from <= CURRENT_DATE",
            "(t_valid_to IS NULL OR CURRENT_DATE < t_valid_to)",
            "embedding IS NOT NULL",
        ]
        params: list[Any] = [namespace]
        if subject:
            clauses.append("subject = %s")
            params.append(subject)
        if kind:
            clauses.append("kind = %s")
            params.append(kind.value)
        if exact:
            # 记忆数量很小，精确扫描结果更稳定。
            self.conn.execute("SET LOCAL enable_indexscan = off")
            self.conn.execute("SET LOCAL enable_bitmapscan = off")
        params = [Vector(list(query_vector))] + params + [
            Vector(list(query_vector)),
            limit,
        ]
        rows = self.conn.execute(
            f"""
            SELECT {_ITEM_COLS},
                   1 - (embedding <=> %s) AS relevance
            FROM memories
            WHERE {' AND '.join(clauses)}
            ORDER BY embedding <=> %s
            LIMIT %s
            """,
            params,
        ).fetchall()
        return [(to_item(r), float(r["relevance"])) for r in rows]

    def recent(
        self,
        *,
        namespace: str,
        limit: int = 20,
        subject: str | None = None,
    ) -> list[MemoryItem]:
        clauses = [
            "namespace = %s",
            "status = 'active'",
            "t_valid_from <= CURRENT_DATE",
            "(t_valid_to IS NULL OR CURRENT_DATE < t_valid_to)",
        ]
        params: list[Any] = [namespace]
        if subject:
            clauses.append("subject = %s")
            params.append(subject)
        params.append(limit)
        rows = self.conn.execute(
            f"""
            SELECT {_ITEM_COLS} FROM memories
            WHERE {' AND '.join(clauses)}
            ORDER BY updated_at DESC
            LIMIT %s
            """,
            params,
        ).fetchall()
        return [to_item(r) for r in rows]

    def stats(self, namespace: str | None = None) -> dict[str, int]:
        clauses = []
        params: list[Any] = []
        if namespace:
            clauses.append("namespace = %s")
            params.append(namespace)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        row = self.conn.execute(
            f"""
            SELECT
                count(*) AS total,
                count(*) FILTER (WHERE status = 'active') AS active,
                count(*) FILTER (WHERE status = 'superseded') AS superseded,
                count(*) FILTER (WHERE status = 'archived') AS archived
            FROM memories {where}
            """,
            params,
        ).fetchone()
        return {k: int(v or 0) for k, v in row.items()}

    def events(
        self,
        limit: int = 20,
        *,
        namespace: str | None = None,
    ) -> list[dict[str, Any]]:
        if namespace:
            rows = self.conn.execute(
                """
                SELECT e.id, e.memory_id, e.action, e.detail, e.created_at
                FROM memory_events e
                JOIN memories m ON m.id = e.memory_id
                WHERE m.namespace = %s
                ORDER BY e.id DESC
                LIMIT %s
                """,
                (namespace, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                """
                SELECT id, memory_id, action, detail, created_at
                FROM memory_events
                ORDER BY id DESC
                LIMIT %s
                """,
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]
