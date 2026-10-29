"""SQL 访问层。

业务代码不直接写 SQL，统一从这里进出。
"""

from __future__ import annotations

from typing import Any, Iterable

import psycopg
from pgvector import Vector
from psycopg.types.json import Jsonb

from .engine import connect


def _json(value: Any) -> Jsonb:
    return Jsonb(value if value is not None else [])


def upsert_paper(conn: psycopg.Connection, paper: dict[str, Any]) -> None:
    conn.execute(
        """
        INSERT INTO papers
            (paper_id, title, authors, affiliations, abstract, keywords,
             doi, journal, year, source, source_url, pdf_path, status,
             quality, meta, updated_at)
        VALUES
            (%(paper_id)s, %(title)s, %(authors)s, %(affiliations)s,
             %(abstract)s, %(keywords)s, %(doi)s, %(journal)s, %(year)s,
             %(source)s, %(source_url)s, %(pdf_path)s, %(status)s,
             %(quality)s, %(meta)s, now())
        ON CONFLICT (paper_id) DO UPDATE SET
            title = EXCLUDED.title,
            authors = EXCLUDED.authors,
            affiliations = EXCLUDED.affiliations,
            abstract = EXCLUDED.abstract,
            keywords = EXCLUDED.keywords,
            doi = EXCLUDED.doi,
            journal = EXCLUDED.journal,
            year = EXCLUDED.year,
            source = EXCLUDED.source,
            source_url = EXCLUDED.source_url,
            pdf_path = EXCLUDED.pdf_path,
            status = EXCLUDED.status,
            quality = EXCLUDED.quality,
            meta = EXCLUDED.meta,
            updated_at = now()
        """,
        {
            **paper,
            "authors": _json(paper.get("authors")),
            "affiliations": _json(paper.get("affiliations")),
            "keywords": _json(paper.get("keywords")),
            "quality": _json(paper.get("quality") or {}),
            "meta": _json(paper.get("meta") or {}),
        },
    )


def replace_sections(
    conn: psycopg.Connection,
    paper_id: str,
    sections: list[dict[str, Any]],
) -> dict[tuple[str, ...], int]:
    """删除论文旧章节，写入新章节，返回 section_path → section_id。"""
    conn.execute("DELETE FROM sections WHERE paper_id = %s", (paper_id,))
    mapping: dict[tuple[str, ...], int] = {}
    for index, section in enumerate(sections, 1):
        path = section.get("section_path") or [
            section.get("title", "")
        ]
        row = conn.execute(
            """
            INSERT INTO sections
                (paper_id, section_index, section_path, title, level,
                 page_start, page_end, section_type)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING section_id
            """,
            (
                paper_id,
                index,
                Jsonb(path),
                section.get("title"),
                section.get("level"),
                section.get("page_start"),
                section.get("page_end"),
                section.get("type"),
            ),
        ).fetchone()
        mapping.setdefault(tuple(path), row["section_id"])
    return mapping


def replace_chunks(
    conn: psycopg.Connection,
    paper_id: str,
    chunks: list[dict[str, Any]],
    tokens_by_id: dict[str, list[str]],
    section_ids: dict[tuple[str, ...], int],
) -> None:
    # embeddings 会随 chunks 的 ON DELETE CASCADE 一起删除。
    conn.execute("DELETE FROM chunks WHERE paper_id = %s", (paper_id,))
    for index, chunk in enumerate(chunks, 1):
        path = chunk.get("section_path") or []
        conn.execute(
            """
            INSERT INTO chunks
                (chunk_id, paper_id, section_id, chunk_index, chunk_type,
                 section_path, page_start, page_end, text, index_text,
                 char_count, tokens, prev_id, next_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                chunk["chunk_id"],
                paper_id,
                section_ids.get(tuple(path)),
                index,
                chunk.get("type", "body"),
                Jsonb(path),
                chunk.get("page_start"),
                chunk.get("page_end"),
                chunk.get("text", ""),
                chunk.get("index_text", ""),
                int(chunk.get("char_count") or len(chunk.get("text", ""))),
                Jsonb(tokens_by_id.get(chunk["chunk_id"], [])),
                chunk.get("prev_id") or None,
                chunk.get("next_id") or None,
            ),
        )


def replace_embeddings(
    conn: psycopg.Connection,
    paper_id: str,
    embeddings: dict[str, list[float]],
    model: str,
    dim: int,
) -> None:
    if not embeddings:
        return
    for chunk_id, vector in embeddings.items():
        conn.execute(
            """
            INSERT INTO embeddings (chunk_id, model, dim, vector)
            VALUES (%s, %s, %s, %s)
            """,
            (chunk_id, model, dim, vector),
        )


def load_chunks(conn: psycopg.Connection | None = None) -> list[dict[str, Any]]:
    """从数据库读取检索所需的切片，字段和原来的 JSON 保持一致。"""
    if conn is None:
        with connect() as managed:
            return load_chunks(managed)
    rows = conn.execute(
        """
        SELECT c.chunk_id, c.paper_id, c.chunk_type AS type,
               c.section_path, c.page_start, c.page_end, c.text,
               c.index_text, c.char_count, c.tokens,
               c.prev_id, c.next_id, p.title AS paper_title
        FROM chunks c
        JOIN papers p ON p.paper_id = c.paper_id
        ORDER BY c.paper_id, c.chunk_index
        """
    ).fetchall()
    return [dict(row) for row in rows]


def embedding_status(
    conn: psycopg.Connection | None = None,
) -> dict[str, Any]:
    """返回向量表的数量、模型和维度。"""
    if conn is None:
        with connect() as managed:
            return embedding_status(managed)
    row = conn.execute(
        """
        SELECT count(*) AS count,
               min(model) AS model,
               min(dim) AS dim,
               max(created_at) AS built_at
        FROM embeddings
        """
    ).fetchone()
    return {
        "count": int(row["count"] or 0),
        "model": row["model"],
        "dim": row["dim"],
        "built_at": row["built_at"],
    }


def vector_scores(
    query_vector: list[float],
    *,
    limit: int,
    min_cosine: float = 0.0,
    exact: bool = True,
    conn: psycopg.Connection | None = None,
) -> dict[str, float]:
    """用 pgvector 的余弦距离取候选，返回 chunk_id → 相似度。"""
    if conn is None:
        with connect() as managed:
            return vector_scores(
                query_vector,
                limit=limit,
                min_cosine=min_cosine,
                exact=exact,
                conn=managed,
            )

    if exact:
        # 数据量小的时候强制走顺序扫描，结果和原来的 numpy 点积一致；
        # 以后数据量大再关掉这里，使用 HNSW 索引。
        conn.execute("SET LOCAL enable_indexscan = off")
        conn.execute("SET LOCAL enable_bitmapscan = off")

    query = Vector(query_vector)
    rows = conn.execute(
        """
        SELECT chunk_id,
               1 - (vector <=> %s) AS similarity
        FROM embeddings
        WHERE 1 - (vector <=> %s) >= %s
        ORDER BY vector <=> %s
        LIMIT %s
        """,
        (
            query,
            query,
            min_cosine,
            query,
            limit,
        ),
    ).fetchall()
    return {
        str(row["chunk_id"]): float(row["similarity"])
        for row in rows
    }


def stats(conn: psycopg.Connection | None = None) -> dict[str, int]:
    if conn is None:
        with connect() as managed:
            return stats(managed)
    row = conn.execute(
        """
        SELECT
            (SELECT count(*) FROM papers) AS papers,
            (SELECT count(*) FROM sections) AS sections,
            (SELECT count(*) FROM chunks) AS chunks,
            (SELECT count(*) FROM embeddings) AS embeddings,
            (SELECT count(*) FROM queries) AS queries,
            (SELECT count(*) FROM agent_runs) AS agent_runs
        """
    ).fetchone()
    return {key: int(value) for key, value in row.items()}


def log_query(
    conn: psycopg.Connection,
    *,
    question: str,
    mode: str,
    rerank: bool,
    structured: bool,
    answer: str,
    structured_answer: dict[str, Any] | None,
    citation_issues: Iterable[str],
    latency_ms: int,
    results: Iterable[dict[str, Any]],
) -> int:
    row = conn.execute(
        """
        INSERT INTO queries
            (question, mode, rerank, structured, answer,
             structured_answer, citation_issues, latency_ms)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING query_id
        """,
        (
            question,
            mode,
            rerank,
            structured,
            answer,
            Jsonb(structured_answer) if structured_answer else None,
            Jsonb(list(citation_issues)),
            latency_ms,
        ),
    ).fetchone()
    query_id = int(row["query_id"])
    for result in results:
        conn.execute(
            """
            INSERT INTO query_results
                (query_id, rank, chunk_id, score, bm25_score,
                 vector_score, rerank_score)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (
                query_id,
                result.get("rank"),
                result.get("chunk_id"),
                result.get("score"),
                result.get("bm25_score"),
                result.get("vector_score"),
                result.get("rerank_score"),
            ),
        )
    return query_id
