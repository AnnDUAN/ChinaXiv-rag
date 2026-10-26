"""建表和把现有 JSON/npz 数据导入 PostgreSQL。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .. import config
from ..indexing.bm25 import tokenize
from .engine import connect
from .repo import (
    replace_chunks,
    replace_embeddings,
    replace_sections,
    upsert_paper,
)


def init_schema() -> None:
    """创建扩展和所有表，重复执行不会报错。"""
    sql = config.DB_SCHEMA.read_text(encoding="utf-8")
    with connect(use_vector=False) as conn:
        conn.execute(sql)


def _sanitize(value: Any) -> Any:
    """去掉 PostgreSQL text 不接受的 NUL 字符。

    个别 PDF 解析结果里会夹带 0x00；它不影响本地 JSON 检索，
    但直接写 PostgreSQL 的 text/jsonb 字段会报错。
    """
    if isinstance(value, str):
        return value.replace("\x00", "")
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    if isinstance(value, dict):
        return {
            _sanitize(key): _sanitize(item)
            for key, item in value.items()
        }
    return value


def _load_tokens(data_dir: Path) -> dict[str, list[str]]:
    path = data_dir / "index" / "jieba_bm25.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    tokens = {
        chunk_id: token_list
        for chunk_id, token_list in zip(
            data.get("chunk_ids", []), data.get("tokens", [])
        )
    }
    return _sanitize(tokens)


def _load_vectors(
    data_dir: Path,
) -> tuple[dict[str, list[float]], str, int]:
    vector_path = data_dir / "index" / "vectors.npz"
    meta_path = data_dir / "index" / "vectors_meta.json"
    if not vector_path.exists():
        return {}, config.EMBED_MODEL, 0

    data = np.load(vector_path, allow_pickle=True)
    ids = [str(x) for x in data["chunk_ids"]]
    vectors = data["vectors"].astype(np.float32)
    mapping = {
        chunk_id: [float(x) for x in vector]
        for chunk_id, vector in zip(ids, vectors)
    }
    meta: dict[str, Any] = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return (
        mapping,
        str(meta.get("model") or config.EMBED_MODEL),
        int(meta.get("dim") or vectors.shape[1]),
    )


def _paper_payload(
    parsed: dict[str, Any],
    data_dir: Path,
) -> dict[str, Any]:
    meta = parsed.get("meta") or {}
    paper_id = str(parsed.get("paper_id") or meta.get("paper_id"))
    pdf_path = meta.get("pdf") or str(
        data_dir / "pdfs" / f"{paper_id}.pdf"
    )
    year = meta.get("year")
    try:
        year = int(year) if year is not None else None
    except (TypeError, ValueError):
        year = None
    return {
        "paper_id": paper_id,
        "title": meta.get("title") or paper_id,
        "authors": meta.get("authors") or [],
        "affiliations": meta.get("affiliations") or [],
        "abstract": meta.get("abstract"),
        "keywords": meta.get("keywords") or [],
        "doi": meta.get("doi"),
        "journal": meta.get("journal"),
        "year": year,
        "source": meta.get("source"),
        "source_url": meta.get("source_url"),
        "pdf_path": pdf_path,
        "status": "ready",
        "quality": parsed.get("quality") or {},
        "meta": meta,
    }


def import_legacy(data_dir: Path | None = None) -> dict[str, int]:
    """把 data/parsed、data/chunks、data/index 导入数据库。

    重复执行是安全的：论文会更新，章节/切片会先删后插。
    """
    data_dir = Path(data_dir or config.DATA_DIR)
    tokens_by_id = _load_tokens(data_dir)
    vectors_by_id, model, dim = _load_vectors(data_dir)

    parsed_files = sorted((data_dir / "parsed").glob("*.json"))
    paper_count = 0
    chunk_count = 0
    embedding_count = 0

    with connect() as conn:
        for parsed_path in parsed_files:
            parsed = _sanitize(
                json.loads(parsed_path.read_text(encoding="utf-8"))
            )
            paper_id = str(
                parsed.get("paper_id")
                or (parsed.get("meta") or {}).get("paper_id")
            )
            chunks_path = data_dir / "chunks" / f"{paper_id}.json"
            if not chunks_path.exists():
                continue

            chunk_data = _sanitize(
                json.loads(chunks_path.read_text(encoding="utf-8"))
            )
            upsert_paper(conn, _paper_payload(parsed, data_dir))
            section_ids = replace_sections(
                conn, paper_id, parsed.get("sections") or []
            )
            chunks = chunk_data.get("chunks") or []
            for chunk in chunks:
                if chunk["chunk_id"] not in tokens_by_id:
                    tokens_by_id[chunk["chunk_id"]] = tokenize(
                        chunk.get("text", "")
                    )
            replace_chunks(
                conn, paper_id, chunks, tokens_by_id, section_ids
            )

            paper_vectors = {
                chunk["chunk_id"]: vectors_by_id[chunk["chunk_id"]]
                for chunk in chunks
                if chunk["chunk_id"] in vectors_by_id
            }
            replace_embeddings(
                conn, paper_id, paper_vectors, model, dim
            )

            paper_count += 1
            chunk_count += len(chunks)
            embedding_count += len(paper_vectors)

    return {
        "papers": paper_count,
        "chunks": chunk_count,
        "embeddings": embedding_count,
    }
