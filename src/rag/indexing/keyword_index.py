"""jieba 分词 + BM25 关键词索引。

索引文件：data/index/jieba_bm25.json
里面保存每个切片用 jieba 切出来的词、词频、文档频率和 IDF，
检索时直接读这个文件，不需要每次重新分词。
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .. import config
from .bm25 import BM25, tokenize

KEYWORD_FILE = config.INDEX_DIR / "jieba_bm25.json"


def _signature(chunks: list[dict[str, Any]]) -> str:
    ids = sorted(c.get("chunk_id", "") for c in chunks)
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()[:16]


def build_keyword_index(
    chunks: list[dict[str, Any]],
    force: bool = False,
) -> dict[str, Any]:
    """用 jieba 给每个切片分词，建立并保存 BM25 索引。"""
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    searchable = [c for c in chunks if c.get("type") != "keywords"]
    if not searchable:
        raise ValueError("没有可检索的切片。")

    signature = _signature(searchable)
    if KEYWORD_FILE.exists() and not force:
        old = json.loads(KEYWORD_FILE.read_text(encoding="utf-8"))
        if old.get("chunk_signature") == signature:
            return {**old, "skipped": True}

    texts = [
        " ".join(c.get("section_path", [])) + " " + c.get("text", "")
        for c in searchable
    ]
    tokens = [tokenize(t) for t in texts]
    bm25 = BM25(tokens)

    index = {
        "model": "jieba+bm25",
        "k1": config.BM25_K1,
        "b": config.BM25_B,
        "chunk_signature": signature,
        "count": len(searchable),
        "chunk_ids": [c.get("chunk_id", "") for c in searchable],
        "tokens": tokens,
        "doc_len": bm25.doc_len,
        "avgdl": round(bm25.avgdl, 4),
        "df": dict(sorted(bm25.df.items())),
        "idf": {k: round(v, 6) for k, v in sorted(bm25.idf.items())},
        "built_at": datetime.now().isoformat(timespec="seconds"),
    }
    KEYWORD_FILE.write_text(
        json.dumps(index, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    return index


def load_keyword_index() -> dict[str, Any] | None:
    if not KEYWORD_FILE.exists():
        return None
    try:
        return json.loads(KEYWORD_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def index_status(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    index = load_keyword_index()
    if not index:
        return {"exists": False}
    searchable = [c for c in chunks if c.get("type") != "keywords"]
    return {
        "exists": True,
        "count": index.get("count"),
        "vocab": len(index.get("df", {})),
        "avgdl": index.get("avgdl"),
        "built_at": index.get("built_at"),
        "up_to_date": index.get("chunk_signature") == _signature(searchable),
    }
