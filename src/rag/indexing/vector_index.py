"""向量索引：切片 → 512 维向量，保存成一个压缩的 npz 文件。"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .. import config

VECTOR_FILE = config.INDEX_DIR / "vectors.npz"
META_FILE = config.INDEX_DIR / "vectors_meta.json"


def _chunk_signature(chunks: list[dict[str, Any]]) -> str:
    ids = sorted(c.get("chunk_id", "") for c in chunks)
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()[:16]


def build_vector_index(
    chunks: list[dict[str, Any]],
    embedder=None,
    force: bool = False,
) -> dict[str, Any]:
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    if not chunks:
        raise ValueError("没有切片，不能建向量索引。")

    signature = _chunk_signature(chunks)
    if VECTOR_FILE.exists() and META_FILE.exists() and not force:
        meta = json.loads(META_FILE.read_text(encoding="utf-8"))
        if meta.get("chunk_signature") == signature and meta.get("model") == config.EMBED_MODEL:
            return {**meta, "skipped": True}

    if embedder is None:
        from .embedder import LocalEmbedder

        embedder = LocalEmbedder.get()
    texts = [c.get("index_text") or c.get("text", "") for c in chunks]
    vectors = embedder.embed_passages(texts)

    chunk_ids = np.array([c.get("chunk_id", "") for c in chunks], dtype=object)
    paper_ids = np.array([c.get("paper_id", "") for c in chunks], dtype=object)
    np.savez_compressed(VECTOR_FILE, vectors=vectors, chunk_ids=chunk_ids, paper_ids=paper_ids)

    meta = {
        "model": embedder.model_name,
        "dim": int(vectors.shape[1]),
        "count": int(vectors.shape[0]),
        "chunk_signature": signature,
        "built_at": datetime.now().isoformat(timespec="seconds"),
    }
    META_FILE.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def load_vector_index() -> dict[str, Any] | None:
    if not VECTOR_FILE.exists() or not META_FILE.exists():
        return None
    meta = json.loads(META_FILE.read_text(encoding="utf-8"))
    data = np.load(VECTOR_FILE, allow_pickle=True)
    return {
        "meta": meta,
        "vectors": data["vectors"].astype(np.float32),
        "chunk_ids": [str(x) for x in data["chunk_ids"]],
        "paper_ids": [str(x) for x in data["paper_ids"]],
    }


def index_status(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    loaded = load_vector_index()
    if not loaded:
        return {"exists": False}
    meta = loaded["meta"]
    searchable = [c for c in chunks if c.get("type") != "keywords"]
    return {
        "exists": True,
        "model": meta.get("model"),
        "dim": meta.get("dim"),
        "count": meta.get("count"),
        "built_at": meta.get("built_at"),
        "up_to_date": (
            meta.get("chunk_signature") == _chunk_signature(searchable)
            and meta.get("model") == config.EMBED_MODEL
        ),
    }
