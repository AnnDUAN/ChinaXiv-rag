"""本地重排序（reranker）。

前面的 BM25 和向量检索负责“多召回一些候选”，
这里再把“问题 + 候选段落”放在一起读一遍，重新判断相关性。

模型是 fastembed 提供的 ONNX 版本，不需要 torch；
第一次运行会下载到 data/models，之后完全本地。
"""

from __future__ import annotations

import os
from typing import Any

from .. import config


def _fix_proxy_env() -> None:
    """Cherry Studio 的 NO_PROXY 里带 [::1]，httpx 会当成非法端口。"""
    for key in ("NO_PROXY", "no_proxy"):
        if key in os.environ:
            os.environ[key] = os.environ[key].replace("[::1]", "::1")


class LocalReranker:
    """懒加载的本地交叉编码器。"""

    _instance: "LocalReranker | None" = None

    def __init__(
        self,
        model_name: str = config.RERANK_MODEL,
        cache_dir: str | None = None,
    ):
        _fix_proxy_env()
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.model_name = model_name
        self.model = TextCrossEncoder(
            model_name=model_name,
            cache_dir=str(cache_dir or (config.DATA_DIR / "models")),
            threads=min(8, os.cpu_count() or 4),
        )

    @classmethod
    def get(cls) -> "LocalReranker":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def score(
        self,
        query: str,
        documents: list[str],
        batch_size: int = config.RERANK_BATCH_SIZE,
    ) -> list[float]:
        """给每个候选段落打一个相关性分数，分数越大越相关。"""
        if not documents:
            return []
        scores = list(
            self.model.rerank(query, documents, batch_size=batch_size)
        )
        if len(scores) != len(documents):
            raise RuntimeError(
                f"重排序模型返回了 {len(scores)} 个分数，"
                f"但候选段落有 {len(documents)} 个。"
            )
        return [float(s) for s in scores]


def build_rerank_text(doc: dict[str, Any]) -> str:
    """拼出送给重排序模型的文本：论文标题 + 章节 + 原文。"""
    index_text = doc.get("index_text")
    if index_text:
        return str(index_text)
    title = doc.get("paper_title") or doc.get("title") or ""
    section = " > ".join(doc.get("section_path") or [])
    text = doc.get("text", "")
    prefix = f"《{title}》 > {section}" if section else f"《{title}》"
    return f"{prefix} > {text}"
