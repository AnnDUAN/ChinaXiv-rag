"""本地中文向量模型。

用 fastembed + onnxruntime，不需要 torch，CPU 就能跑。
模型：BAAI/bge-small-zh-v1.5（约 90MB，512 维）。
第一次运行会下载模型，之后完全本地。
"""

from __future__ import annotations

import os

# Cherry Studio 设置的 NO_PROXY 里带有 "[::1]"，httpx 会把它当成非法端口。
# 在 huggingface_hub 建立连接前修一下，否则下载模型会直接报错。
for _key in ("NO_PROXY", "no_proxy"):
    if _key in os.environ:
        os.environ[_key] = os.environ[_key].replace("[::1]", "::1")

import numpy as np  # noqa: E402
from fastembed import TextEmbedding  # noqa: E402

from .. import config  # noqa: E402

DEFAULT_MODEL = config.EMBED_MODEL
MODEL_DIR = config.DATA_DIR / "models"


class LocalEmbedder:
    _instance: "LocalEmbedder | None" = None

    def __init__(self, model_name: str = DEFAULT_MODEL, cache_dir: str | None = None):
        self.model_name = model_name
        self.model = TextEmbedding(
            model_name=model_name,
            cache_dir=str(cache_dir or MODEL_DIR),
            threads=min(8, os.cpu_count() or 4),
        )
        self.dim = self.model.embedding_size

    @classmethod
    def get(cls) -> "LocalEmbedder":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @staticmethod
    def _normalize(array: np.ndarray) -> np.ndarray:
        array = np.asarray(array, dtype=np.float32)
        norms = np.linalg.norm(array, axis=-1, keepdims=True)
        return array / np.clip(norms, 1e-12, None)

    def embed_passages(self, texts: list[str], batch_size: int = config.EMBED_BATCH_SIZE) -> np.ndarray:
        vectors = list(self.model.passage_embed(texts, batch_size=batch_size))
        return self._normalize(np.vstack(vectors))

    def embed_query(self, text: str) -> np.ndarray:
        vector = list(self.model.query_embed([text]))[0]
        return self._normalize(np.asarray(vector, dtype=np.float32))
