"""HuggingFace sentence-transformers embedding adapter."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

import numpy as np

from openrag.embeddings.base import BaseEmbeddingAdapter

if TYPE_CHECKING:
    from openrag.config import EmbeddingConfig


class HuggingFaceEmbeddingAdapter(BaseEmbeddingAdapter):
    """Adapter for HuggingFace local models (e.g., sentence-transformers).

    Uses sentence_transformers to generate embeddings locally.
    Runs in the default loop executor to avoid blocking the async loop.
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence_transformers is required for HuggingFaceEmbeddingAdapter. "
                "Install it with `pip install sentence-transformers`."
            ) from None

        target_model = config.model if config.model and config.model != "default" else "BAAI/bge-base-en-v1.5"
        
        try:
            self._model_name = target_model
            self._model = SentenceTransformer(self._model_name)
        except Exception as e:
            print(f"WARNING: Failed to load {target_model} natively ({e}). Falling back to lightweight all-MiniLM-L6-v2.")
            self._model_name = "all-MiniLM-L6-v2"
            self._model = SentenceTransformer(self._model_name)

        # Dynamically sense dimension sizes by test encoding
        test_vec = self._model.encode(["test_dimension"], convert_to_numpy=True)
        self._dimensions = int(test_vec.shape[1])

    async def embed(self, texts: list[str]) -> list[np.ndarray]:
        """Generate embeddings using the local model in a thread pool."""
        if not texts:
            return []

        # sentence-transformers encode is cpu/gpu bound, run in executor
        import functools
        loop = asyncio.get_event_loop()
        func = functools.partial(self._model.encode, convert_to_numpy=True)
        embeddings: Any = await loop.run_in_executor(None, func, texts)
        return [np.array(e, dtype=np.float32) for e in embeddings]

    def dimension(self) -> int:
        return self._dimensions
