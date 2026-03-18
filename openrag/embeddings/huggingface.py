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

        self._model_name = config.model
        self._model = SentenceTransformer(self._model_name)
        self._dimensions = config.dimensions

    async def embed(self, texts: list[str]) -> list[np.ndarray]:
        """Generate embeddings using the local model in a thread pool."""
        if not texts:
            return []

        # sentence-transformers encode is cpu/gpu bound, run in executor
        loop = asyncio.get_event_loop()
        embeddings: Any = await loop.run_in_executor(
            None, self._model.encode, texts, {"convert_to_numpy": True}
        )
        return [np.array(e, dtype=np.float32) for e in embeddings]

    def dimension(self) -> int:
        return self._dimensions
