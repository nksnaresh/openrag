"""OpenAI embedding adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from openai import AsyncOpenAI

from openrag.embeddings.base import BaseEmbeddingAdapter

if TYPE_CHECKING:
    from openrag.config import EmbeddingConfig


class OpenAIEmbeddingAdapter(BaseEmbeddingAdapter):
    """Adapter for OpenAI Embeddings API (e.g., text-embedding-3-large)."""

    def __init__(self, config: EmbeddingConfig) -> None:
        self._client = AsyncOpenAI()
        self._model = config.model
        self._dimensions = config.dimensions

    async def embed(self, texts: list[str]) -> list[np.ndarray]:
        """Call OpenAI embeddings API in a single batch."""
        if not texts:
            return []

        response = await self._client.embeddings.create(
            input=texts,
            model=self._model,
            dimensions=self._dimensions,
        )

        return [np.array(item.embedding, dtype=np.float32) for item in response.data]

    def dimension(self) -> int:
        return self._dimensions
