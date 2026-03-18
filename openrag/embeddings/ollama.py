"""Ollama local embedding adapter."""

from __future__ import annotations

from typing import TYPE_CHECKING

import httpx
import numpy as np

from openrag.embeddings.base import BaseEmbeddingAdapter

if TYPE_CHECKING:
    from openrag.config import EmbeddingConfig


class OllamaEmbeddingAdapter(BaseEmbeddingAdapter):
    """Adapter for local Ollama embeddings endpoint (e.g., llama3 or mxbai-embed-large)."""

    def __init__(self, config: EmbeddingConfig) -> None:
        self._url = "http://localhost:11434/api/embeddings"
        self._model = config.model
        self._dimensions = config.dimensions if config.dimensions > 0 else 1024  # fallback or dynamic?

    async def embed(self, texts: list[str]) -> list[np.ndarray]:
        """Generate embeddings using the Ollama REST API."""
        if not texts:
            return []

        results: list[np.ndarray] = []
        async with httpx.AsyncClient(timeout=60.0) as client:
            for text in texts:
                response = await client.post(
                    self._url,
                    json={"model": self._model, "prompt": text},
                )
                response.raise_for_status()
                embedding = response.json().get("embedding", [])
                results.append(np.array(embedding, dtype=np.float32))

        return results

    def dimension(self) -> int:
        return self._dimensions
