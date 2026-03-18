"""Embedding engine that orchestrates batching and caching.

Wires together a BaseEmbeddingAdapter and a BaseEmbeddingCache to provide
high-level embedding services for both ingestion and retrieval.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from openrag.embeddings.base import BaseEmbeddingAdapter
from openrag.embeddings.cache import BaseEmbeddingCache

if TYPE_CHECKING:
    from openrag.models.processing import ProcessedBlock


class EmbeddingEngine:
    """Orchestrates embedding generation with batching and caching.

    Args:
        adapter:    Concrete provider (e.g., OpenAI, HuggingFace).
        cache:      Embedding cache (e.g., InMemory, Redis).
        batch_size: Max number of texts to embed in a single provider call.
    """

    def __init__(
        self,
        adapter: BaseEmbeddingAdapter,
        cache: BaseEmbeddingCache,
        batch_size: int = 100,
    ) -> None:
        self._adapter = adapter
        self._cache = cache
        self._batch_size = batch_size

    async def embed_blocks(self, blocks: list[ProcessedBlock]) -> None:
        """Generate and attach embeddings to a list of ProcessedBlocks.

        Uses the cache to avoid re-embedding identical text. Updates each
        block's internal state (embedding_text is used as the key).
        Note: The vector itself is typically stored in the Vector DB,
        but the engine ensures the computation is done/cached.
        """
        if not blocks:
            return

        # 1. Collect unique texts that need embedding
        # Map: text_hash -> (text, list_of_blocks)
        to_embed_map: dict[str, tuple[str, list[ProcessedBlock]]] = {}
        
        for block in blocks:
            text = block.embedding_text
            if not text:
                continue
            
            text_hash = hashlib.sha256(text.encode()).hexdigest()
            # Check cache
            cached_vector = await self._cache.get(text_hash)
            if cached_vector is not None:
                # Cache hit: we don't store the vector in ProcessedBlock (no field for it),
                # but we could if the model supported it. For now, the engine's job
                # is to ensure the cache is warm for the vector store upsert.
                continue
                
            if text_hash not in to_embed_map:
                to_embed_map[text_hash] = (text, [])
            to_embed_map[text_hash][1].append(block)

        if not to_embed_map:
            return

        # 2. Batch process uncached texts
        hashes = list(to_embed_map.keys())
        texts = [to_embed_map[h][0] for h in hashes]
        
        for i in range(0, len(texts), self._batch_size):
            batch_texts = texts[i : i + self._batch_size]
            batch_hashes = hashes[i : i + self._batch_size]
            
            vectors = await self._adapter.embed(batch_texts)
            
            # 3. Update cache
            for text_hash, vector in zip(batch_hashes, vectors, strict=True):
                await self._cache.set(text_hash, vector)

    async def embed_query(self, text: str) -> Any:  # noqa: ANN401
        """Generate an embedding for a query string, using cache if available."""
        if not text:
            return None
            
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        cached = await self._cache.get(text_hash)
        if cached is not None:
            return cached
            
        vectors = await self._adapter.embed([text])
        if not vectors:
            return None
            
        vector = vectors[0]
        await self._cache.set(text_hash, vector)
        return vector
