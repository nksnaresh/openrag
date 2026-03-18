"""Unit tests for the InMemoryEmbeddingCache."""

from __future__ import annotations

import numpy as np
import pytest

from openrag.embeddings.cache import InMemoryEmbeddingCache


class TestInMemoryEmbeddingCache:
    @pytest.fixture()
    def cache(self) -> InMemoryEmbeddingCache:
        return InMemoryEmbeddingCache()

    @pytest.mark.asyncio
    async def test_get_empty_returns_none(self, cache: InMemoryEmbeddingCache) -> None:
        result = await cache.get("missing-key")
        assert result is None

    @pytest.mark.asyncio
    async def test_set_then_get_returns_vector(self, cache: InMemoryEmbeddingCache) -> None:
        vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        await cache.set("key1", vec)
        result = await cache.get("key1")
        assert result is not None
        np.testing.assert_array_equal(result, vec)

    @pytest.mark.asyncio
    async def test_overwrite_key(self, cache: InMemoryEmbeddingCache) -> None:
        old = np.array([1.0, 0.0])
        new = np.array([0.0, 1.0])
        await cache.set("k", old)
        await cache.set("k", new)
        result = await cache.get("k")
        assert result is not None
        np.testing.assert_array_equal(result, new)

    @pytest.mark.asyncio
    async def test_size_tracking(self, cache: InMemoryEmbeddingCache) -> None:
        assert await cache.size() == 0
        await cache.set("a", np.zeros(4))
        await cache.set("b", np.zeros(4))
        assert await cache.size() == 2

    @pytest.mark.asyncio
    async def test_delete_removes_entry(self, cache: InMemoryEmbeddingCache) -> None:
        await cache.set("a", np.ones(3))
        await cache.delete("a")
        assert await cache.get("a") is None
        assert await cache.size() == 0

    @pytest.mark.asyncio
    async def test_delete_nonexistent_is_silent(self, cache: InMemoryEmbeddingCache) -> None:
        await cache.delete("ghost")  # must not raise

    @pytest.mark.asyncio
    async def test_clear_removes_all(self, cache: InMemoryEmbeddingCache) -> None:
        await cache.set("a", np.zeros(2))
        await cache.set("b", np.zeros(2))
        await cache.clear()
        assert await cache.size() == 0
        assert await cache.get("a") is None

    @pytest.mark.asyncio
    async def test_max_size_evicts_oldest(self) -> None:
        cache = InMemoryEmbeddingCache(max_size=2)
        await cache.set("first",  np.array([1.0]))
        await cache.set("second", np.array([2.0]))
        await cache.set("third",  np.array([3.0]))  # should evict "first"
        assert await cache.size() == 2
        assert await cache.get("first") is None
        assert await cache.get("second") is not None
        assert await cache.get("third") is not None

    @pytest.mark.asyncio
    async def test_update_existing_does_not_evict(self) -> None:
        cache = InMemoryEmbeddingCache(max_size=2)
        await cache.set("a", np.array([1.0]))
        await cache.set("b", np.array([2.0]))
        await cache.set("a", np.array([9.0]))  # update, not new — no eviction
        assert await cache.size() == 2
        assert await cache.get("b") is not None
