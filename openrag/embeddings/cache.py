"""Embedding cache implementations.

Provides an abstract `BaseEmbeddingCache` interface and a concrete
`InMemoryEmbeddingCache` for development and testing.
"""

from __future__ import annotations

import contextlib
import threading
from abc import ABC, abstractmethod

import numpy as np


class BaseEmbeddingCache(ABC):
    """Abstract interface for embedding vector caches."""

    @abstractmethod
    async def get(self, key: str) -> np.ndarray | None:
        """Return the cached vector for ``key``, or None on cache miss."""

    @abstractmethod
    async def set(self, key: str, vector: np.ndarray) -> None:
        """Store ``vector`` under ``key``."""

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Evict a key from the cache."""

    @abstractmethod
    async def clear(self) -> None:
        """Evict all entries from the cache."""

    @abstractmethod
    async def size(self) -> int:
        """Return the number of cached entries."""


class InMemoryEmbeddingCache(BaseEmbeddingCache):
    """Thread-safe in-memory embedding cache backed by a plain Python dict.

    All cached vectors are stored as ``numpy.ndarray`` objects.
    Data is lost when the process exits.

    Args:
        max_size: Optional maximum number of entries. When exceeded, the
                  oldest inserted key is evicted (FIFO). ``None`` = unlimited.
    """

    def __init__(self, max_size: int | None = None) -> None:
        self._cache: dict[str, np.ndarray] = {}
        self._insertion_order: list[str] = []  # FIFO eviction tracking
        self._max_size = max_size
        self._lock = threading.Lock()

    async def get(self, key: str) -> np.ndarray | None:
        """Return cached vector or None on miss."""
        with self._lock:
            return self._cache.get(key)

    async def set(self, key: str, vector: np.ndarray) -> None:
        """Cache ``vector`` under ``key``. Evicts oldest entry if at capacity."""
        with self._lock:
            if key in self._cache:
                # Update existing without changing insertion order
                self._cache[key] = vector
            else:
                if self._max_size and len(self._cache) >= self._max_size:
                    # FIFO eviction
                    oldest = self._insertion_order.pop(0)
                    self._cache.pop(oldest, None)
                self._cache[key] = vector
                self._insertion_order.append(key)

    async def delete(self, key: str) -> None:
        """Remove a single key from the cache."""
        with self._lock:
            self._cache.pop(key, None)
            with contextlib.suppress(ValueError):
                self._insertion_order.remove(key)

    async def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()
            self._insertion_order.clear()

    async def size(self) -> int:
        """Return number of cached entries."""
        with self._lock:
            return len(self._cache)
