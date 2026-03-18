"""In-memory vector database adapter — for development and testing.

Uses NumPy for vector similarity computation. All data is held in RAM
and lost when the process exits. Register via:

    AdapterRegistry.register_vector_db("in_memory", InMemoryVectorAdapter)
"""

from __future__ import annotations

import threading
from typing import Any

import numpy as np

from openrag.storage.base import BaseVectorDBAdapter, SearchResult, VectorRecord


class InMemoryVectorAdapter(BaseVectorDBAdapter):
    """Pure-Python, thread-safe in-memory vector store backed by NumPy.

    Suitable for unit tests and local development.
    Not suitable for production (no persistence, O(N) scan per query).
    """

    def __init__(self, config: object = None) -> None:
        # namespace → {id: VectorRecord}
        self._store: dict[str, dict[str, VectorRecord]] = {}
        self._lock = threading.Lock()

    async def initialize(self) -> None:
        """No-op for in-memory adapter."""

    async def upsert(self, namespace: str, records: list[VectorRecord]) -> None:
        """Insert or overwrite records by ID."""
        with self._lock:
            ns = self._store.setdefault(namespace, {})
            for record in records:
                ns[record.id] = record

    async def query(
        self,
        namespace: str,
        vector: np.ndarray,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Return top_k most similar records using cosine similarity."""
        with self._lock:
            ns = self._store.get(namespace, {})
            if not ns:
                return []

            records = list(ns.values())

            # Apply metadata filters first
            if filters:
                records = [
                    r for r in records
                    if all(r.payload.get(k) == v for k, v in filters.items())
                ]
            if not records:
                return []

            # Stack vectors into a matrix for batch cosine similarity
            matrix = np.stack([r.vector for r in records])
            q_norm = vector / (np.linalg.norm(vector) + 1e-10)
            m_norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
            scores = (matrix / m_norms) @ q_norm

            # Sort descending, take top_k
            k = min(top_k, len(records))
            top_indices = np.argpartition(scores, -k)[-k:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

            return [
                SearchResult(
                    id=records[i].id,
                    score=float(scores[i]),
                    payload=records[i].payload,
                )
                for i in top_indices
            ]

    async def delete(self, namespace: str, ids: list[str]) -> None:
        """Delete records by ID. Silently ignores IDs that don't exist."""
        with self._lock:
            ns = self._store.get(namespace, {})
            for record_id in ids:
                ns.pop(record_id, None)

    async def count(self, namespace: str) -> int:
        """Return total number of stored vectors in the namespace."""
        with self._lock:
            return len(self._store.get(namespace, {}))

    async def clear_namespace(self, namespace: str) -> None:
        """Remove all records in a namespace (useful for tests)."""
        with self._lock:
            self._store.pop(namespace, None)
