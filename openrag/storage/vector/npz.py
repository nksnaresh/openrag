"""File-backed NPZ vector database adapter — persistent across restarts.

Stores vectors as NumPy .npz archives on disk. On startup, loads any
previously-saved vectors. On every upsert, saves the updated namespace
to disk atomically.

Usage::

    AdapterRegistry.register_vector_db("npz", NPZVectorAdapter)

"""
from __future__ import annotations

import asyncio
import os
import threading
from pathlib import Path
from typing import Any

import numpy as np

from openrag.storage.base import BaseVectorDBAdapter, SearchResult, VectorRecord


class NPZVectorAdapter(BaseVectorDBAdapter):
    """Persistent file-backed vector store using NumPy NPZ archives.

    All data is held in RAM for fast similarity search, and written to
    disk on every upsert so it survives server restarts.

    Args:
        config: Optional config object; reads ``working_dir`` if present.
    """

    def __init__(self, config: object = None, working_dir: str | None = None) -> None:
        self._store: dict[str, dict[str, VectorRecord]] = {}
        self._lock = threading.Lock()

        # Determine storage directory
        if not working_dir and config and hasattr(config, "working_dir"):
            working_dir = str(config.working_dir)
        
        base_dir = Path(working_dir or ".")
        self._vector_dir = base_dir / "vectors"

    async def initialize(self) -> None:
        """Load all previously-saved namespaces from disk."""
        self._vector_dir.mkdir(parents=True, exist_ok=True)
        for npz_path in self._vector_dir.glob("*.npz"):
            namespace = npz_path.stem
            await self._load_namespace(namespace)

    # ── Core API ───────────────────────────────────────────────────────────────

    async def upsert(self, namespace: str, records: list[VectorRecord]) -> None:
        """Insert or overwrite records by ID, then persist to disk."""
        with self._lock:
            ns = self._store.setdefault(namespace, {})
            for record in records:
                ns[record.id] = record
        # Persist outside the lock to avoid blocking queries
        await asyncio.to_thread(self._save_namespace, namespace)

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

        if filters:
            records = [
                r for r in records
                if all(r.payload.get(k) == v for k, v in filters.items())
            ]
        if not records:
            return []

        matrix = np.stack([r.vector for r in records])
        q_norm = vector / (np.linalg.norm(vector) + 1e-10)
        m_norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
        scores = (matrix / m_norms) @ q_norm

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
        """Delete records by ID."""
        with self._lock:
            ns = self._store.get(namespace, {})
            for record_id in ids:
                ns.pop(record_id, None)
        await asyncio.to_thread(self._save_namespace, namespace)

    async def count(self, namespace: str) -> int:
        """Return total number of stored vectors in the namespace."""
        with self._lock:
            return len(self._store.get(namespace, {}))

    async def clear_namespace(self, namespace: str) -> None:
        """Remove all records in a namespace."""
        with self._lock:
            self._store.pop(namespace, None)
        npz_path = self._vector_dir / f"{namespace}.npz"
        if npz_path.exists():
            npz_path.unlink()

    async def close(self) -> None:
        """Flush all namespaces to disk."""
        with self._lock:
            namespaces = list(self._store.keys())
        for ns in namespaces:
            await asyncio.to_thread(self._save_namespace, ns)

    # ── Serialization ──────────────────────────────────────────────────────────

    def _save_namespace(self, namespace: str) -> None:
        """Atomically save namespace vectors to an NPZ file."""
        with self._lock:
            ns = self._store.get(namespace, {})
            if not ns:
                return
            records = list(ns.values())

        ids = np.array([r.id for r in records])
        vectors = np.stack([r.vector for r in records])
        # Payloads serialised as JSON strings (one per record)
        import json
        payloads = np.array([json.dumps(r.payload) for r in records])

        tmp_path = self._vector_dir / f"{namespace}.tmp.npz"
        final_path = self._vector_dir / f"{namespace}.npz"
        np.savez_compressed(tmp_path, ids=ids, vectors=vectors, payloads=payloads)
        os.replace(tmp_path, final_path)

    async def _load_namespace(self, namespace: str) -> None:
        """Load a namespace from its NPZ file."""
        import json
        npz_path = self._vector_dir / f"{namespace}.npz"
        if not npz_path.exists():
            return

        data = np.load(npz_path, allow_pickle=False)
        ids = data["ids"]
        vectors = data["vectors"]
        payloads = data["payloads"]

        with self._lock:
            ns = self._store.setdefault(namespace, {})
            for rec_id, vec, payload_str in zip(ids, vectors, payloads):
                ns[str(rec_id)] = VectorRecord(
                    id=str(rec_id),
                    vector=vec,
                    payload=json.loads(str(payload_str)),
                )
        print(f"INFO: Loaded {len(ids)} vectors for namespace '{namespace}' from disk.")
