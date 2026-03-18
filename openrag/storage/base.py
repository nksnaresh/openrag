"""Abstract base classes for all storage adapters.

Every concrete storage adapter must implement the interface defined here.
This ensures all adapters are interchangeable and can be tested against
the shared AdapterContractTestSuite.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from openrag.models.content import ContentPayload
from openrag.models.graph import GraphEdge, GraphNode, SubGraph

# ── Vector DB ─────────────────────────────────────────────────────────────────

class VectorRecord:
    """A single record to upsert into the vector DB."""
    __slots__ = ("id", "vector", "payload")

    def __init__(self, id: str, vector: np.ndarray, payload: dict[str, Any]) -> None:
        self.id = id
        self.vector = vector
        self.payload = payload


class SearchResult:
    """A single result returned from a vector DB query."""
    __slots__ = ("id", "score", "payload")

    def __init__(self, id: str, score: float, payload: dict[str, Any]) -> None:
        self.id = id
        self.score = score
        self.payload = payload


class BaseVectorDBAdapter(ABC):
    """Abstract interface for all vector database adapters."""

    @abstractmethod
    async def initialize(self) -> None:
        """Set up collections/indices. Called once on startup."""

    @abstractmethod
    async def upsert(self, namespace: str, records: list[VectorRecord]) -> None:
        """Insert or update records in the vector DB."""

    @abstractmethod
    async def query(
        self,
        namespace: str,
        vector: np.ndarray,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Find the top_k nearest vectors, optionally filtered by metadata."""

    @abstractmethod
    async def delete(self, namespace: str, ids: list[str]) -> None:
        """Delete records by ID."""

    @abstractmethod
    async def count(self, namespace: str) -> int:
        """Return total number of vectors in the namespace."""

    async def health_check(self) -> bool:
        """Return True if the adapter can reach its backend."""
        try:
            await self.count("__health__")
            return True
        except Exception:
            return False


# ── Graph DB ──────────────────────────────────────────────────────────────────

class BaseGraphDBAdapter(ABC):
    """Abstract interface for all graph database adapters."""

    @abstractmethod
    async def initialize(self) -> None:
        """Create schema / constraints. Called once on startup."""

    @abstractmethod
    async def upsert_node(self, node: GraphNode) -> None:
        """Insert or update a graph node."""

    @abstractmethod
    async def upsert_edge(self, edge: GraphEdge) -> None:
        """Insert or update a directed graph edge."""

    @abstractmethod
    async def get_node(self, node_id: str) -> GraphNode | None:
        """Retrieve a single node by ID."""

    @abstractmethod
    async def traverse(
        self,
        start_node_id: str,
        depth: int = 2,
        edge_types: list[str] | None = None,
    ) -> SubGraph:
        """BFS/DFS traversal from a starting node up to `depth` hops."""

    @abstractmethod
    async def find_nodes(
        self,
        namespace: str,
        node_type: str | None = None,
        label_contains: str | None = None,
        properties: dict[str, Any] | None = None,
        limit: int = 50,
    ) -> list[GraphNode]:
        """Search for nodes matching the given criteria."""

    @abstractmethod
    async def delete_node(self, node_id: str) -> None:
        """Delete a node and all its incident edges."""

    async def health_check(self) -> bool:
        try:
            await self.find_nodes("__health__", limit=1)
            return True
        except Exception:
            return False


# ── Document Store ────────────────────────────────────────────────────────────

class BaseDocumentAdapter(ABC):
    """Abstract interface for the document metadata and BM25 store."""

    @abstractmethod
    async def initialize(self) -> None:
        """Create tables/collections. Called once on startup."""

    @abstractmethod
    async def save_document(self, payload: ContentPayload, namespace: str) -> None:
        """Persist a document record."""

    @abstractmethod
    async def get_document(self, document_id: str, namespace: str) -> dict[str, Any] | None:
        """Retrieve a document record by ID."""

    @abstractmethod
    async def find_by_hash(self, namespace: str, content_hash: str) -> dict[str, Any] | None:
        """Return the document record matching the SHA-256 content hash."""

    @abstractmethod
    async def list_documents(
        self,
        namespace: str,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List all documents in a namespace with pagination."""

    @abstractmethod
    async def delete_document(self, document_id: str, namespace: str) -> None:
        """Delete a document and its associated BM25 tokens."""

    @abstractmethod
    async def save_bm25_tokens(
        self, namespace: str, chunk_id: str, tokens: list[str]
    ) -> None:
        """Persist tokenised content for BM25 indexing."""

    @abstractmethod
    async def get_all_bm25_tokens(
        self, namespace: str
    ) -> list[tuple[str, list[str]]]:
        """Return all (chunk_id, tokens) pairs for BM25 corpus construction."""

    @abstractmethod
    async def save_session(
        self, session_id: str, turns: list[dict[str, str]]
    ) -> None:
        """Persist conversational memory turns."""

    @abstractmethod
    async def get_session(
        self, session_id: str
    ) -> list[dict[str, str]]:
        """Retrieve all turns for a session."""

    async def health_check(self) -> bool:
        try:
            await self.list_documents("__health__", limit=1)
            return True
        except Exception:
            return False
