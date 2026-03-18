"""Query-layer data contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from openrag.models.content import BlockType


class QueryMode(str, Enum):
    """Available retrieval strategies."""

    DENSE = "dense"
    """Pure semantic vector similarity search."""

    SPARSE = "sparse"
    """BM25 keyword search."""

    GRAPH = "graph"
    """Knowledge-graph traversal anchored on extracted entities."""

    HYBRID = "hybrid"
    """Dense + sparse fused via Reciprocal Rank Fusion (default)."""

    MULTIMODAL = "multimodal"
    """Hybrid + VLM pass for image blocks in retrieved context."""


@dataclass
class QueryRequest:
    """All parameters for a single query invocation."""

    text: str
    """The user's question or search string."""

    mode: QueryMode = QueryMode.HYBRID
    namespace: str = "default"
    top_k: int = 20
    """Number of candidate chunks to retrieve before re-ranking."""

    output_schema: dict[str, Any] | None = None
    """Optional JSON Schema constraining the answer format."""

    session_id: str | None = None
    """Enables conversational multi-turn memory when provided."""

    stream: bool = False
    """Return a streaming generator instead of a full response."""

    tools: list[str] | None = None
    """Names of registered tools the LLM may invoke during synthesis."""

    filters: dict[str, Any] = field(default_factory=dict)
    """Extra metadata filters forwarded to the vector DB."""


@dataclass
class Citation:
    """A provenance reference attached to a query answer."""

    document_id: str
    document_title: str | None
    page_number: int | None
    block_id: str
    block_type: BlockType
    score: float
    """Retrieval relevance score (0.0–1.0, higher = more relevant)."""

    excerpt: str | None = None
    """Short text snippet from the source block."""


@dataclass
class QueryResponse:
    """The complete result of a query invocation."""

    answer: str
    """Natural language answer synthesized by the LLM."""

    citations: list[Citation]
    """Ordered list of source references used in the answer."""

    query_mode: QueryMode
    latency_ms: float

    session_id: str | None = None
    structured: dict[str, Any] | None = None
    """Populated when output_schema was provided and parsing succeeded."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Extra diagnostic info: tokens_used, retriever_counts, etc."""
