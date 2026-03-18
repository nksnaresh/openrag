"""Shared data contracts — the lingua franca of OpenRAG.

All models in this package are plain dataclasses / pydantic models with
zero external dependencies beyond the stdlib and pydantic. Every module
in OpenRAG imports from here; nothing here imports from other openrag modules.
"""

from openrag.models.content import (
    BlockType,
    BoundingBox,
    ContentBlock,
    ContentPayload,
    DocumentMeta,
)
from openrag.models.graph import (
    EdgeType,
    GraphEdge,
    GraphNode,
    NodeType,
)
from openrag.models.jobs import (
    BatchJobResult,
    BatchOptions,
    IngestMetadata,
    JobResult,
    JobStatus,
)
from openrag.models.processing import (
    ContextWindowConfig,
    EntityCandidate,
    EntityType,
    ProcessedBlock,
    ProcessingContext,
)
from openrag.models.query import (
    Citation,
    QueryMode,
    QueryRequest,
    QueryResponse,
)

__all__ = [
    # content
    "BlockType", "BoundingBox", "ContentBlock", "ContentPayload", "DocumentMeta",
    # processing
    "ContextWindowConfig", "EntityCandidate", "EntityType",
    "ProcessedBlock", "ProcessingContext",
    # query
    "Citation", "QueryMode", "QueryRequest", "QueryResponse",
    # jobs
    "JobResult", "JobStatus", "BatchJobResult", "BatchOptions", "IngestMetadata",
    # graph
    "GraphNode", "GraphEdge", "EdgeType", "NodeType",
]
