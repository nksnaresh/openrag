"""Knowledge graph data contracts — nodes, edges, and type ontology."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class NodeType(str, Enum):
    DOCUMENT = "Document"
    SECTION = "Section"
    CHUNK = "Chunk"
    ENTITY = "Entity"


class EdgeType(str, Enum):
    """Typed, directed relationships in the OpenRAG knowledge graph."""

    CONTAINS = "CONTAINS"
    """Document→Section, Section→Chunk (hierarchical containment)."""

    NEXT = "NEXT"
    """Chunk→Chunk (sequential ordering within document)."""

    REFERENCES = "REFERENCES"
    """Entity→Entity (semantic cross-reference)."""

    ILLUSTRATES = "ILLUSTRATES"
    """Chunk(image)→Entity (image visually represents the entity)."""

    DEFINES = "DEFINES"
    """Chunk(code/text)→Entity (defines or introduces the entity)."""

    IMPLEMENTS = "IMPLEMENTS"
    """Chunk(code)→Entity (code implements algorithm/concept)."""

    PROVES = "PROVES"
    """Chunk(equation)→Entity (mathematical proof of concept)."""

    CITES = "CITES"
    """Document→Document (bibliographic citation)."""

    CONTAINS_DATA_ABOUT = "CONTAINS_DATA_ABOUT"
    """Chunk(table)→Entity (table row/column pertains to entity)."""


@dataclass
class GraphNode:
    """A node in the OpenRAG knowledge graph."""

    node_id: str
    node_type: NodeType
    label: str
    tenant_id: str
    namespace: str
    properties: dict[str, Any] = field(default_factory=dict)
    """
    Type-specific properties:
    - Document:  source_path, page_count, ingested_at
    - Section:   title, level, document_id
    - Chunk:     content, block_type, page_number, block_id, embedding_id
    - Entity:    canonical_name, entity_type, description, confidence
    """


@dataclass
class GraphEdge:
    """A directed, typed edge between two nodes."""

    edge_id: str
    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float = 1.0
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class SubGraph:
    """A partial view of the graph returned by traversal queries."""

    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)

    def node_ids(self) -> list[str]:
        return [n.node_id for n in self.nodes]
