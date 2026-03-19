"""Knowledge Graph Builder that transforms blocks into graph records."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from openrag.knowledge.entity_extractor import EntityExtractor
from openrag.knowledge.relationship_extractor import RelationshipExtractor
from openrag.models.content import BlockType
from openrag.models.graph import EdgeType, GraphEdge, GraphNode, NodeType

if TYPE_CHECKING:
    from openrag.models.content import ContentPayload
    from openrag.models.processing import ProcessedBlock, ProcessingContext
    from openrag.storage.base import BaseGraphDBAdapter


class KnowledgeGraphBuilder:
    """Builds a Knowledge Graph from document hierarchy and entity resolution.

    Orchestrates entity extraction, deduplication, and relationship logic,
    then persists the results to the graph store.
    """

    def __init__(
        self,
        graph_db: BaseGraphDBAdapter,
        entity_extractor: EntityExtractor | None = None,
        rel_extractor: RelationshipExtractor | None = None,
    ) -> None:
        self._graph_db = graph_db
        self._entity_extractor = entity_extractor or EntityExtractor()
        self._rel_extractor = rel_extractor or RelationshipExtractor()

    async def build_from_blocks(
        self,
        blocks: list[ProcessedBlock],
        payload: ContentPayload,
        context: ProcessingContext,
    ) -> None:
        """Process a list of blocks and generate nodes/edges.

        Steps:
        1. Create Document/Core nodes.
        2. Resolve entities from current blocks.
        3. Identify relationships (internal and external to doc).
        4. Upsert to Graph DB.
        """
        if not blocks:
            return

        namespace = context.namespace
        tenant_id = context.tenant_id
        doc_id = payload.document_id

        # 1. UPSERT Document node
        doc_node = GraphNode(
            node_id=f"DOC:{doc_id}",
            node_type=NodeType.DOCUMENT,
            label=payload.metadata.title or doc_id,
            tenant_id=tenant_id,
            namespace=namespace,
            properties={"source_path": payload.source_path}
        )
        await self._graph_db.upsert_node(namespace, doc_node)

        # 2. RESOLVE Entities
        all_candidates = []
        for b in blocks:
            all_candidates.extend(b.entity_candidates)
        
        resolved_entities = self._entity_extractor.resolve(all_candidates)

        # 3. UPSERT Entity nodes and edges to Document
        for ent in resolved_entities:
            ent_id = f"ENT:{ent['canonical_name']}"
            ent_node = GraphNode(
                node_id=ent_id,
                node_type=NodeType.ENTITY,
                label=ent["name"],
                tenant_id=tenant_id,
                namespace=namespace,
                properties={
                    "kind": ent["type"],
                    "canonical_name": ent["canonical_name"],
                }
            )
            await self._graph_db.upsert_node(namespace, ent_node)

            # Edge: Document -> Entity (MENTIONS)
            edge = GraphEdge(
                edge_id=f"DOC:{doc_id}->{ent_id}",
                source_id=f"DOC:{doc_id}",
                target_id=ent_id,
                edge_type=EdgeType.CONTAINS, # Or MENTIONS if added to EdgeType
                weight=float(ent["confidence"]),
                properties={"relationship": "mentions"}
            )
            await self._graph_db.upsert_edge(namespace, edge)

        # 4. RELATIONSHIP EXTRACTION (Optional)
        for b in blocks[:5]:
            block_entities = [e for e in resolved_entities if e["canonical_name"] in [v.canonical_name for v in b.entity_candidates]]
            
            rels = await self._rel_extractor.extract_from_block(b, block_entities, context.llm_func)
            for rel in rels:
                src_name = rel.get("source", "").lower()
                tgt_name = rel.get("target", "").lower()
                rel_type_str = rel.get("type", "RELATED_TO").upper()
                
                edge = GraphEdge(
                    edge_id=f"ENT:{src_name}->ENT:{tgt_name}",
                    source_id=f"ENT:{src_name}",
                    target_id=f"ENT:{tgt_name}",
                    edge_type=EdgeType.CONTAINS, # Default to CONTAINS if specific not in Enum
                    properties={
                        "rel_type": rel_type_str,
                        "description": rel.get("description", ""),
                        "source_doc": doc_id
                    }
                )
                await self._graph_db.upsert_edge(namespace, edge)
