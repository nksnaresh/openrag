"""Knowledge Graph Builder that transforms blocks into graph records."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from openrag.knowledge.entity_extractor import EntityExtractor
from openrag.knowledge.relationship_extractor import RelationshipExtractor
from openrag.models.content import BlockType

if TYPE_CHECKING:
    from openrag.models.content import ContentPayload
    from openrag.models.processing import ProcessedBlock, ProcessingContext
    from openrag.storage.graph.base import BaseGraphDBAdapter


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
        doc_id = payload.document_id

        # 1. UPSERT Document node
        await self._graph_db.upsert_node(
            namespace,
            {
                "id": f"DOC:{doc_id}",
                "type": "Document",
                "title": payload.metadata.title or doc_id,
                "source_path": payload.source_path,
            },
        )

        # 2. RESOLVE Entities
        all_candidates = []
        for b in blocks:
            all_candidates.extend(b.entity_candidates)
        
        resolved_entities = self._entity_extractor.resolve(all_candidates)

        # 3. UPSERT Entity nodes and edges to Document
        for ent in resolved_entities:
            ent_id = f"ENT:{ent['canonical_name']}"
            await self._graph_db.upsert_node(
                namespace,
                {
                    "id": ent_id,
                    "type": "Entity",
                    "name": ent["name"],
                    "kind": ent["type"],
                    "canonical_name": ent["canonical_name"],
                },
            )
            # Edge: Document -> Entity (CONTAINS_KNOWLEDGE_ABOUT)
            await self._graph_db.upsert_edge(
                namespace,
                f"DOC:{doc_id}",
                ent_id,
                "MENTIONS",
                {"confidence": ent["confidence"]},
            )

        # 4. RELATIONSHIP EXTRACTION (Optional)
        # For small docs or critical segments, call LLM to find relationships
        # For Phase 3, we'll demonstrate it on the first 5 blocks to avoid over-calling LLM.
        for b in blocks[:5]:
            # Each block also has local entities that might not be in the global set
            block_entities = [e for e in resolved_entities if e["canonical_name"] in [v.canonical_name for v in b.entity_candidates]]
            
            rels = await self._rel_extractor.extract_from_block(b, block_entities, context.llm_func)
            for rel in rels:
                src_name = rel.get("source", "").lower()
                tgt_name = rel.get("target", "").lower()
                rel_type = rel.get("type", "RELATED_TO").upper()
                
                await self._graph_db.upsert_edge(
                    namespace,
                    f"ENT:{src_name}",
                    f"ENT:{tgt_name}",
                    rel_type,
                    {"description": rel.get("description", ""), "source_doc": doc_id},
                )
