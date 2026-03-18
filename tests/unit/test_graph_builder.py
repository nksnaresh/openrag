"""Unit tests for the Knowledge Graph Builder."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from openrag.knowledge.entity_extractor import EntityExtractor
from openrag.knowledge.graph_builder import KnowledgeGraphBuilder
from openrag.knowledge.relationship_extractor import RelationshipExtractor
from openrag.models.content import BlockType, ContentBlock, ContentPayload, DocumentMeta
from openrag.models.processing import (
    EntityCandidate,
    ProcessedBlock,
    ProcessingContext,
)


class TestKnowledgeGraphBuilder:
    @pytest.fixture
    def mock_graph_db(self) -> MagicMock:
        db = MagicMock()
        db.upsert_node = AsyncMock()
        db.upsert_edge = AsyncMock()
        return db

    @pytest.fixture
    def builder(self, mock_graph_db: MagicMock) -> KnowledgeGraphBuilder:
        return KnowledgeGraphBuilder(mock_graph_db)

    @pytest.mark.asyncio
    async def test_build_from_blocks(
        self, builder: KnowledgeGraphBuilder, mock_graph_db: MagicMock
    ) -> None:
        payload = ContentPayload(
            document_id="doc1",
            source_path="file.pdf",
            tenant_id="t1",
            metadata=DocumentMeta(title="Test Doc"),
            blocks=[],
        )
        
        blocks = [
            ProcessedBlock(
                source_block=ContentBlock("b1", BlockType.TEXT, 0, "content"),
                natural_language_description="desc",
                embedding_text="emb",
                entity_candidates=[
                    EntityCandidate("Apple", "Apple Inc.", "ORG", 0.9)
                ],
            )
        ]
        
        context = MagicMock(spec=ProcessingContext)
        context.namespace = "ns1"
        context.llm_func = AsyncMock(return_value='{"relationships": []}')
        
        await builder.build_from_blocks(blocks, payload, context)
        
        # Check node creation
        assert mock_graph_db.upsert_node.call_count >= 2  # Document + 1 Entity
        
        # Check edge creation
        assert mock_graph_db.upsert_edge.call_count >= 1  # MENTIONS
        
        # Verify document node ID
        doc_node = mock_graph_db.upsert_node.call_args_list[0][0][1]
        assert doc_node["id"] == "DOC:doc1"
        assert doc_node["type"] == "Document"
        
        # Verify entity node
        ent_node = mock_graph_db.upsert_node.call_args_list[1][0][1]
        assert ent_node["id"] == "ENT:Apple Inc."
        assert ent_node["canonical_name"] == "Apple Inc."
