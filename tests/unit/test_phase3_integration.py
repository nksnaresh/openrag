"""Integration tests for Phase 3: Knowledge Layer Indexing."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from openrag.config import OpenRAGConfig
from openrag.embeddings.engine import EmbeddingEngine
from openrag.ingestion.orchestrator import IngestionOrchestrator
from openrag.knowledge.graph_builder import KnowledgeGraphBuilder
from openrag.models.content import BlockType, ContentBlock, ContentPayload, DocumentMeta
from openrag.models.jobs import IngestMetadata, JobStatus
from openrag.models.processing import ProcessedBlock
from openrag.pipeline.dag_engine import DAGPipelineEngine
from openrag.search.bm25_indexer import BM25Indexer


class TestPhase3Integration:
    """Tests the orchestrator's ability to run all 3 indexing engines."""

    @pytest.mark.asyncio
    async def test_orchestrator_parallel_indexing(self, mocker: MagicMock) -> None:
        # 1. Setup mocks
        doc_store = MagicMock()
        doc_store.save_document = AsyncMock()
        doc_store.find_by_hash = AsyncMock(return_value=None)
        
        # Mock parser to return a payload
        from openrag.parsers.base import BaseParserAdapter
        mock_parser_instance = MagicMock(spec=BaseParserAdapter)
        mock_parser_instance.parse = AsyncMock(return_value=ContentPayload(
            document_id="h1", source_path="f.txt", tenant_id="t1", 
            metadata=DocumentMeta(), blocks=[ContentBlock("b1", "h1", BlockType.TEXT, 0, "txt")]
        ))
        
        # Mock registry
        registry = MagicMock()
        registry.get_parser.return_value = lambda: mock_parser_instance
        
        # Mock DAG Engine
        dag_engine = MagicMock(spec=DAGPipelineEngine)
        dag_engine.execute = AsyncMock(return_value=[
            ProcessedBlock(
                source_block=ContentBlock("b1", "h1", BlockType.TEXT, 0, "txt"),
                natural_language_description="desc",
                embedding_text="emb",
                entity_candidates=[],
            )
        ])
        
        # 2. Phase 3 engines
        emb_engine = MagicMock(spec=EmbeddingEngine)
        emb_engine.embed_blocks = AsyncMock()
        
        kg_builder = MagicMock(spec=KnowledgeGraphBuilder)
        kg_builder.build_from_blocks = AsyncMock()
        
        bm25_indexer = MagicMock(spec=BM25Indexer)
        bm25_indexer.index_blocks = AsyncMock()

        vector_db = MagicMock()
        vector_db.upsert = AsyncMock()
        
        # 3. Instantiate Orchestrator
        orchestrator = IngestionOrchestrator(
            config=OpenRAGConfig(),
            registry=registry,
            dag_engine=dag_engine,
            doc_store=doc_store,
            vector_db=vector_db,
            embedding_engine=emb_engine,
            kg_builder=kg_builder,
            bm25_indexer=bm25_indexer,
        )
        
        # Mock compute_file_hash
        mocker.patch("openrag.ingestion.orchestrator.compute_file_hash", return_value="h1")
        
        # 4. Ingest
        from pathlib import Path
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".txt") as tmp:
            tmp.write(b"hell")
            tmp.flush()
            result = await orchestrator.ingest_file(tmp.name, IngestMetadata(tenant_id="t1", namespace="ns1"))
        
        # 5. Verify Parallel Calls
        if result.status == JobStatus.FAILED:
            print(f"\nDEBUG: result.reason = {result.reason}")
        assert result.status == JobStatus.COMPLETED
        assert doc_store.save_document.call_count == 1
        assert emb_engine.embed_blocks.call_count == 1
        assert kg_builder.build_from_blocks.call_count == 1
        assert bm25_indexer.index_blocks.call_count == 1
