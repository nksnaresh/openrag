"""Unit tests for the IngestionOrchestrator."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from openrag.config import OpenRAGConfig
from openrag.ingestion.orchestrator import IngestionOrchestrator
from openrag.models.content import BlockType, ContentBlock, ContentPayload, DocumentMeta
from openrag.models.jobs import IngestMetadata, JobStatus
from openrag.models.processing import ProcessedBlock, ProcessingContext
from openrag.parsers.base import BaseParserAdapter, ParseOptions
from openrag.pipeline.dag_engine import DAGPipelineEngine, PipelineStage
from openrag.processors.base import BaseModalityProcessor
from openrag.registry import AdapterRegistry
from openrag.storage.document.sqlite import SQLiteDocumentAdapter

# ── Mocks / Helpers ──────────────────────────────────────────────────────────

class MockParser(BaseParserAdapter):
    async def parse(self, file_path: str | Path, options: ParseOptions) -> ContentPayload:
        import hashlib
        path = Path(file_path)
        content_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        return ContentPayload(
            document_id=content_hash,
            source_path=str(file_path),
            tenant_id=options.tenant_id,
            metadata=DocumentMeta(),
            blocks=[
                ContentBlock(
                    document_id=content_hash,
                    block_id=f"{content_hash[:8]}-0",
                    block_type=BlockType.TEXT,
                    sequence_index=0,
                    raw_content="Mock parsed text",
                )
            ],
        )

    def supported_extensions(self) -> list[str]:
        return [".mock"]


class MockProcessor(BaseModalityProcessor):
    async def process(self, block: ContentBlock, context: ProcessingContext) -> ProcessedBlock:
        return ProcessedBlock(
            source_block=block,
            natural_language_description="Processed description",
            embedding_text="Processed text",
        )

    def supported_block_types(self) -> list[BlockType]:
        return [BlockType.TEXT]


@pytest.fixture()
def registry() -> AdapterRegistry:
    reg = AdapterRegistry()
    reg.register_parser(".mock", MockParser)
    return reg


@pytest.fixture()
def dag_engine() -> DAGPipelineEngine:
    stages = [
        PipelineStage("text", MockProcessor(), block_types=[BlockType.TEXT])
    ]
    return DAGPipelineEngine(stages, OpenRAGConfig())


@pytest.fixture()
def doc_store(tmp_path: Path) -> SQLiteDocumentAdapter:
    class _Cfg:
        url = f"sqlite+aiosqlite:///{tmp_path}/orchestrator_test.db"
    return SQLiteDocumentAdapter(_Cfg())


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestIngestionOrchestrator:
    @pytest.mark.asyncio
    async def test_ingest_file_success(
        self,
        registry: AdapterRegistry,
        dag_engine: DAGPipelineEngine,
        doc_store: SQLiteDocumentAdapter,
    ) -> None:
        await doc_store.initialize()
        orchestrator = IngestionOrchestrator(OpenRAGConfig(), registry, dag_engine, doc_store)

        with tempfile.NamedTemporaryFile(suffix=".mock", delete=False) as tmp:
            tmp.write(b"content")
            tmp_path = tmp.name

        metadata = IngestMetadata(tenant_id="t1", namespace="ns1")
        result = await orchestrator.ingest_file(tmp_path, metadata)

        assert result.status == JobStatus.COMPLETED
        assert result.block_count == 1
        assert result.document_id is not None

        # Verify persistence
        doc = await doc_store.get_document(result.document_id, "ns1")
        assert doc is not None

    @pytest.mark.asyncio
    async def test_ingest_file_duplicate_skipped(
        self,
        registry: AdapterRegistry,
        dag_engine: DAGPipelineEngine,
        doc_store: SQLiteDocumentAdapter,
    ) -> None:
        await doc_store.initialize()
        orchestrator = IngestionOrchestrator(OpenRAGConfig(), registry, dag_engine, doc_store)

        with tempfile.NamedTemporaryFile(suffix=".mock", delete=False) as tmp:
            tmp.write(b"content")
            tmp_path = tmp.name

        metadata = IngestMetadata(tenant_id="t1", namespace="ns1")

        # First ingest
        await orchestrator.ingest_file(tmp_path, metadata)

        # Second ingest of same file
        result = await orchestrator.ingest_file(tmp_path, metadata)
        assert result.status == JobStatus.SKIPPED
        assert "Duplicate" in (result.reason or "")

    @pytest.mark.asyncio
    async def test_ingest_file_unsupported_extension_fails(
        self,
        registry: AdapterRegistry,
        dag_engine: DAGPipelineEngine,
        doc_store: SQLiteDocumentAdapter,
    ) -> None:
        await doc_store.initialize()
        orchestrator = IngestionOrchestrator(OpenRAGConfig(), registry, dag_engine, doc_store)

        with tempfile.NamedTemporaryFile(suffix=".unknown", delete=False) as tmp:
            tmp.write(b"content")
            tmp_path = tmp.name

        metadata = IngestMetadata(tenant_id="t1", namespace="ns1")
        result = await orchestrator.ingest_file(tmp_path, metadata)

        assert result.status == JobStatus.FAILED
        assert "No parser registered" in (result.reason or "")

    @pytest.mark.asyncio
    async def test_ingest_batch_success(
        self,
        registry: AdapterRegistry,
        dag_engine: DAGPipelineEngine,
        doc_store: SQLiteDocumentAdapter,
    ) -> None:
        from openrag.models.jobs import BatchOptions
        await doc_store.initialize()
        orchestrator = IngestionOrchestrator(OpenRAGConfig(), registry, dag_engine, doc_store)

        paths = []
        for i in range(3):
            with tempfile.NamedTemporaryFile(suffix=".mock", delete=False) as tmp:
                tmp.write(f"content {i}".encode())
                paths.append(tmp.name)

                options = BatchOptions(
            metadata=IngestMetadata(tenant_id="t1", namespace="ns1"),
            max_workers=2,
        )
        batch_result = await orchestrator.ingest_batch(paths, options)

        assert batch_result.total == 3
        assert len(batch_result.successful) == 3
        assert batch_result.failed == []
        assert batch_result.skipped == []
