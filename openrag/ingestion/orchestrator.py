from __future__ import annotations

import traceback
"""Ingestion orchestrator — top-level coordinator for file ingestion.

Wires together: deduplication → parser selection → DAG pipeline →
document storage.
"""

from pathlib import Path

import anyio

from openrag.config import OpenRAGConfig
from openrag.embeddings.engine import EmbeddingEngine
from openrag.ingestion.deduplicator import compute_file_hash, is_duplicate
from openrag.knowledge.graph_builder import KnowledgeGraphBuilder
from openrag.models.content import ContentPayload
from openrag.models.jobs import (
    BatchJobResult,
    BatchOptions,
    IngestMetadata,
    JobResult,
    JobStatus,
)
from openrag.models.processing import ContextWindowConfig, ProcessingContext
from openrag.parsers.base import ParseOptions
from openrag.pipeline.dag_engine import DAGPipelineEngine
from openrag.registry import AdapterRegistry, RegistryError
from openrag.search.bm25_indexer import BM25Indexer
from openrag.storage.base import BaseDocumentAdapter, BaseVectorDBAdapter
from openrag.observability.tracing import get_tracer
from openrag.observability.metrics import MetricsManager
import time

tracer = get_tracer(__name__)


class IngestionOrchestrator:
    """High-level coordinator that drives the full ingest pipeline.

    Args:
        config:     OpenRAGConfig instance (for pipeline defaults).
        registry:   AdapterRegistry with registered parsers and processors.
        dag_engine: Configured DAGPipelineEngine ready to execute.
        doc_store:  Initialized BaseDocumentAdapter for persistence.
    """

    def __init__(
        self,
        config: OpenRAGConfig,
        registry: AdapterRegistry,
        dag_engine: DAGPipelineEngine,
        doc_store: BaseDocumentAdapter,
        vector_db: BaseVectorDBAdapter | None = None,
        embedding_engine: EmbeddingEngine | None = None,
        kg_builder: KnowledgeGraphBuilder | None = None,
        bm25_indexer: BM25Indexer | None = None,
    ) -> None:
        self._config = config
        self._registry = registry
        self._dag_engine = dag_engine
        self._doc_store = doc_store
        self._vector_db = vector_db
        self._embedding_engine = embedding_engine
        self._kg_builder = kg_builder
        self._bm25_indexer = bm25_indexer

    async def ingest_file(
        self,
        path: str | Path,
        metadata: IngestMetadata,
        acl: dict[str, list[str]] | None = None,
    ) -> JobResult:
        start_time = time.time()
        path = Path(path)
        with tracer.start_as_current_span("ingest.file") as span:
            span.set_attribute("openrag.file_path", str(path))
            span.set_attribute("openrag.namespace", metadata.namespace)

        # 1. Hash + dedup
        try:
            content_hash = await compute_file_hash(path)
        except OSError as exc:
            return JobResult(
                job_id="",
                status=JobStatus.FAILED,
                document_id=None,
                reason=f"Cannot read file: {exc}",
            )

        # Only skip if the document exists in doc_store AND vector DB has data.
        # After a server restart the in-memory vector DB is wiped, so we must
        # re-ingest even if the doc_store still has the hash recorded.
        is_dup = await is_duplicate(content_hash, metadata.namespace, self._doc_store)
        if is_dup and self._vector_db is not None:
            vector_count = await self._vector_db.count(metadata.namespace)
            if vector_count == 0:
                is_dup = False  # Vector DB is empty — force re-ingest
        if is_dup:
            return JobResult(
                job_id=content_hash,
                status=JobStatus.SKIPPED,
                document_id=content_hash,
                reason="Duplicate document — already ingested.",
            )

        # 2. Parser selection
        ext = path.suffix.lower()
        try:
            parser_cls = self._registry.get_parser(ext)
        except (KeyError, RegistryError):
            return JobResult(
                job_id=content_hash,
                status=JobStatus.FAILED,
                document_id=content_hash,
                reason=f"No parser registered for extension '{ext}'.",
            )
        parser = parser_cls()

        # 3. Parse
        try:
            options = ParseOptions(
                tenant_id=metadata.tenant_id,
                namespace=metadata.namespace,
            )
            from openrag.parsers.base import BaseParserAdapter  # noqa: PLC0415
            assert isinstance(parser, BaseParserAdapter)
            payload: ContentPayload = await parser.parse(path, options)
        except Exception as exc:  # noqa: BLE001
            return JobResult(
                job_id=content_hash,
                status=JobStatus.FAILED,
                document_id=content_hash,
                reason=f'Parser error: {exc} (Path: {path})',
            )

        # 4. Build processing context
        ctx_config = ContextWindowConfig(
            strategy=self._config.context.strategy,
            window_size=self._config.context.window_size,
            max_tokens=self._config.context.max_tokens,
        )
        context = ProcessingContext(
            payload=payload,
            llm_func=None,  # type: ignore[arg-type]
            vlm_func=None,
            context_config=ctx_config,
            tenant_id=metadata.tenant_id,
            namespace=metadata.namespace,
        )

        # 5. DAG pipeline
        try:
            processed_blocks = await self._dag_engine.execute(payload, context)
        except Exception as exc:  # noqa: BLE001
            return JobResult(
                job_id=content_hash,
                status=JobStatus.FAILED,
                document_id=content_hash,
                reason=f"Pipeline error: {exc}",
            )

        # 6. Persist results (Parallel Indexing)
        try:
            async with anyio.create_task_group() as tg:
                # A. Document record (primary)
                tg.start_soon(self._doc_store.save_document, payload, metadata.namespace)
                
                # B. Embedding indexing
                if self._embedding_engine:
                    await self._embedding_engine.embed_blocks(processed_blocks)
                    # Sync blocks to Vector DB
                    from openrag.storage.base import VectorRecord
                    records = [
                        VectorRecord(
                            id=b.block_id,
                            vector=b.embedding,
                            payload={
                                "content": b.content,
                                "document_id": b.document_id,
                                "source_path": str(path),
                                "block_type": b.block_type.value,
                                "page_number": b.page_number,
                                "index": b.index
                            }
                        )
                        for b in processed_blocks if b.embedding is not None
                    ]
                    tg.start_soon(self._vector_db.upsert, metadata.namespace, records)
                
                # C. Knowledge Graph indexing
                if self._kg_builder:
                    tg.start_soon(self._kg_builder.build_from_blocks, processed_blocks, payload, context)
                
                # D. BM25 indexing
                if self._bm25_indexer:
                    tg.start_soon(self._bm25_indexer.index_blocks, processed_blocks, metadata.namespace)

        except Exception as exc:  # noqa: BLE001
            return JobResult(
                job_id=content_hash,
                status=JobStatus.FAILED,
                document_id=content_hash,
                reason=f"Indexing error: {exc}\n{traceback.format_exc()}",
            )

        duration = time.time() - start_time
        MetricsManager.INGEST_DURATION.labels(
            tenant=metadata.tenant_id,
            status="completed"
        ).observe(duration)
        
        MetricsManager.INGEST_DOCS.labels(
            tenant=metadata.tenant_id,
            status="completed"
        ).inc()
        
        MetricsManager.INGEST_BLOCKS.labels(
            tenant=metadata.tenant_id,
            type="total"
        ).inc(len(processed_blocks))

        return JobResult(
            job_id=content_hash,
            status=JobStatus.COMPLETED,
            document_id=content_hash,
            block_count=len(processed_blocks),
        )

    async def ingest_batch(
        self,
        paths: list[str | Path],
        options: BatchOptions,
    ) -> BatchJobResult:
        """Ingest multiple files concurrently, bounded by max_workers.

        Args:
            paths:   List of file paths to ingest.
            options: BatchOptions with max_workers and shared IngestMetadata.

        Returns:
            BatchJobResult with per-file JobResults.
        """
        results: list[JobResult] = [
            JobResult(job_id="", status=JobStatus.PENDING)
        ] * len(paths)

        sem = anyio.Semaphore(options.max_workers)

        async def _bounded(index: int, file_path: str | Path) -> None:
            async with sem:
                results[index] = await self.ingest_file(file_path, options.metadata)

        async with anyio.create_task_group() as tg:
            for i, p in enumerate(paths):
                tg.start_soon(_bounded, i, p)

        successful = [r for r in results if r.status == JobStatus.COMPLETED]
        skipped_list = [r for r in results if r.status == JobStatus.SKIPPED]
        failed_list = [r for r in results if r.status == JobStatus.FAILED]

        return BatchJobResult(
            total=len(paths),
            successful=successful,
            skipped=skipped_list,
            failed=failed_list,
        )
