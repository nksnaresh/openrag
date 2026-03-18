"""Ingestion orchestrator — top-level coordinator for file ingestion.

Wires together: deduplication → parser selection → DAG pipeline →
document storage.
"""

from __future__ import annotations

from pathlib import Path

import anyio

from openrag.config import OpenRAGConfig
from openrag.ingestion.deduplicator import compute_file_hash, is_duplicate
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
from openrag.storage.base import BaseDocumentAdapter


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
    ) -> None:
        self._config = config
        self._registry = registry
        self._dag_engine = dag_engine
        self._doc_store = doc_store

    async def ingest_file(
        self,
        path: str | Path,
        metadata: IngestMetadata,
    ) -> JobResult:
        """Ingest a single file through the full pipeline.

        Steps:
            1. Compute SHA-256 hash → dedup check.
            2. Look up parser by file extension.
            3. Parse file → ContentPayload.
            4. Build ProcessingContext.
            5. Execute DAG pipeline → list[ProcessedBlock].
            6. Persist document record to doc store.
            7. Return JobResult.

        Returns:
            JobResult with ``status="completed"`` or ``status="skipped"``
            (duplicate) or ``status="error"`` on failure.
        """
        path = Path(path)

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

        if await is_duplicate(content_hash, metadata.namespace, self._doc_store):
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
                reason=f"Parser error: {exc}",
            )

        # 4. Build processing context (no LLM/VLM in Phase 2 base config)
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

        # 6. Persist document record
        try:
            await self._doc_store.save_document(payload, metadata.namespace)
        except Exception as exc:  # noqa: BLE001
            return JobResult(
                job_id=content_hash,
                status=JobStatus.FAILED,
                document_id=content_hash,
                reason=f"Storage error: {exc}",
            )

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
