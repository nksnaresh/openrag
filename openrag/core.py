"""
OpenRAG core façade — the primary user-facing class.

Usage::

    import asyncio
    from openrag import OpenRAG, OpenRAGConfig

    async def main():
        rag = OpenRAG(config=OpenRAGConfig(namespace="demo"))
        await rag.initialize()

        # Ingest
        await rag.ingest("report.pdf")
        await rag.ingest_folder("./docs/")

        # Query
        result = await rag.query("What are the main findings?")
        print(result.answer)
        for cite in result.citations:
            print(f"  [{cite.document_title}, p.{cite.page_number}]")

        await rag.close()

    asyncio.run(main())
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable
from pathlib import Path
from typing import Any

from openrag.config import OpenRAGConfig
from openrag.models.jobs import BatchJobResult, BatchOptions, IngestMetadata, JobResult
from openrag.models.query import QueryMode, QueryRequest, QueryResponse
from openrag.registry import AdapterRegistry


class OpenRAG:
    """
    Primary entry point for the OpenRAG framework.

    This class acts as a façade over the ingestion, retrieval, and query
    sub-systems. It wires together the configured adapters and exposes a
    clean async API.

    Args:
        config:         OpenRAGConfig instance (or will use defaults).
        llm_func:       Async callable — text LLM: async(prompt) -> str.
        vlm_func:       Async callable — vision LLM: async(prompt, image_b64) -> str.
        embedding_func: Async callable — embeddings: async(texts) -> list[list[float]].
    """

    def __init__(
        self,
        config: OpenRAGConfig | None = None,
        llm_func: Callable[..., Any] | None = None,
        vlm_func: Callable[..., Any] | None = None,
        embedding_func: Callable[..., Any] | None = None,
    ) -> None:
        self.config = config or OpenRAGConfig()
        self.llm_func = llm_func
        self.vlm_func = vlm_func
        self.embedding_func = embedding_func

        self._initialized = False
        self._orchestrator: Any = None
        self._query_engine: Any = None

    async def initialize(self) -> None:
        """
        Initialize all storage adapters, processors, and engines.

        Must be called before any ingest() or query() calls.
        Can be used as an async context manager instead::

            async with OpenRAG(config=cfg) as rag:
                await rag.ingest("file.pdf")
        """
        if self._initialized:
            return

        self.config.ensure_working_dir()

        # Storage adapters (implementations added in Phase 1)
        vector_db_cls = AdapterRegistry.get_vector_db(self.config.vector_db.adapter)
        graph_db_cls = AdapterRegistry.get_graph_db(self.config.graph_db.adapter)
        doc_store_cls = AdapterRegistry.get_doc_store(self.config.document_store.adapter)

        self._vector_db = vector_db_cls(self.config.vector_db)
        self._graph_db = graph_db_cls(self.config.graph_db)
        self._doc_store = doc_store_cls(self.config.document_store)

        await self._vector_db.initialize()
        await self._graph_db.initialize()
        await self._doc_store.initialize()

        self._initialized = True

    async def close(self) -> None:
        """Release all resources and flush pending writes."""
        self._initialized = False

    async def __aenter__(self) -> OpenRAG:
        await self.initialize()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    # ── Ingestion ──────────────────────────────────────────────────────────────

    async def ingest(
        self,
        path: str | Path,
        namespace: str | None = None,
        tags: list[str] | None = None,
        acl: dict[str, list[str]] | None = None,
    ) -> JobResult:
        """Ingest a single file or URL into the knowledge base."""
        self._require_initialized()

        meta = IngestMetadata(
            tenant_id=self.config.tenant_id,
            namespace=namespace or self.config.namespace,
            tags=tags or [],
        )
        orch = self._get_orchestrator()
        return await orch.ingest_file(str(path), meta, acl=acl)  # type: ignore[no-any-return]

    async def ingest_folder(
        self,
        folder: str | Path,
        namespace: str | None = None,
        max_workers: int = 4,
        recursive: bool = True,
    ) -> BatchJobResult:
        """Batch-ingest all supported files in a folder."""
        self._require_initialized()
        meta = IngestMetadata(
            tenant_id=self.config.tenant_id,
            namespace=namespace or self.config.namespace,
        )
        options = BatchOptions(metadata=meta, max_workers=max_workers, recursive=recursive)
        orch = self._get_orchestrator()
        paths = self._collect_files(Path(folder), options)
        return await orch.ingest_batch(paths, options)  # type: ignore[no-any-return]

    # ── Query ──────────────────────────────────────────────────────────────────

    async def query(
        self,
        text: str,
        mode: QueryMode | str = QueryMode.HYBRID,
        namespace: str | None = None,
        top_k: int = 20,
        output_schema: dict[str, Any] | None = None,
        session_id: str | None = None,
    ) -> QueryResponse:
        """Execute a query and return the synthesised answer with citations."""
        self._require_initialized()
        request = QueryRequest(
            text=text,
            mode=QueryMode(mode) if isinstance(mode, str) else mode,
            namespace=namespace or self.config.namespace,
            top_k=top_k,
            output_schema=output_schema,
            session_id=session_id,
        )
        engine = self._get_query_engine()
        return await engine.execute(request)  # type: ignore[no-any-return]

    async def stream_query(
        self,
        text: str,
        mode: QueryMode | str = QueryMode.HYBRID,
        namespace: str | None = None,
    ) -> AsyncGenerator[str, None]:
        """Execute a query and yield answer tokens incrementally."""
        self._require_initialized()
        request = QueryRequest(
            text=text,
            mode=QueryMode(mode) if isinstance(mode, str) else mode,
            namespace=namespace or self.config.namespace,
            stream=True,
        )
        engine = self._get_query_engine()
        async for token in engine.stream(request):
            yield token

    # ── Internals ──────────────────────────────────────────────────────────────

    def _require_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError(
                "OpenRAG is not initialized. Call `await rag.initialize()` "
                "or use it as an async context manager."
            )

    def _get_orchestrator(self) -> Any:  # noqa: ANN401  # returns IngestionOrchestrator (Phase 1)
        if self._orchestrator is None:
            from openrag.ingestion.orchestrator import IngestionOrchestrator
            self._orchestrator = IngestionOrchestrator(
                config=self.config,
                vector_db=self._vector_db,
                graph_db=self._graph_db,
                doc_store=self._doc_store,
                llm_func=self.llm_func,
                vlm_func=self.vlm_func,
                embedding_func=self.embedding_func,
            )
        return self._orchestrator

    def _get_query_engine(self) -> Any:  # noqa: ANN401  # returns QueryOrchestrator (Phase 1)
        if self._query_engine is None:
            from openrag.query.orchestrator import QueryOrchestrator
            self._query_engine = QueryOrchestrator(
                config=self.config,
                vector_db=self._vector_db,
                graph_db=self._graph_db,
                doc_store=self._doc_store,
                llm_func=self.llm_func,
                vlm_func=self.vlm_func,
            )
        return self._query_engine

    @staticmethod
    def _collect_files(folder: Path, options: BatchOptions) -> list[str]:
        exts = set(options.supported_extensions)
        pattern = "**/*" if options.recursive else "*"
        return [
            str(p) for p in folder.glob(pattern)
            if p.is_file() and p.suffix.lower() in exts
        ]
