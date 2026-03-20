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

        self._vector_db = vector_db_cls(self.config.vector_db, working_dir=self.config.working_dir)
        self._graph_db = graph_db_cls(self.config.graph_db)
        self._doc_store = doc_store_cls(self.config.document_store)

        await self._vector_db.initialize()
        await self._graph_db.initialize()
        await self._doc_store.initialize()

        self._initialized = True

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def _require_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError("OpenRAG not initialized. Call await rag.initialize() first.")

    async def __aenter__(self) -> OpenRAG:
        await self.initialize()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    async def close(self) -> None:
        """Close all storage connections."""
        if not self._initialized:
            return
        await self._vector_db.close()
        await self._graph_db.close()
        await self._doc_store.close()
        self._initialized = False

    # ── Ingestion ──────────────────────────────────────────────────────────────

    async def ingest(
        self,
        path: str | Path,
        metadata: IngestMetadata | None = None,
        acl: dict[str, list[str]] | None = None,
        on_progress: Callable[[str], Any] | None = None,
    ) -> JobResult:
        """Parse and ingest a file into the knowledge base."""
        self._require_initialized()
        orchestrator = self._get_orchestrator()
        metadata = metadata or IngestMetadata(
            tenant_id=self.config.tenant_id,
            namespace=self.config.namespace
        )
        return await orchestrator.ingest_file(path, metadata, acl, on_progress=on_progress)

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
        on_progress: Callable[[str], Any] | None = None,
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
        return await engine.execute(request, on_progress=on_progress)  # type: ignore[no-any-return]

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

    def _get_orchestrator(self) -> Any:  # noqa: ANN401  # returns IngestionOrchestrator
        if self._orchestrator is None:
            from openrag.embeddings.cache import InMemoryEmbeddingCache
            from openrag.embeddings.engine import EmbeddingEngine
            from openrag.ingestion.orchestrator import IngestionOrchestrator
            from openrag.knowledge.graph_builder import KnowledgeGraphBuilder
            from openrag.pipeline.dag_engine import build_default_pipeline
            from openrag.processors.text_processor import TextProcessor
            from openrag.processors.code_processor import CodeProcessor
            from openrag.processors.image_processor import ImageProcessor
            from openrag.processors.table_processor import TableProcessor
            from openrag.processors.equation_processor import EquationProcessor
            from openrag.search.bm25_indexer import BM25Indexer

            processors = {
                'text': TextProcessor(),
                'code': CodeProcessor(),
                'image': ImageProcessor(),
                'table': TableProcessor(),
                'equation': EquationProcessor(),
            }
            dag_engine = build_default_pipeline(self.config, processors=processors)
            registry = AdapterRegistry()

            # Initialize Phase 3 indexing engines
            emb_cls = registry.get_embedding(self.config.embedding.provider)
            embedding_engine = EmbeddingEngine(
                adapter=emb_cls(self.config.embedding),
                cache=InMemoryEmbeddingCache(),
                embedding_func=self.embedding_func,
            )
            kg_builder = KnowledgeGraphBuilder(graph_db=self._graph_db)
            bm25_indexer = BM25Indexer(doc_store=self._doc_store)

            self._orchestrator = IngestionOrchestrator(
                config=self.config,
                registry=registry,
                dag_engine=dag_engine,
                doc_store=self._doc_store,
                vector_db=self._vector_db,
                embedding_engine=embedding_engine,
                kg_builder=kg_builder,
                bm25_indexer=bm25_indexer,
                llm_func=self.llm_func,
                vlm_func=self.vlm_func,
            )
        return self._orchestrator

    def _get_query_engine(self) -> Any:  # noqa: ANN401  # returns QueryOrchestrator
        if self._query_engine is None:
            from openrag.embeddings.cache import InMemoryEmbeddingCache
            from openrag.embeddings.engine import EmbeddingEngine
            from openrag.query.orchestrator import QueryOrchestrator
            from openrag.query.query_processor import QueryProcessor
            from openrag.search.hybrid_search import HybridSearcher
            from openrag.search.reranker import CohereReranker, HuggingFaceReranker

            registry = AdapterRegistry()
            
            # 1. Embedding Engine (shared with orchestrator)
            emb_cls = registry.get_embedding(self.config.embedding.provider)
            embedding_engine = EmbeddingEngine(
                adapter=emb_cls(self.config.embedding),
                cache=InMemoryEmbeddingCache(),
                embedding_func=self.embedding_func,
            )

            # 2. Hybrid Searcher
            hybrid_searcher = HybridSearcher(
                vector_db=self._vector_db,
                graph_db=self._graph_db,
                doc_store=self._doc_store,
                embedding_engine=embedding_engine,
            )

            # 3. Reranker (optional)
            reranker = None
            if self.config.rerank.enabled:
                if self.config.rerank.adapter == "cohere":
                    reranker = CohereReranker(model=self.config.rerank.model)
                else:
                    reranker = HuggingFaceReranker(model_name=self.config.rerank.model)

            # 4. Query Processor
            query_processor = None
            if self.llm_func:
                query_processor = QueryProcessor(llm_func=self.llm_func)

            self._query_engine = QueryOrchestrator(
                config=self.config,
                hybrid_searcher=hybrid_searcher,
                reranker=reranker,
                query_processor=query_processor,
                llm_func=self.llm_func,
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
