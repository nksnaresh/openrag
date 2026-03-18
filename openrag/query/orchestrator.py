"""Query Orchestrator — end-to-end coordinator for the RAG query flow.

Wires together: Query Processing → Hybrid Retrieval → Reranking →
Synthesis (LLM).
"""

from __future__ import annotations

import anyio
from typing import TYPE_CHECKING, Any

from openrag.models.query import QueryResponse, Citation

if TYPE_CHECKING:
    from openrag.config import OpenRAGConfig
    from openrag.models.query import QueryRequest
    from openrag.query.query_processor import QueryProcessor
    from openrag.search.hybrid_search import HybridSearcher
    from openrag.search.reranker import BaseReranker
    from collections.abc import Callable


class QueryOrchestrator:
    """High-level coordinator for querying the knowledge base.

    Args:
        config:           OpenRAGConfig instance.
        hybrid_searcher:  Initialized HybridSearcher.
        reranker:         Optional reranker (Cohere/HF).
        query_processor:  Optional query processor (HyDE/Multi-Query).
        llm_func:         Async callable for synthesis: async(prompt) -> str.
    """

    def __init__(
        self,
        config: OpenRAGConfig,
        hybrid_searcher: HybridSearcher,
        reranker: BaseReranker | None = None,
        query_processor: QueryProcessor | None = None,
        llm_func: Callable[..., Any] | None = None,
    ) -> None:
        self._config = config
        self._hybrid_searcher = hybrid_searcher
        self._reranker = reranker
        self._query_processor = query_processor
        self._llm_func = llm_func

    async def execute(self, request: QueryRequest) -> QueryResponse:
        """Execute a full RAG query flow."""
        
        # 1. Query Processing (Expansion/HyDE)
        search_queries = [request.text]
        if self._query_processor:
            # For simplicity, we'll just use Multi-Query for now
            variations = await self._query_processor.generate_multi_queries(request.text)
            search_queries.extend(variations)
            # Deduplicate
            search_queries = list(dict.fromkeys(search_queries))

        # 2. Hybrid Retrieval (Concurrent for all queries)
        all_hits = []
        async with anyio.create_task_group() as tg:
            for q in search_queries:
                tg.start_soon(self._collect_hits, q, request.namespace, request.top_k, all_hits)

        # 3. Fusion of multi-query results (RRF)
        # The hybrid_searcher already does RRF for one query. 
        # Here we do it again across multiple search queries if needed.
        fused_hits = self._hybrid_searcher.reciprocal_rank_fusion([all_hits])
        
        # 4. Reranking
        final_hits = fused_hits
        if self._reranker and fused_hits:
            final_hits = await self._reranker.rerank(
                query=request.text,
                documents=fused_hits,
                top_k=request.top_k
            )

        # 5. Synthesis (LLM)
        if not self._llm_func:
            return QueryResponse(
                answer="Synthesis disabled (no LLM provided).",
                citations=self._build_citations(final_hits),
                query_mode=request.mode,
                latency_ms=0.0  # TODO: measure latency
            )

        context_text = self._build_context(final_hits)
        prompt = self._build_synthesis_prompt(request.text, context_text)
        answer = await self._llm_func(prompt)

        return QueryResponse(
            answer=answer,
            citations=self._build_citations(final_hits),
            query_mode=request.mode,
            latency_ms=0.0
        )

    async def _collect_hits(
        self, query: str, namespace: str, top_k: int, out: list[dict[str, Any]]
    ) -> None:
        hits = await self._hybrid_searcher.search(query, namespace, top_k)
        out.extend(hits)

    def _build_context(self, hits: list[dict[str, Any]]) -> str:
        """Combine hits into a single context string for the LLM."""
        sections = []
        for i, hit in enumerate(hits, start=1):
            content = hit.get("content") or hit.get("text") or ""
            source = hit.get("source_path") or "Unknown"
            sections.append(f"Source [{i}] ({source}):\n{content}")
        return "\n\n".join(sections)

    def _build_synthesis_prompt(self, query: str, context: str) -> str:
        """Construct the final prompt for the LLM."""
        return (
            f"You are a helpful and accurate AI assistant for OpenRAG. "
            f"Answer the user's question using ONLY the provided context blocks. "
            f"If the context does not contain the answer, say that you don't know. "
            f"Cite your sources using [number] notation corresponding to the context blocks.\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION: {query}\n\n"
            f"ANSWER:"
        )

    def _build_citations(self, hits: list[dict[str, Any]]) -> list[Citation]:
        """Convert hits into Citation models."""
        from openrag.models.content import BlockType
        citations = []
        for hit in hits:
            citations.append(Citation(
                document_id=hit.get("document_id", ""),
                document_title=hit.get("title") or hit.get("source_path", "Untitled"),
                page_number=hit.get("page_number"),
                block_id=hit.get("block_id") or hit.get("id", ""),
                block_type=hit.get("block_type") or BlockType.TEXT,
                score=hit.get("rrf_score") or hit.get("rerank_score") or 0.0,
                excerpt=hit.get("content") or hit.get("text", "")[:200]
            ))
        return citations
