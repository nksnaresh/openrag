"""Hybrid Search Orchestration — combining Vector, KG, and Lexical results.

Uses Reciprocal Rank Fusion (RRF) to merge ranked lists from different
retrieval sub-systems into a single unified result set.
"""

from __future__ import annotations

import anyio
from typing import TYPE_CHECKING, Any

from openrag.search.bm25_retriever import BM25Retriever

if TYPE_CHECKING:
    from openrag.embeddings.engine import EmbeddingEngine
    from openrag.storage.base import (
        BaseDocumentAdapter,
        BaseGraphDBAdapter,
        BaseVectorDBAdapter,
    )


class HybridSearcher:
    """Orchestrates multi-modal retrieval and result fusion.

    Args:
        vector_db:        Initialized vector database adapter.
        graph_db:         Initialized graph database adapter.
        doc_store:        Initialized document store adapter (for BM25).
        embedding_engine: Engine for converting query text to vectors.
        rrf_k:            Smoothing constant for RRF (default 60).
    """

    def __init__(
        self,
        vector_db: BaseVectorDBAdapter,
        graph_db: BaseGraphDBAdapter,
        doc_store: BaseDocumentAdapter,
        embedding_engine: EmbeddingEngine,
        rrf_k: int = 60,
    ) -> None:
        self._vector_db = vector_db
        self._graph_db = graph_db
        self._doc_store = doc_store
        self._embedding_engine = embedding_engine
        self._rrf_k = rrf_k
        self._bm25_retriever = BM25Retriever(doc_store)

    async def search(
        self,
        query: str,
        namespace: str,
        top_k: int = 20,
        weights: dict[str, float] | None = None,
    ) -> list[dict[str, Any]]:
        """Perform hybrid search across all available indices.

        Args:
            query:     User's search query string.
            namespace: The namespace to search within.
            top_k:     Number of final fused results to return.
            weights:   Optional multipliers for each mode (vector, graph, bm25).

        Returns:
            List of fused result dictionaries with 'content', 'id', and 'score'.
        """
        results_vector: list[dict[str, Any]] = []
        results_graph: list[dict[str, Any]] = []
        results_bm25: list[dict[str, Any]] = []

        async with anyio.create_task_group() as tg:
            # 1. Vector Search
            tg.start_soon(self._fetch_vector, query, namespace, top_k * 2, results_vector)
            
            # 2. Graph Search
            tg.start_soon(self._fetch_graph, query, namespace, top_k * 2, results_graph)
            
            # 3. BM25 Search
            tg.start_soon(self._fetch_bm25, query, namespace, top_k * 2, results_bm25)

        # DEBUG: print hit counts
        print(f"DEBUG: Vector hits: {len(results_vector)}")
        print(f"DEBUG: Graph hits: {len(results_graph)}")
        print(f"DEBUG: BM25 hits: {len(results_bm25)}")

        # 4. Fusion (RRF)
        fused = self.reciprocal_rank_fusion(
            [results_vector, results_graph, results_bm25],
            k=self._rrf_k
        )

        return fused[:top_k]

    async def _fetch_vector(
        self, query: str, namespace: str, limit: int, out: list[dict[str, Any]]
    ) -> None:
        """Call vector DB for semantic results."""
        vector = await self._embedding_engine.embed_query(query)
        if vector is None:
            return
        search_results = await self._vector_db.query(namespace, vector, top_k=limit)
        
        hits = []
        for res in search_results:
            hit = {
                "id": res.id,
                "score": float(res.score),
                "mode": "vector",
                "namespace": namespace
            }
            if res.payload:
                hit.update(res.payload)
            hits.append(hit)
        out.extend(hits)

    async def _fetch_graph(
        self, query: str, namespace: str, limit: int, out: list[dict[str, Any]]
    ) -> None:
        """Query the graph for entity-relevant context."""
        # Simple implementation: find entities in query, then find their mentions
        # In later phases, this will be more sophisticated (e.g., GraphRAG)
        hits = await self._graph_db.search_context(query, namespace, limit=limit)
        out.extend(hits)

    async def _fetch_bm25(
        self, query: str, namespace: str, limit: int, out: list[dict[str, Any]]
    ) -> None:
        """Lexical search using stored tokens."""
        hits = await self._bm25_retriever.search(query, namespace, limit=limit)
        
        # Hydrate missing BM25 payloads using direct vector storage lookup
        for hit in hits:
            if hasattr(self._vector_db, "get_payload"):
                payload = await self._vector_db.get_payload(namespace, hit["id"])
                if payload:
                    hit.update(payload)
                    
        out.extend(hits)

    def reciprocal_rank_fusion(
        self,
        rankings: list[list[dict[str, Any]]],
        k: int = 60
    ) -> list[dict[str, Any]]:
        """Combine multiple ranked lists into one using RRF."""
        scores: dict[str, float] = {}
        metadata: dict[str, dict[str, Any]] = {}

        for ranking in rankings:
            for rank, item in enumerate(ranking, start=1):
                doc_id = item.get("id") or item.get("chunk_id")
                if not doc_id:
                    continue
                
                if doc_id not in scores:
                    scores[doc_id] = 0.0
                    metadata[doc_id] = item
                
                scores[doc_id] += 1.0 / (k + rank)

        # Sort by fused score descending
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        fused_results = []
        for doc_id in sorted_ids:
            item = metadata[doc_id].copy()
            item["rrf_score"] = scores[doc_id]
            fused_results.append(item)
            
        return fused_results
