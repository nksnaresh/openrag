"""BM25 Retriever — lexical search using Rank-BM25.

Loads tokenized content from the document store and performs ranking
using the BM25 algorithm.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

if TYPE_CHECKING:
    from openrag.storage.base import BaseDocumentAdapter


class BM25Retriever:
    """Retrieves ranked results using lexical BM25 matching.

    Args:
        doc_store: Initialized document store adapter.
    """

    def __init__(self, doc_store: BaseDocumentAdapter) -> None:
        self._doc_store = doc_store
        self._index_cache: dict[str, BM25Okapi] = {}
        self._id_map_cache: dict[str, list[str]] = {}

    async def search(
        self, query: str, namespace: str, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Perform lexical search for a query in a namespace.

        This implementation rebuilds/loads the index for the namespace if not cached.
        """
        if BM25Okapi is None:
            return []

        # 1. Load/Build index for namespace
        bm25, ids = await self._get_index(namespace)
        if not bm25 or not ids:
            return []

        # 2. Tokenize query (using same logic as indexer)
        # Simple split for now, should match BM25Indexer.tokenize
        tokenized_query = query.lower().split() 

        # 3. Get scores
        scores = bm25.get_scores(tokenized_query)
        
        # 4. Rank and format
        results = []
        for idx, score in enumerate(scores):
            if score > 0:
                results.append({
                    "id": ids[idx],
                    "score": float(score),
                    "namespace": namespace,
                    "mode": "bm25"
                })
        
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]

    async def _get_index(self, namespace: str) -> tuple[BM25Okapi | None, list[str]]:
        """Retrieve the BM25 index and ID map for a namespace."""
        if namespace in self._index_cache:
            return self._index_cache[namespace], self._id_map_cache[namespace]

        # Fetch all tokens for this namespace
        token_pairs = await self._doc_store.get_all_bm25_tokens(namespace)
        if not token_pairs:
            return None, []

        ids = [p[0] for p in token_pairs]
        corpus = [p[1] for p in token_pairs]

        bm25 = BM25Okapi(corpus)
        
        # Cache it (In production, this should have an expiry or use a persistent FTS5 index)
        self._index_cache[namespace] = bm25
        self._id_map_cache[namespace] = ids
        
        return bm25, ids
