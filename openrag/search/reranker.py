"""Reranking Engine — high-precision re-scoring of retrieval candidates.

Uses Cross-Encoders or managed services (Cohere) to re-evaluate the relevance
of the top-K hits from the hybrid search stage.
"""

from __future__ import annotations

import abc
import os
from typing import Any


class BaseReranker(abc.ABC):
    """Abstract base class for all reranker adapters."""

    @abc.abstractmethod
    async def rerank(
        self, query: str, documents: list[dict[str, Any]], top_k: int
    ) -> list[dict[str, Any]]:
        """Re-score and sort the provided documents based on the query."""


class CohereReranker(BaseReranker):
    """Managed reranking using Cohere's Rerank API.

    Requires COHERE_API_KEY in environment or config.
    """

    def __init__(self, api_key: str | None = None, model: str = "rerank-english-v3.0") -> None:
        self._api_key = api_key or os.getenv("COHERE_API_KEY")
        self._model = model
        self._client = None
        if self._api_key:
            try:
                import cohere
                self._client = cohere.AsyncClient(api_key=self._api_key)
            except ImportError:
                pass

    async def rerank(
        self, query: str, documents: list[dict[str, Any]], top_k: int
    ) -> list[dict[str, Any]]:
        if not self._client or not documents:
            return documents[:top_k]

        texts = [doc.get("content", doc.get("text", "")) for doc in documents]
        response = await self._client.rerank(
            model=self._model,
            query=query,
            documents=texts,
            top_n=top_k,
        )

        ranked_results = []
        for result in response.results:
            orig_doc = documents[result.index].copy()
            orig_doc["rerank_score"] = result.relevance_score
            ranked_results.append(orig_doc)
        
        return ranked_results


class HuggingFaceReranker(BaseReranker):
    """Local reranking using Cross-Encoder models from Hugging Face."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base") -> None:
        self._model_name = model_name
        self._model = None
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(model_name)
        except ImportError:
            pass

    async def rerank(
        self, query: str, documents: list[dict[str, Any]], top_k: int
    ) -> list[dict[str, Any]]:
        if not self._model or not documents:
            return documents[:top_k]

        texts = [doc.get("content", doc.get("text", "")) for doc in documents]
        pairs = [[query, text] for text in texts]
        
        # Cross-encoder inference (can be slow, ideally run in executor)
        import anyio
        scores = await anyio.to_thread.run_sync(self._model.predict, pairs)

        results = []
        for i, score in enumerate(scores):
            doc = documents[i].copy()
            doc["rerank_score"] = float(score)
            results.append(doc)

        # Sort by rerank score descending
        results.sort(key=lambda x: x["rerank_score"], reverse=True)
        return results[:top_k]
