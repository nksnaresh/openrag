"""Query Processor — advanced query expansion and rewriting.

Implements techniques like HyDE (Hypothetical Document Embeddings) and
Multi-Query expansion to improve retrieval recall.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any


class QueryProcessor:
    """Handles query enrichment and expansion before retrieval.

    Args:
        llm_func: Async callable for LLM queries (async(prompt) -> str).
    """

    def __init__(self, llm_func: Callable[..., Any]) -> None:
        self._llm_func = llm_func

    async def generate_hyde_document(self, query: str) -> str:
        """Generate a hypothetical document based on the query."""
        prompt = (
            f"Please write a scientific passage or a technical documentation snippet "
            f"that answers the following query. Focus on providing detailed, factual information "
            f"as if it were part of a larger reference library.\n\n"
            f"Query: {query}\n\n"
            f"Passage:"
        )
        try:
            return await self._llm_func(prompt)
        except Exception:
            return query  # Fallback to original query on failure

    async def generate_multi_queries(self, query: str, count: int = 3) -> list[str]:
        """Generate multiple variations of the same query for better coverage."""
        prompt = (
            f"You are an AI search assistant. Your task is to generate {count} "
            f"different versions of the given user query to retrieve relevant documents from a vector database. "
            f"By generating multiple perspectives on the user query, your goal is to help the user overcome "
            f"some of the limitations of the distance-based similarity search.\n\n"
            f"Original query: {query}\n\n"
            f"Provide the queries as a JSON list of strings."
        )
        try:
            response = await self._llm_func(prompt)
            # Basic JSON extraction (strip markdown if present)
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].strip()
            
            queries = json.loads(response)
            if isinstance(queries, list):
                return [str(q) for q in queries][:count]
            return [query]
        except Exception:
            return [query]
