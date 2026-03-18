"""Unit tests for Query Processor (HyDE & Multi-Query)."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from openrag.query.query_processor import QueryProcessor


class TestQueryProcessor:
    @pytest.mark.asyncio
    async def test_generate_multi_queries(self) -> None:
        llm = AsyncMock(return_value='["query 1", "query 2"]')
        processor = QueryProcessor(llm_func=llm)
        
        queries = await processor.generate_multi_queries("test")
        assert len(queries) == 2
        assert "query 1" in queries
        assert "query 2" in queries

    @pytest.mark.asyncio
    async def test_hyde_generation(self) -> None:
        llm = AsyncMock(return_value="This is a hypothetical answer.")
        processor = QueryProcessor(llm_func=llm)
        
        doc = await processor.generate_hyde_document("What is X?")
        assert doc == "This is a hypothetical answer."
        assert llm.call_count == 1
