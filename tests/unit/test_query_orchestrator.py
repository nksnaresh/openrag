"""Unit tests for the Query Orchestrator."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from openrag.config import OpenRAGConfig
from openrag.query.orchestrator import QueryOrchestrator
from openrag.models.query import QueryRequest


class TestQueryOrchestrator:
    @pytest.fixture
    def mock_hybrid(self) -> MagicMock:
        searcher = MagicMock()
        searcher.search = AsyncMock(return_value=[
            {"id": "d1", "content": "context content", "source_path": "file.txt"}
        ])
        # Need to mock RRF fusion as well if called directly
        searcher.reciprocal_rank_fusion = MagicMock(side_effect=lambda x: x[0])
        return searcher

    @pytest.fixture
    def orch(self, mock_hybrid: MagicMock) -> QueryOrchestrator:
        return QueryOrchestrator(
            config=OpenRAGConfig(),
            hybrid_searcher=mock_hybrid,
            llm_func=AsyncMock(return_value="The answer is 42.")
        )

    @pytest.mark.asyncio
    async def test_execute_full_flow(self, orch: QueryOrchestrator, mock_hybrid: MagicMock) -> None:
        request = QueryRequest(text="What is the answer?", namespace="ns1")
        response = await orch.execute(request)
        
        assert response.answer == "The answer is 42."
        assert len(response.citations) == 1
        assert response.citations[0].document_title == "file.txt"
        assert mock_hybrid.search.call_count == 1
