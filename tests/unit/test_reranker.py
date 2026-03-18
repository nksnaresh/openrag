"""Unit tests for the Reranking Engine."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from openrag.search.reranker import CohereReranker, HuggingFaceReranker


class TestReranker:
    @pytest.mark.asyncio
    async def test_cohere_rerank_call(self, mocker: MagicMock) -> None:
        mock_cohere = mocker.patch("cohere.AsyncClient")
        mock_client = mock_cohere.return_value
        
        # Build a mock response object
        mock_result = MagicMock()
        mock_result.index = 0
        mock_result.relevance_score = 0.99
        
        mock_response = MagicMock()
        mock_response.results = [mock_result]
        
        mock_client.rerank = AsyncMock(return_value=mock_response)
        
        reranker = CohereReranker(api_key="fake")
        docs = [{"id": "d1", "content": "text 1"}]
        results = await reranker.rerank("query", docs, top_k=1)
        
        assert len(results) == 1
        assert results[0]["id"] == "d1"
        assert results[0]["rerank_score"] == 0.99

    @pytest.mark.asyncio
    async def test_hf_rerank_call(self, mocker: MagicMock) -> None:
        mock_ce = mocker.patch("sentence_transformers.CrossEncoder")
        mock_model = mock_ce.return_value
        mock_model.predict.return_value = [0.1, 0.8]
        
        reranker = HuggingFaceReranker(model_name="fake")
        docs = [
            {"id": "d1", "content": "text 1"},
            {"id": "d2", "content": "text 2"}
        ]
        results = await reranker.rerank("query", docs, top_k=2)
        
        assert results[0]["id"] == "d2" # higher score
        assert results[1]["id"] == "d1"
        assert results[0]["rerank_score"] == 0.8
