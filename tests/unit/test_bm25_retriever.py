"""Unit tests for BM25Retriever."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from openrag.search.bm25_retriever import BM25Retriever


class TestBM25Retriever:
    @pytest.fixture
    def mock_doc_store(self) -> MagicMock:
        doc_store = MagicMock()
        # Returns (id, [tokens]) list
        doc_store.get_all_bm25_tokens = AsyncMock(return_value=[
            ("d1", ["cat", "sat", "mat"]),
            ("d2", ["dog", "barked"]),
            ("d3", ["bird", "flew"]),
            ("d4", ["fish", "swam"]),
        ])
        return doc_store

    @pytest.mark.asyncio
    async def test_bm25_search_hit(self, mock_doc_store: MagicMock, mocker: MagicMock) -> None:
        # Mock BM25Okapi if it's imported
        retriever = BM25Retriever(doc_store=mock_doc_store)
        
        # Search for 'cat'
        results = await retriever.search("the cat", "ns1", limit=10)
        
        # DEBUG: print results
        print(f"\nDEBUG: results={results}")
        
        assert len(results) >= 1
        assert results[0]["id"] == "d1"
        assert results[0]["score"] > 0

    @pytest.mark.asyncio
    async def test_bm25_search_miss(self, mock_doc_store: MagicMock) -> None:
        retriever = BM25Retriever(doc_store=mock_doc_store)
        
        # Search for something non-existent
        results = await retriever.search("elephant", "ns1", limit=10)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_caching(self, mock_doc_store: MagicMock) -> None:
        retriever = BM25Retriever(doc_store=mock_doc_store)
        
        await retriever.search("cat", "ns1")
        await retriever.search("cat", "ns1")
        
        # get_all_bm25_tokens should only be called once due to caching
        assert mock_doc_store.get_all_bm25_tokens.call_count == 1

    @pytest.mark.asyncio
    async def test_empty_index(self, mock_doc_store: MagicMock) -> None:
        mock_doc_store.get_all_bm25_tokens = AsyncMock(return_value=[])
        retriever = BM25Retriever(doc_store=mock_doc_store)
        results = await retriever.search("cat", "empty_ns")
        assert results == []
