"""Unit tests for BM25 Indexer."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from openrag.models.content import BlockType, ContentBlock
from openrag.models.processing import ProcessedBlock
from openrag.search.bm25_indexer import BM25Indexer


class TestBM25Indexer:
    @pytest.fixture
    def mock_doc_store(self) -> MagicMock:
        store = MagicMock()
        store.save_bm25_tokens = AsyncMock()
        return store

    @pytest.fixture
    def indexer(self, mock_doc_store: MagicMock) -> BM25Indexer:
        return BM25Indexer(mock_doc_store)

    def test_tokenize(self, indexer: BM25Indexer) -> None:
        text = "Hello, world! This is a test."
        tokens = indexer.tokenize(text)
        assert tokens == ["hello", "world", "this", "is", "a", "test"]

    @pytest.mark.asyncio
    async def test_index_blocks(
        self, indexer: BM25Indexer, mock_doc_store: MagicMock
    ) -> None:
        blocks = [
            ProcessedBlock(
                source_block=ContentBlock(
                    block_id="b1",
                    document_id="doc1",
                    block_type=BlockType.TEXT,
                    sequence_index=0,
                    raw_content="content 1"
                ),
                natural_language_description="desc 1",
                embedding_text="tokenized text 1",
                entity_candidates=[],
            )
        ]
        
        await indexer.index_blocks(blocks, "ns1")
        
        assert mock_doc_store.save_bm25_tokens.call_count == 1
        call_args = mock_doc_store.save_bm25_tokens.call_args[1]
        assert call_args["namespace"] == "ns1"
        assert call_args["chunk_id"] == "b1"
        assert "tokenized" in call_args["tokens"]
        assert "text" in call_args["tokens"]
