"""Unit tests for the Embedding Engine and Adapters."""

import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock

from openrag.config import EmbeddingConfig
from openrag.embeddings.cache import InMemoryEmbeddingCache
from openrag.embeddings.engine import EmbeddingEngine
from openrag.embeddings.openai import OpenAIEmbeddingAdapter
from openrag.models.content import BlockType, ContentBlock
from openrag.models.processing import ProcessedBlock


class TestEmbeddingEngine:
    @pytest.fixture
    def mock_adapter(self) -> MagicMock:
        adapter = MagicMock()
        adapter.embed = AsyncMock(side_effect=lambda texts: [np.random.rand(1536) for _ in texts])
        adapter.dimension.return_value = 1536
        return adapter

    @pytest.fixture
    def cache(self) -> InMemoryEmbeddingCache:
        return InMemoryEmbeddingCache()

    @pytest.fixture
    def engine(self, mock_adapter: MagicMock, cache: InMemoryEmbeddingCache) -> EmbeddingEngine:
        return EmbeddingEngine(mock_adapter, cache, batch_size=2)

    @pytest.mark.asyncio
    async def test_embed_blocks_batching_and_caching(
        self, engine: EmbeddingEngine, mock_adapter: MagicMock, cache: InMemoryEmbeddingCache
    ) -> None:
        # Create 3 blocks with 2 unique texts
        blocks = [
            ProcessedBlock(
                source_block=ContentBlock("b1", "doc1", BlockType.TEXT, 0, "text 1"),
                natural_language_description="text 1",
                embedding_text="text 1",
                entity_candidates=[],
            ),
            ProcessedBlock(
                source_block=ContentBlock("b2", "doc1", BlockType.TEXT, 1, "text 2"),
                natural_language_description="text 2",
                embedding_text="text 2",
                entity_candidates=[],
            ),
            ProcessedBlock(
                source_block=ContentBlock("b3", "doc1", BlockType.TEXT, 2, "text 1"),
                natural_language_description="text 1",
                embedding_text="text 1",
                entity_candidates=[],
            ),
        ]

        # First run: should call adapter for 2 unique texts
        await engine.embed_blocks(blocks)
        assert mock_adapter.embed.call_count == 1  # 2 unique texts, batch size 2 -> 1 call
        assert await cache.size() == 2

        # Second run: all should be cached
        mock_adapter.embed.reset_mock()
        await engine.embed_blocks(blocks)
        assert mock_adapter.embed.call_count == 0

    @pytest.mark.asyncio
    async def test_embed_query_caching(
        self, engine: EmbeddingEngine, mock_adapter: MagicMock, cache: InMemoryEmbeddingCache
    ) -> None:
        text = "sample query"
        
        # First call
        vec1 = await engine.embed_query(text)
        assert mock_adapter.embed.call_count == 1
        assert isinstance(vec1, np.ndarray)
        
        # Second call
        vec2 = await engine.embed_query(text)
        assert mock_adapter.embed.call_count == 1 # still 1
        np.testing.assert_array_equal(vec1, vec2)


class TestOpenAIAdapter:
    @pytest.mark.asyncio
    async def test_openai_embed_call(self, mocker: MagicMock) -> None:
        # Mock the AsyncOpenAI client
        mock_openai = mocker.patch("openrag.embeddings.openai.AsyncOpenAI")
        mock_client = mock_openai.return_value
        mock_client.embeddings.create = AsyncMock(return_value=MagicMock(data=[
            MagicMock(embedding=[0.1, 0.2, 0.3])
        ]))

        config = EmbeddingConfig(model="text-embedding-3-small", dimensions=3)
        adapter = OpenAIEmbeddingAdapter(config)
        
        vecs = await adapter.embed(["hello"])
        assert len(vecs) == 1
        assert vecs[0].tolist() == pytest.approx([0.1, 0.2, 0.3])
        assert adapter.dimension() == 3
