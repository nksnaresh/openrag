"""Unit tests for HybridSearcher and RRF fusion."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from openrag.search.hybrid_search import HybridSearcher


class TestHybridSearcher:
    @pytest.fixture
    def mock_deps(self) -> dict[str, MagicMock]:
        return {
            "vdb": MagicMock(),
            "gdb": MagicMock(),
            "doc": MagicMock(),
            "engine": MagicMock(),
        }

    @pytest.fixture
    def searcher(self, mock_deps: dict[str, MagicMock]) -> HybridSearcher:
        return HybridSearcher(
            vector_db=mock_deps["vdb"],
            graph_db=mock_deps["gdb"],
            doc_store=mock_deps["doc"],
            embedding_engine=mock_deps["engine"],
        )

    @pytest.mark.asyncio
    async def test_search_fusion_logic(
        self, searcher: HybridSearcher, mock_deps: dict[str, MagicMock], mocker: MagicMock
    ) -> None:
        # Mock individual fetchers
        searcher._fetch_vector = AsyncMock(side_effect=lambda q, ns, l, out: out.extend([
            {"id": "a", "content": "block a"},
            {"id": "b", "content": "block b"},
        ]))
        searcher._fetch_graph = AsyncMock(side_effect=lambda q, ns, l, out: out.extend([
            {"id": "c", "content": "block c"},
            {"id": "a", "content": "block a"},
        ]))
        searcher._fetch_bm25 = AsyncMock(side_effect=lambda q, ns, l, out: out.extend([
            {"id": "b", "content": "block b"},
            {"id": "d", "content": "block d"},
        ]))

        results = await searcher.search("test query", "ns1", top_k=5)
        
        # Verify RRF combined them
        # 'a' is rank 1 (V) and rank 2 (G)
        # 'b' is rank 2 (V) and rank 1 (B)
        # 'c' is rank 1 (G)
        # 'd' is rank 2 (B)
        
        ids = [r["id"] for r in results]
        assert "a" in ids
        assert "b" in ids
        assert "c" in ids
        assert "d" in ids
        assert len(results) == 4

    def test_rrf_with_missing_ids(self, searcher: HybridSearcher) -> None:
        # One item has no id
        rankings = [[{"content": "no id"}, {"id": "d1"}]]
        fused = searcher.reciprocal_rank_fusion(rankings)
        assert len(fused) == 1
        assert fused[0]["id"] == "d1"

    def test_rrf_scoring(self, searcher: HybridSearcher) -> None:
        rankings = [
            [{"id": "d1"}, {"id": "d2"}],
            [{"id": "d2"}, {"id": "d3"}]
        ]
        fused = searcher.reciprocal_rank_fusion(rankings, k=60)
        
        # d2 score: 1/(60+2) + 1/(60+1) = 0.0161 + 0.0163 = 0.0324
        # d1 score: 1/(60+1) = 0.01639
        # d3 score: 1/(60+2) = 0.01612
        
        assert fused[0]["id"] == "d2"
        assert fused[1]["id"] == "d1"
        assert fused[2]["id"] == "d3"
