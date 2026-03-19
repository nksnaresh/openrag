"""Contract test suite for all Phase 1 storage adapters.

Each concrete adapter is tested against the same set of assertions to
guarantee they are interchangeable. Tests are fully in-process — no
external services required.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import pytest_asyncio  # noqa: F401 — needed for async fixture discovery

from openrag.models.content import BlockType, ContentBlock, ContentPayload, DocumentMeta
from openrag.models.graph import EdgeType, GraphEdge, GraphNode, NodeType
from openrag.storage.base import SearchResult, VectorRecord
from openrag.storage.document.sqlite import SQLiteDocumentAdapter
from openrag.storage.graph.networkx import NetworkXAdapter
from openrag.storage.vector.in_memory import InMemoryVectorAdapter

# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _vec(values: list[float]) -> np.ndarray:
    return np.array(values, dtype=np.float32)


def _make_payload(doc_id: str = "sha256abc", namespace: str = "test-ns") -> ContentPayload:
    return ContentPayload(
        document_id=doc_id,
        source_path=f"/tmp/{doc_id}.pdf",
        tenant_id="tenant1",
        metadata=DocumentMeta(title="Test Doc", author="Author"),
        blocks=[ContentBlock(
            document_id=doc_id,
            block_id="b0", block_type=BlockType.TEXT,
            sequence_index=0, raw_content="hello",
        )],
    )


def _make_node(node_id: str, namespace: str = "ns1") -> GraphNode:
    return GraphNode(
        node_id=node_id,
        node_type=NodeType.ENTITY,
        label=f"Entity {node_id}",
        tenant_id="t1",
        namespace=namespace,
        properties={"score": 0.9},
    )


def _make_edge(source: str, target: str, etype: EdgeType = EdgeType.CONTAINS) -> GraphEdge:
    return GraphEdge(
        edge_id=f"{source}->{target}",
        source_id=source,
        target_id=target,
        edge_type=etype,
        weight=1.0,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# InMemoryVectorAdapter
# ═══════════════════════════════════════════════════════════════════════════════

class TestInMemoryVectorAdapter:
    @pytest.fixture()
    def adapter(self) -> InMemoryVectorAdapter:
        return InMemoryVectorAdapter()

    @pytest.mark.asyncio
    async def test_initialize_is_noop(self, adapter: InMemoryVectorAdapter) -> None:
        await adapter.initialize()  # must not raise

    @pytest.mark.asyncio
    async def test_count_empty_namespace(self, adapter: InMemoryVectorAdapter) -> None:
        assert await adapter.count("empty-ns") == 0

    @pytest.mark.asyncio
    async def test_upsert_and_count(self, adapter: InMemoryVectorAdapter) -> None:
        records = [
            VectorRecord("v1", _vec([1.0, 0.0]), {"tag": "a"}),
            VectorRecord("v2", _vec([0.0, 1.0]), {"tag": "b"}),
        ]
        await adapter.upsert("ns", records)
        assert await adapter.count("ns") == 2

    @pytest.mark.asyncio
    async def test_upsert_is_idempotent(self, adapter: InMemoryVectorAdapter) -> None:
        await adapter.upsert("ns", [VectorRecord("v1", _vec([1.0, 0.0]), {})])
        await adapter.upsert("ns", [VectorRecord("v1", _vec([0.5, 0.5]), {})])
        assert await adapter.count("ns") == 1

    @pytest.mark.asyncio
    async def test_query_returns_top_k_sorted(self, adapter: InMemoryVectorAdapter) -> None:
        await adapter.upsert("ns", [
            VectorRecord("close", _vec([1.0, 0.01]), {}),
            VectorRecord("far",   _vec([0.0, 1.0]),  {}),
            VectorRecord("mid",   _vec([0.7, 0.7]),  {}),
        ])
        results = await adapter.query("ns", _vec([1.0, 0.0]), top_k=2)
        assert len(results) == 2
        assert results[0].id == "close"  # most similar to [1,0]
        assert all(isinstance(r, SearchResult) for r in results)

    @pytest.mark.asyncio
    async def test_query_empty_namespace_returns_empty(
        self, adapter: InMemoryVectorAdapter
    ) -> None:
        results = await adapter.query("missing", _vec([1.0, 0.0]), top_k=5)
        assert results == []

    @pytest.mark.asyncio
    async def test_query_with_metadata_filter(self, adapter: InMemoryVectorAdapter) -> None:
        await adapter.upsert("ns", [
            VectorRecord("v1", _vec([1.0, 0.0]), {"color": "red"}),
            VectorRecord("v2", _vec([1.0, 0.0]), {"color": "blue"}),
        ])
        results = await adapter.query("ns", _vec([1.0, 0.0]), top_k=5, filters={"color": "red"})
        assert len(results) == 1
        assert results[0].id == "v1"

    @pytest.mark.asyncio
    async def test_delete_removes_records(self, adapter: InMemoryVectorAdapter) -> None:
        await adapter.upsert("ns", [
            VectorRecord("v1", _vec([1.0, 0.0]), {}),
            VectorRecord("v2", _vec([0.0, 1.0]), {}),
        ])
        await adapter.delete("ns", ["v1"])
        assert await adapter.count("ns") == 1
        results = await adapter.query("ns", _vec([1.0, 0.0]), top_k=5)
        assert all(r.id != "v1" for r in results)

    @pytest.mark.asyncio
    async def test_delete_nonexistent_is_silent(self, adapter: InMemoryVectorAdapter) -> None:
        await adapter.delete("ns", ["ghost"])  # must not raise

    @pytest.mark.asyncio
    async def test_namespace_isolation(self, adapter: InMemoryVectorAdapter) -> None:
        await adapter.upsert("ns1", [VectorRecord("v1", _vec([1.0, 0.0]), {})])
        await adapter.upsert("ns2", [VectorRecord("v2", _vec([0.0, 1.0]), {})])
        results = await adapter.query("ns1", _vec([1.0, 0.0]), top_k=5)
        assert all(r.id != "v2" for r in results)

    @pytest.mark.asyncio
    async def test_top_k_capped_at_store_size(self, adapter: InMemoryVectorAdapter) -> None:
        await adapter.upsert("ns", [VectorRecord("v1", _vec([1.0, 0.0]), {})])
        results = await adapter.query("ns", _vec([1.0, 0.0]), top_k=100)
        assert len(results) == 1


# ═══════════════════════════════════════════════════════════════════════════════
# NetworkXAdapter
# ═══════════════════════════════════════════════════════════════════════════════

class TestNetworkXAdapter:
    @pytest.fixture()
    def adapter(self) -> NetworkXAdapter:
        return NetworkXAdapter()

    @pytest.mark.asyncio
    async def test_initialize_is_noop(self, adapter: NetworkXAdapter) -> None:
        await adapter.initialize()

    @pytest.mark.asyncio
    async def test_upsert_and_get_node(self, adapter: NetworkXAdapter) -> None:
        node = _make_node("n1")
        await adapter.upsert_node("ns1", node)
        retrieved = await adapter.get_node("n1")
        assert retrieved is not None
        assert retrieved.node_id == "n1"
        assert retrieved.node_type == NodeType.ENTITY

    @pytest.mark.asyncio
    async def test_get_nonexistent_node_returns_none(self, adapter: NetworkXAdapter) -> None:
        result = await adapter.get_node("ghost")
        assert result is None

    @pytest.mark.asyncio
    async def test_upsert_edge_connects_nodes(self, adapter: NetworkXAdapter) -> None:
        await adapter.upsert_node("ns1", _make_node("a"))
        await adapter.upsert_node("ns1", _make_node("b"))
        edge = _make_edge("a", "b", EdgeType.CONTAINS)
        await adapter.upsert_edge("ns1", edge)
        sg = await adapter.traverse("a", depth=1)
        assert "b" in sg.node_ids()

    @pytest.mark.asyncio
    async def test_traverse_respects_depth(self, adapter: NetworkXAdapter) -> None:
        for nid in ["r", "c1", "c2"]:
            await adapter.upsert_node("ns1", _make_node(nid))
        await adapter.upsert_edge("ns1", _make_edge("r", "c1"))
        await adapter.upsert_edge("ns1", _make_edge("c1", "c2"))
        # depth=1: only root and c1
        sg = await adapter.traverse("r", depth=1)
        assert "c1" in sg.node_ids()
        assert "c2" not in sg.node_ids()
        # depth=2: all three nodes
        sg2 = await adapter.traverse("r", depth=2)
        assert "c2" in sg2.node_ids()

    @pytest.mark.asyncio
    async def test_traverse_nonexistent_returns_empty(self, adapter: NetworkXAdapter) -> None:
        sg = await adapter.traverse("ghost")
        assert sg.nodes == []
        assert sg.edges == []

    @pytest.mark.asyncio
    async def test_find_nodes_by_type(self, adapter: NetworkXAdapter) -> None:
        node_e = GraphNode("e1", NodeType.ENTITY, "Entity", "t1", "ns1")
        node_c = GraphNode("c1", NodeType.CHUNK,  "Chunk",  "t1", "ns1")
        await adapter.upsert_node("ns1", node_e)
        await adapter.upsert_node("ns1", node_c)
        entities = await adapter.find_nodes("ns1", node_type=NodeType.ENTITY.value)
        assert all(n.node_type == NodeType.ENTITY for n in entities)
        assert any(n.node_id == "e1" for n in entities)

    @pytest.mark.asyncio
    async def test_find_nodes_by_label(self, adapter: NetworkXAdapter) -> None:
        await adapter.upsert_node(
            "ns1", GraphNode("x1", NodeType.ENTITY, "Transformer model", "t1", "ns1")
        )
        await adapter.upsert_node("ns1", GraphNode("x2", NodeType.ENTITY, "BERT tokenizer", "t1", "ns1"))
        results = await adapter.find_nodes("ns1", label_contains="Transformer")
        assert len(results) == 1
        assert results[0].node_id == "x1"

    @pytest.mark.asyncio
    async def test_find_nodes_limit(self, adapter: NetworkXAdapter) -> None:
        for i in range(10):
            await adapter.upsert_node("ns1", _make_node(f"node{i}", namespace="ns1"))
        results = await adapter.find_nodes("ns1", limit=3)
        assert len(results) <= 3

    @pytest.mark.asyncio
    async def test_delete_node_removes_edges(self, adapter: NetworkXAdapter) -> None:
        await adapter.upsert_node("ns1", _make_node("a"))
        await adapter.upsert_node("ns1", _make_node("b"))
        await adapter.upsert_edge("ns1", _make_edge("a", "b"))
        await adapter.delete_node("a")
        assert await adapter.get_node("a") is None
        sg = await adapter.traverse("b", depth=1)
        assert "a" not in sg.node_ids()

    @pytest.mark.asyncio
    async def test_delete_nonexistent_node_is_silent(self, adapter: NetworkXAdapter) -> None:
        await adapter.delete_node("ghost")  # must not raise

    @pytest.mark.asyncio
    async def test_find_nodes_empty_namespace(self, adapter: NetworkXAdapter) -> None:
        results = await adapter.find_nodes("nonexistent-ns")
        assert results == []


# ═══════════════════════════════════════════════════════════════════════════════
# SQLiteDocumentAdapter
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture()
def sqlite_adapter(tmp_path: Path) -> SQLiteDocumentAdapter:
    """Provide a fresh SQLite adapter backed by a temp file per test."""

    class _TmpConfig:
        url = f"sqlite+aiosqlite:///{tmp_path}/test.db"

    return SQLiteDocumentAdapter(_TmpConfig())


class TestSQLiteDocumentAdapter:
    @pytest.mark.asyncio
    async def test_initialize_creates_tables(self, sqlite_adapter: SQLiteDocumentAdapter) -> None:
        await sqlite_adapter.initialize()  # must not raise

    @pytest.mark.asyncio
    async def test_save_and_get_document(self, sqlite_adapter: SQLiteDocumentAdapter) -> None:
        await sqlite_adapter.initialize()
        payload = _make_payload("hash001")
        await sqlite_adapter.save_document(payload, "ns1")
        doc = await sqlite_adapter.get_document("hash001", "ns1")
        assert doc is not None
        assert doc["document_id"] == "hash001"
        assert doc["namespace"] == "ns1"

    @pytest.mark.asyncio
    async def test_get_nonexistent_document_returns_none(
        self, sqlite_adapter: SQLiteDocumentAdapter
    ) -> None:
        await sqlite_adapter.initialize()
        result = await sqlite_adapter.get_document("ghost", "ns1")
        assert result is None

    @pytest.mark.asyncio
    async def test_save_document_is_idempotent(
        self, sqlite_adapter: SQLiteDocumentAdapter
    ) -> None:
        await sqlite_adapter.initialize()
        payload = _make_payload("hash001")
        await sqlite_adapter.save_document(payload, "ns1")
        await sqlite_adapter.save_document(payload, "ns1")  # must not raise / duplicate
        docs = await sqlite_adapter.list_documents("ns1")
        assert len(docs) == 1

    @pytest.mark.asyncio
    async def test_find_by_hash_returns_document(
        self, sqlite_adapter: SQLiteDocumentAdapter
    ) -> None:
        await sqlite_adapter.initialize()
        await sqlite_adapter.save_document(_make_payload("hash002"), "ns1")
        result = await sqlite_adapter.find_by_hash("ns1", "hash002")
        assert result is not None
        assert result["content_hash"] == "hash002"

    @pytest.mark.asyncio
    async def test_find_by_hash_returns_none_when_missing(
        self, sqlite_adapter: SQLiteDocumentAdapter
    ) -> None:
        await sqlite_adapter.initialize()
        result = await sqlite_adapter.find_by_hash("ns1", "nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_list_documents_pagination(
        self, sqlite_adapter: SQLiteDocumentAdapter
    ) -> None:
        await sqlite_adapter.initialize()
        for i in range(5):
            await sqlite_adapter.save_document(_make_payload(f"hash{i:03d}"), "ns1")
        page1 = await sqlite_adapter.list_documents("ns1", limit=3, offset=0)
        page2 = await sqlite_adapter.list_documents("ns1", limit=3, offset=3)
        assert len(page1) == 3
        assert len(page2) == 2
        ids1 = {d["document_id"] for d in page1}
        ids2 = {d["document_id"] for d in page2}
        assert ids1.isdisjoint(ids2)

    @pytest.mark.asyncio
    async def test_delete_document(self, sqlite_adapter: SQLiteDocumentAdapter) -> None:
        await sqlite_adapter.initialize()
        await sqlite_adapter.save_document(_make_payload("hash003"), "ns1")
        await sqlite_adapter.delete_document("hash003", "ns1")
        result = await sqlite_adapter.get_document("hash003", "ns1")
        assert result is None

    @pytest.mark.asyncio
    async def test_bm25_tokens_round_trip(
        self, sqlite_adapter: SQLiteDocumentAdapter
    ) -> None:
        await sqlite_adapter.initialize()
        tokens = ["retrieval", "augmented", "generation"]
        await sqlite_adapter.save_bm25_tokens("ns1", "chunk-001", tokens)
        corpus = await sqlite_adapter.get_all_bm25_tokens("ns1")
        assert len(corpus) == 1
        chunk_id, retrieved_tokens = corpus[0]
        assert chunk_id == "chunk-001"
        assert retrieved_tokens == tokens

    @pytest.mark.asyncio
    async def test_bm25_tokens_multiple_chunks(
        self, sqlite_adapter: SQLiteDocumentAdapter
    ) -> None:
        await sqlite_adapter.initialize()
        await sqlite_adapter.save_bm25_tokens("ns1", "c1", ["a", "b"])
        await sqlite_adapter.save_bm25_tokens("ns1", "c2", ["c", "d"])
        corpus = await sqlite_adapter.get_all_bm25_tokens("ns1")
        assert len(corpus) == 2

    @pytest.mark.asyncio
    async def test_bm25_tokens_upsert(self, sqlite_adapter: SQLiteDocumentAdapter) -> None:
        await sqlite_adapter.initialize()
        await sqlite_adapter.save_bm25_tokens("ns1", "c1", ["old"])
        await sqlite_adapter.save_bm25_tokens("ns1", "c1", ["new"])
        corpus = await sqlite_adapter.get_all_bm25_tokens("ns1")
        assert len(corpus) == 1
        assert corpus[0][1] == ["new"]

    @pytest.mark.asyncio
    async def test_session_round_trip(self, sqlite_adapter: SQLiteDocumentAdapter) -> None:
        await sqlite_adapter.initialize()
        turns = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi!"}]
        await sqlite_adapter.save_session("sess-1", turns)
        retrieved = await sqlite_adapter.get_session("sess-1")
        assert retrieved == turns

    @pytest.mark.asyncio
    async def test_get_nonexistent_session_returns_empty(
        self, sqlite_adapter: SQLiteDocumentAdapter
    ) -> None:
        await sqlite_adapter.initialize()
        result = await sqlite_adapter.get_session("ghost-session")
        assert result == []

    @pytest.mark.asyncio
    async def test_session_update(self, sqlite_adapter: SQLiteDocumentAdapter) -> None:
        await sqlite_adapter.initialize()
        await sqlite_adapter.save_session("s", [{"role": "user", "content": "Q1"}])
        await sqlite_adapter.save_session("s", [{"role": "user", "content": "Q2"}])
        retrieved = await sqlite_adapter.get_session("s")
        assert retrieved[0]["content"] == "Q2"

    @pytest.mark.asyncio
    async def test_list_documents_empty_namespace(
        self, sqlite_adapter: SQLiteDocumentAdapter
    ) -> None:
        await sqlite_adapter.initialize()
        result = await sqlite_adapter.list_documents("nonexistent-ns")
        assert result == []
