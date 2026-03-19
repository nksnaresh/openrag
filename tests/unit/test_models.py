"""Unit tests for all shared data contracts.

These tests verify the model contracts that every other module depends on.
They must pass with zero external dependencies (stdlib + pydantic only).
"""

from __future__ import annotations

import pytest

from openrag.models.content import (
    BlockType,
    BoundingBox,
    ContentBlock,
    ContentPayload,
    DocumentMeta,
)
from openrag.models.graph import EdgeType, GraphEdge, GraphNode, NodeType, SubGraph
from openrag.models.jobs import (
    BatchJobResult,
    BatchOptions,
    IngestMetadata,
    JobResult,
    JobStatus,
)
from openrag.models.processing import (
    ContextWindowConfig,
    EntityCandidate,
    EntityType,
    ProcessedBlock,
)
from openrag.models.query import Citation, QueryMode, QueryRequest, QueryResponse

# ── BlockType ────────────────────────────────────────────────────────────────

class TestBlockType:
    def test_all_variants_exist(self) -> None:
        expected = {"text", "image", "table", "equation", "code", "audio_transcript",
                    "video_frame", "unknown"}
        assert {bt.value for bt in BlockType} == expected

    def test_str_coercion(self) -> None:
        assert BlockType("image") is BlockType.IMAGE


# ── BoundingBox ───────────────────────────────────────────────────────────────

class TestBoundingBox:
    def test_properties(self) -> None:
        bb = BoundingBox(x0=10, y0=20, x1=110, y1=70, page=1)
        assert bb.width == 100
        assert bb.height == 50
        assert bb.area == 5000

    def test_frozen(self) -> None:
        import dataclasses
        bb = BoundingBox(0, 0, 10, 10, page=0)
        with pytest.raises(dataclasses.FrozenInstanceError):
            bb.x0 = 5  # type: ignore[misc]


# ── ContentBlock ─────────────────────────────────────────────────────────────

class TestContentBlock:
    def test_minimal_creation(self) -> None:
        block = ContentBlock(
            block_id="abc123",
            document_id="abc123doc",
            block_type=BlockType.TEXT,
            sequence_index=0,
            raw_content="Hello world",
        )
        assert block.block_id == "abc123"
        assert block.page_number is None
        assert block.metadata == {}

    def test_with_all_fields(self) -> None:
        bb = BoundingBox(0, 0, 100, 100, page=1)
        block = ContentBlock(
            block_id="img001",
            document_id="doc001",
            block_type=BlockType.IMAGE,
            sequence_index=3,
            raw_content=b"\xff\xd8\xff",
            page_number=2,
            bounding_box=bb,
            metadata={"caption": "Figure 1", "mime_type": "image/jpeg"},
        )
        assert block.bounding_box.area == 10000
        assert block.metadata["caption"] == "Figure 1"


# ── ContentPayload ────────────────────────────────────────────────────────────

class TestContentPayload:
    def _make_payload(self) -> ContentPayload:
        blocks = [
            ContentBlock("b0", "doc1", BlockType.TEXT, 0, "Text content"),
            ContentBlock("b1", "doc1", BlockType.IMAGE, 1, b"img"),
            ContentBlock("b2", "doc1", BlockType.TEXT, 2, "More text"),
        ]
        return ContentPayload(
            document_id="sha256abc",
            source_path="/tmp/doc.pdf",
            tenant_id="test-tenant",
            metadata=DocumentMeta(title="Test Doc"),
            blocks=blocks,
        )

    def test_len(self) -> None:
        payload = self._make_payload()
        assert len(payload) == 3

    def test_blocks_of_type(self) -> None:
        payload = self._make_payload()
        text_blocks = payload.blocks_of_type(BlockType.TEXT)
        assert len(text_blocks) == 2
        assert all(b.block_type == BlockType.TEXT for b in text_blocks)

    def test_default_acl(self) -> None:
        payload = self._make_payload()
        assert payload.acl == {"read": [], "write": []}


# ── QueryMode ─────────────────────────────────────────────────────────────────

class TestQueryMode:
    def test_all_variants(self) -> None:
        modes = {m.value for m in QueryMode}
        assert "hybrid" in modes
        assert "multimodal" in modes

    def test_coercion(self) -> None:
        assert QueryMode("dense") is QueryMode.DENSE


# ── QueryRequest ──────────────────────────────────────────────────────────────

class TestQueryRequest:
    def test_defaults(self) -> None:
        req = QueryRequest(text="What is OpenRAG?")
        assert req.mode is QueryMode.HYBRID
        assert req.namespace == "default"
        assert req.top_k == 20
        assert req.stream is False

    def test_custom(self) -> None:
        req = QueryRequest(
            text="Show me tables",
            mode=QueryMode.SPARSE,
            namespace="my-ns",
            top_k=5,
            output_schema={"type": "object"},
        )
        assert req.output_schema == {"type": "object"}


# ── JobStatus & JobResult ─────────────────────────────────────────────────────

class TestJobs:
    def test_job_status_values(self) -> None:
        assert JobStatus.COMPLETED.value == "completed"
        assert JobStatus.FAILED.value == "failed"

    def test_batch_result_success_rate(self) -> None:
        result = BatchJobResult(
            total=10,
            successful=[JobResult("j1", JobStatus.COMPLETED)] * 8,
            failed=[JobResult("j2", JobStatus.FAILED)] * 2,
        )
        assert result.success_rate == 0.8

    def test_empty_batch_success_rate(self) -> None:
        result = BatchJobResult(total=0)
        assert result.success_rate == 0.0


# ── Graph Models ──────────────────────────────────────────────────────────────

class TestGraphModels:
    def test_edge_type_values(self) -> None:
        assert EdgeType.CONTAINS.value == "CONTAINS"
        assert EdgeType.ILLUSTRATES.value == "ILLUSTRATES"

    def test_subgraph_node_ids(self) -> None:
        nodes = [
            GraphNode("n1", NodeType.ENTITY, "Entity A", "t1", "ns1"),
            GraphNode("n2", NodeType.CHUNK, "Chunk B", "t1", "ns1"),
        ]
        sg = SubGraph(nodes=nodes)
        assert set(sg.node_ids()) == {"n1", "n2"}


# ── ContextWindowConfig ───────────────────────────────────────────────────────

class TestContextWindowConfig:
    def test_defaults(self) -> None:
        cfg = ContextWindowConfig()
        assert cfg.strategy == "page_window"
        assert cfg.window_size == 2
        assert cfg.max_tokens == 1500

    def test_custom(self) -> None:
        cfg = ContextWindowConfig(strategy="semantic_neighbour", max_tokens=500)
        assert cfg.max_tokens == 500


# ── EntityCandidate ───────────────────────────────────────────────────────────


class TestEntityCandidate:
    def test_minimal_creation(self) -> None:
        e = EntityCandidate(
            name="OpenRAG",
            canonical_name="openrag",
            entity_type=EntityType.PRODUCT,
        )
        assert e.confidence == 1.0
        assert e.source_text is None

    def test_all_fields(self) -> None:
        e = EntityCandidate(
            name="Transformer",
            canonical_name="transformer",
            entity_type=EntityType.ALGORITHM,
            confidence=0.87,
            source_text="The Transformer architecture was introduced in 2017.",
        )
        assert e.entity_type is EntityType.ALGORITHM
        assert e.confidence == 0.87

    def test_all_entity_types_defined(self) -> None:
        expected = {
            "PERSON", "ORGANIZATION", "LOCATION", "CONCEPT", "ALGORITHM",
            "DATASET", "EQUATION", "FUNCTION", "CLASS", "MODULE",
            "PRODUCT", "EVENT", "UNKNOWN",
        }
        assert {et.value for et in EntityType} == expected


# ── ProcessedBlock ────────────────────────────────────────────────────────────


class TestProcessedBlock:
    def _make_source_block(self) -> ContentBlock:
        return ContentBlock(
            block_id="blk0",
            document_id="doc1",
            block_type=BlockType.TEXT,
            sequence_index=0,
            raw_content="Some text",
        )

    def test_minimal_creation(self) -> None:
        pb = ProcessedBlock(
            source_block=self._make_source_block(),
            natural_language_description="A paragraph about OpenRAG.",
            embedding_text="A paragraph about OpenRAG.",
        )
        assert pb.confidence_score == 1.0
        assert pb.entity_candidates == []
        assert pb.structured_data is None
        assert pb.processing_metadata == {}

    def test_with_entities(self) -> None:
        entity = EntityCandidate("RAG", "rag", EntityType.CONCEPT)
        pb = ProcessedBlock(
            source_block=self._make_source_block(),
            natural_language_description="RAG stands for retrieval augmented generation.",
            embedding_text="RAG stands for retrieval augmented generation.",
            entity_candidates=[entity],
            confidence_score=0.95,
        )
        assert len(pb.entity_candidates) == 1
        assert pb.confidence_score == 0.95

    def test_structured_data_for_table(self) -> None:
        pb = ProcessedBlock(
            source_block=self._make_source_block(),
            natural_language_description="A table of results.",
            embedding_text="A table of results.",
            structured_data=[{"col1": "val1", "col2": 42}],
        )
        assert isinstance(pb.structured_data, list)


# ── ProcessingContext ─────────────────────────────────────────────────────────


class TestProcessingContext:
    def _make_payload(self) -> ContentPayload:
        return ContentPayload(
            document_id="docabc",
            source_path="/tmp/test.pdf",
            tenant_id="t1",
            metadata=DocumentMeta(),
        )

    def test_with_vlm_none(self) -> None:
        from openrag.models.processing import ProcessingContext

        ctx = ProcessingContext(
            payload=self._make_payload(),
            llm_func=lambda p: p,  # type: ignore[arg-type]
            vlm_func=None,
            context_config=ContextWindowConfig(),
            tenant_id="t1",
            namespace="default",
        )
        assert ctx.vlm_func is None
        assert ctx.extra == {}

    def test_with_extra(self) -> None:
        from openrag.models.processing import ProcessingContext

        ctx = ProcessingContext(
            payload=self._make_payload(),
            llm_func=lambda p: p,  # type: ignore[arg-type]
            vlm_func=lambda p, img: p,  # type: ignore[arg-type]
            context_config=ContextWindowConfig(window_size=5),
            tenant_id="t1",
            namespace="ns1",
            extra={"debug": True},
        )
        assert ctx.extra["debug"] is True
        assert ctx.context_config.window_size == 5


# ── Citation ──────────────────────────────────────────────────────────────────


class TestCitation:
    def test_required_fields(self) -> None:
        c = Citation(
            document_id="doc001",
            document_title="Annual Report 2025",
            page_number=5,
            block_id="blk42",
            block_type=BlockType.TABLE,
            score=0.91,
        )
        assert c.document_title == "Annual Report 2025"
        assert c.excerpt is None

    def test_optional_excerpt_and_null_title(self) -> None:
        c = Citation(
            document_id="doc002",
            document_title=None,
            page_number=None,
            block_id="blk01",
            block_type=BlockType.TEXT,
            score=0.75,
            excerpt="The quick brown fox...",
        )
        assert c.excerpt == "The quick brown fox..."
        assert c.document_title is None


# ── QueryResponse ─────────────────────────────────────────────────────────────


class TestQueryResponse:
    def test_minimal_creation(self) -> None:
        resp = QueryResponse(
            answer="The answer is 42.",
            citations=[],
            query_mode=QueryMode.HYBRID,
            latency_ms=320.5,
        )
        assert resp.session_id is None
        assert resp.structured is None
        assert resp.metadata == {}

    def test_structured_output_field(self) -> None:
        resp = QueryResponse(
            answer='{"result": "yes"}',
            citations=[],
            query_mode=QueryMode.DENSE,
            latency_ms=100.0,
            structured={"result": "yes"},
        )
        assert resp.structured == {"result": "yes"}


# ── GraphEdge ─────────────────────────────────────────────────────────────────


class TestGraphEdge:
    def test_defaults(self) -> None:
        edge = GraphEdge(
            edge_id="e1",
            source_id="n1",
            target_id="n2",
            edge_type=EdgeType.CONTAINS,
        )
        assert edge.weight == 1.0
        assert edge.properties == {}

    def test_custom_weight_and_properties(self) -> None:
        edge = GraphEdge(
            edge_id="e2",
            source_id="n3",
            target_id="n4",
            edge_type=EdgeType.REFERENCES,
            weight=0.5,
            properties={"confidence": 0.8},
        )
        assert edge.weight == 0.5
        assert edge.properties["confidence"] == 0.8

    def test_all_edge_types(self) -> None:
        expected = {
            "CONTAINS", "NEXT", "REFERENCES", "ILLUSTRATES", "DEFINES",
            "IMPLEMENTS", "PROVES", "CITES", "CONTAINS_DATA_ABOUT",
        }
        assert {et.value for et in EdgeType} == expected


# ── DocumentMeta ──────────────────────────────────────────────────────────────


class TestDocumentMeta:
    def test_all_defaults_none(self) -> None:
        meta = DocumentMeta()
        assert meta.title is None
        assert meta.author is None
        assert meta.page_count is None
        assert meta.source_url is None
        assert meta.language == "en"
        assert meta.custom == {}

    def test_with_values(self) -> None:
        meta = DocumentMeta(
            title="Deep Learning",
            author="Goodfellow et al.",
            page_count=800,
            language="en",
            file_size_bytes=5_000_000,
        )
        assert meta.page_count == 800
        assert meta.file_size_bytes == 5_000_000


# ── IngestMetadata & BatchOptions ─────────────────────────────────────────────


class TestIngestMetadata:
    def test_namespace_default(self) -> None:
        meta = IngestMetadata(tenant_id="acme")
        assert meta.namespace == "default"
        assert meta.tags == []
        assert meta.custom == {}

    def test_custom_namespace(self) -> None:
        meta = IngestMetadata(tenant_id="acme", namespace="legal", tags=["contracts"])
        assert meta.namespace == "legal"
        assert "contracts" in meta.tags


class TestBatchOptions:
    def test_default_extensions_non_empty(self) -> None:
        meta = IngestMetadata(tenant_id="t1")
        opts = BatchOptions(metadata=meta)
        assert len(opts.supported_extensions) > 0
        assert ".pdf" in opts.supported_extensions
        assert ".py" in opts.supported_extensions

    def test_default_workers_and_flags(self) -> None:
        opts = BatchOptions(metadata=IngestMetadata(tenant_id="t1"))
        assert opts.max_workers == 4
        assert opts.recursive is True
        assert opts.force_reingest is False
