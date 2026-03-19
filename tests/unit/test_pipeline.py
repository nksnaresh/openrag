"""Unit tests for the DAG pipeline engine."""

from __future__ import annotations

import pytest

from openrag.config import OpenRAGConfig
from openrag.models.content import BlockType, ContentBlock, ContentPayload, DocumentMeta
from openrag.models.processing import (
    ContextWindowConfig,
    ProcessedBlock,
    ProcessingContext,
)
from openrag.pipeline.dag_engine import CyclicDependencyError, DAGPipelineEngine, PipelineStage
from openrag.processors.base import BaseModalityProcessor

# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_processor(name: str, block_types: list[BlockType]) -> BaseModalityProcessor:
    """Return a fake processor that produces one ProcessedBlock per block."""
    class _FakeProcessor(BaseModalityProcessor):
        async def process(self, block: ContentBlock, context: ProcessingContext) -> ProcessedBlock:
            return ProcessedBlock(
                source_block=block,
                natural_language_description=f"{name}_desc",
                embedding_text=f"{name}_embed",
            )

        def supported_block_types(self) -> list[BlockType]:
            return block_types

    return _FakeProcessor()


def _make_payload(*block_types: BlockType) -> ContentPayload:
    return ContentPayload(
        document_id="doc-123",
        source_path="/tmp/test.txt",
        tenant_id="t1",
        metadata=DocumentMeta(),
        blocks=[
            ContentBlock(
                block_id=f"b{i}",
                document_id="doc-123",
                block_type=bt,
                sequence_index=i,
                raw_content=f"content {i}",
            )
            for i, bt in enumerate(block_types)
        ],
    )


def _make_context(payload: ContentPayload) -> ProcessingContext:
    return ProcessingContext(
        payload=payload,
        llm_func=None,  # type: ignore[arg-type]
        vlm_func=None,
        context_config=ContextWindowConfig(),
        tenant_id="t1",
        namespace="ns1",
    )


def _make_config() -> OpenRAGConfig:
    return OpenRAGConfig()


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestTopologicalSort:
    def test_single_stage_no_deps(self) -> None:
        p = _make_processor("p", [BlockType.TEXT])
        stages = [PipelineStage("text", p, depends_on=[])]
        engine = DAGPipelineEngine(stages, _make_config())
        assert engine._order == [["text"]]

    def test_two_independent_stages_same_level(self) -> None:
        p1 = _make_processor("p1", [BlockType.TEXT])
        p2 = _make_processor("p2", [BlockType.IMAGE])
        stages = [
            PipelineStage("text",  p1, depends_on=[]),
            PipelineStage("image", p2, depends_on=[]),
        ]
        engine = DAGPipelineEngine(stages, _make_config())
        assert len(engine._order) == 1
        assert set(engine._order[0]) == {"text", "image"}

    def test_three_stage_chain_ordered(self) -> None:
        a = _make_processor("a", [BlockType.TEXT])
        b = _make_processor("b", [BlockType.IMAGE])
        c = _make_processor("c", [BlockType.TABLE])
        stages = [
            PipelineStage("a", a, depends_on=[]),
            PipelineStage("b", b, depends_on=["a"]),
            PipelineStage("c", c, depends_on=["b"]),
        ]
        engine = DAGPipelineEngine(stages, _make_config())
        assert len(engine._order) == 3
        assert engine._order[0] == ["a"]
        assert engine._order[1] == ["b"]
        assert engine._order[2] == ["c"]

    def test_cycle_raises_error(self) -> None:
        p = _make_processor("p", [BlockType.TEXT])
        stages = [
            PipelineStage("a", p, depends_on=["b"]),
            PipelineStage("b", p, depends_on=["a"]),
        ]
        with pytest.raises(CyclicDependencyError):
            DAGPipelineEngine(stages, _make_config())

    def test_unknown_dependency_raises(self) -> None:
        p = _make_processor("p", [BlockType.TEXT])
        stages = [PipelineStage("a", p, depends_on=["nonexistent"])]
        with pytest.raises(ValueError, match="unknown stage"):
            DAGPipelineEngine(stages, _make_config())


class TestDAGExecution:
    @pytest.mark.asyncio
    async def test_execute_returns_processed_blocks(self) -> None:
        p = _make_processor("text", [BlockType.TEXT])
        stages = [PipelineStage("text", p, depends_on=[], block_types=[BlockType.TEXT])]
        engine = DAGPipelineEngine(stages, _make_config())
        payload = _make_payload(BlockType.TEXT, BlockType.TEXT)
        result = await engine.execute(payload, _make_context(payload))
        assert len(result) == 2
        assert all(isinstance(b, ProcessedBlock) for b in result)

    @pytest.mark.asyncio
    async def test_block_type_filter_applied(self) -> None:
        text_proc = _make_processor("text_proc", [BlockType.TEXT])
        stages = [
            PipelineStage("text", text_proc, depends_on=[], block_types=[BlockType.TEXT])
        ]
        engine = DAGPipelineEngine(stages, _make_config())
        # Payload has 1 TEXT and 1 IMAGE block
        payload = _make_payload(BlockType.TEXT, BlockType.IMAGE)
        result = await engine.execute(payload, _make_context(payload))
        # Only the TEXT block processed
        assert len(result) == 1
        assert result[0].natural_language_description == "text_proc_desc"

    @pytest.mark.asyncio
    async def test_empty_payload_returns_empty(self) -> None:
        p = _make_processor("text", [BlockType.TEXT])
        stages = [PipelineStage("text", p, depends_on=[], block_types=[BlockType.TEXT])]
        engine = DAGPipelineEngine(stages, _make_config())
        payload = _make_payload()  # no blocks
        result = await engine.execute(payload, _make_context(payload))
        assert result == []

    @pytest.mark.asyncio
    async def test_two_independent_stages_run(self) -> None:
        text_p = _make_processor("text_p", [BlockType.TEXT])
        image_p = _make_processor("image_p", [BlockType.IMAGE])
        stages = [
            PipelineStage("text",  text_p,  depends_on=[], block_types=[BlockType.TEXT]),
            PipelineStage("image", image_p, depends_on=[], block_types=[BlockType.IMAGE]),
        ]
        engine = DAGPipelineEngine(stages, _make_config())
        payload = _make_payload(BlockType.TEXT, BlockType.IMAGE)
        result = await engine.execute(payload, _make_context(payload))
        assert len(result) == 2
        descs = {r.natural_language_description for r in result}
        assert "text_p_desc" in descs
        assert "image_p_desc" in descs

    @pytest.mark.asyncio
    async def test_stage_with_no_matching_blocks_returns_empty(self) -> None:
        p = _make_processor("eq", [BlockType.EQUATION])
        stages = [PipelineStage("eq", p, depends_on=[], block_types=[BlockType.EQUATION])]
        engine = DAGPipelineEngine(stages, _make_config())
        payload = _make_payload(BlockType.TEXT)  # No equations
        result = await engine.execute(payload, _make_context(payload))
        assert result == []
