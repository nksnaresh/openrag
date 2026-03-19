"""Unit tests for all 5 modality processors.

All tests use mocked LLM/VLM functions — no real API calls.
"""

from __future__ import annotations

import json

import pytest

from openrag.models.content import BlockType, ContentBlock, ContentPayload, DocumentMeta
from openrag.models.processing import (
    ContextWindowConfig,
    ProcessedBlock,
    ProcessingContext,
)
from openrag.processors.code_processor import CodeProcessor
from openrag.processors.equation_processor import EquationProcessor
from openrag.processors.image_processor import ImageProcessor
from openrag.processors.table_processor import TableProcessor
from openrag.processors.text_processor import TextProcessor

# ── Helpers ────────────────────────────────────────────────────────────────────

def _block(btype: BlockType, content: object, meta: dict | None = None) -> ContentBlock:
    return ContentBlock(
        document_id="doc-test",
        block_id="blk-test",
        block_type=btype,
        sequence_index=0,
        raw_content=content,
        metadata=meta or {},
    )


def _ctx(block: ContentBlock, llm_fn=None, vlm_fn=None) -> ProcessingContext:
    payload = ContentPayload(
        document_id="doc1", source_path="/tmp/d.pdf", tenant_id="t1",
        metadata=DocumentMeta(), blocks=[block],
    )
    return ProcessingContext(
        payload=payload,
        llm_func=llm_fn,  # type: ignore[arg-type]
        vlm_func=vlm_fn,
        context_config=ContextWindowConfig(),
        tenant_id="t1",
        namespace="ns1",
    )


def _mock_llm(response_dict: dict):
    async def _fn(prompt: str, **kwargs) -> str:
        return json.dumps(response_dict)
    return _fn


def _mock_vlm(response_dict: dict):
    async def _fn(prompt: str, image_b64: str = "", **kwargs) -> str:
        return json.dumps(response_dict)
    return _fn


# ── TextProcessor ──────────────────────────────────────────────────────────────

class TestTextProcessor:
    @pytest.mark.asyncio
    async def test_short_text_single_chunk(self) -> None:
        proc = TextProcessor(chunk_size=512)
        block = _block(BlockType.TEXT, "Hello world. This is a short test.")
        result = await proc.process(block, _ctx(block))
        assert isinstance(result, ProcessedBlock)
        assert result.natural_language_description == "Hello world. This is a short test."

    @pytest.mark.asyncio
    async def test_long_text_splits_into_chunks(self) -> None:
        proc = TextProcessor(chunk_size=100, chunk_overlap=10)
        long_text = "word " * 200  # 1000 chars
        block = _block(BlockType.TEXT, long_text)
        result = await proc.process(block, _ctx(block))
        data = result.structured_data
        assert isinstance(data, dict)
        assert data["chunk_count"] > 1

    @pytest.mark.asyncio
    async def test_supported_block_types(self) -> None:
        proc = TextProcessor()
        assert BlockType.TEXT in proc.supported_block_types()

    @pytest.mark.asyncio
    async def test_entity_candidates_empty_in_phase2(self) -> None:
        proc = TextProcessor()
        block = _block(BlockType.TEXT, "Some text")
        result = await proc.process(block, _ctx(block))
        assert result.entity_candidates == []


# ── ImageProcessor ─────────────────────────────────────────────────────────────

class TestImageProcessor:
    @pytest.mark.asyncio
    async def test_no_vlm_returns_fallback(self) -> None:
        proc = ImageProcessor()
        block = _block(BlockType.IMAGE, b"\x89PNG\r\n")
        result = await proc.process(block, _ctx(block, vlm_fn=None))
        assert result.confidence_score == 0.0
        assert "No VLM" in result.natural_language_description

    @pytest.mark.asyncio
    async def test_with_vlm_returns_parsed_description(self) -> None:
        proc = ImageProcessor()
        vlm = _mock_vlm({
            "description": "A bar chart showing sales data.",
            "ocr_text": "Sales 2024",
            "entities": [{"name": "Sales", "type": "CONCEPT"}],
            "confidence": 0.95,
        })
        block = _block(BlockType.IMAGE, b"\x89PNG\r\n")
        result = await proc.process(block, _ctx(block, vlm_fn=vlm))
        assert "bar chart" in result.natural_language_description
        assert "Sales 2024" in result.embedding_text
        assert result.confidence_score == pytest.approx(0.95)

    @pytest.mark.asyncio
    async def test_empty_bytes_returns_fallback(self) -> None:
        proc = ImageProcessor()
        block = _block(BlockType.IMAGE, b"")
        result = await proc.process(block, _ctx(block, vlm_fn=_mock_vlm({})))
        assert result.confidence_score == 0.0

    @pytest.mark.asyncio
    async def test_entities_extracted_from_vlm_response(self) -> None:
        proc = ImageProcessor()
        vlm = _mock_vlm({
            "description": "Logo of OpenAI",
            "ocr_text": "",
            "entities": [{"name": "OpenAI", "type": "ORG"}],
            "confidence": 0.9,
        })
        block = _block(BlockType.IMAGE, b"\xff\xd8\xff")
        result = await proc.process(block, _ctx(block, vlm_fn=vlm))
        assert len(result.entity_candidates) == 1
        assert result.entity_candidates[0].name == "OpenAI"


# ── TableProcessor ─────────────────────────────────────────────────────────────

class TestTableProcessor:
    _TABLE_MD = "| Name | Score |\n|---|---|\n| Alice | 95 |\n| Bob | 87 |"

    @pytest.mark.asyncio
    async def test_no_llm_returns_schema_description(self) -> None:
        proc = TableProcessor()
        block = _block(BlockType.TABLE, self._TABLE_MD)
        result = await proc.process(block, _ctx(block, llm_fn=None))
        desc = result.natural_language_description
        assert "Name" in desc or "table" in desc.lower()

    @pytest.mark.asyncio
    async def test_with_llm_returns_summary(self) -> None:
        proc = TableProcessor()
        llm = _mock_llm({
            "summary": "A table showing student scores.",
            "domain_tags": ["education"],
            "key_insights": ["Alice scored highest"],
            "entities": [{"name": "Alice", "type": "PERSON"}],
        })
        block = _block(BlockType.TABLE, self._TABLE_MD)
        result = await proc.process(block, _ctx(block, llm_fn=llm))
        assert "student scores" in result.natural_language_description

    @pytest.mark.asyncio
    async def test_structured_data_contains_markdown(self) -> None:
        proc = TableProcessor()
        block = _block(BlockType.TABLE, self._TABLE_MD)
        result = await proc.process(block, _ctx(block, llm_fn=None))
        assert isinstance(result.structured_data, dict)
        assert result.structured_data.get("markdown") == self._TABLE_MD


# ── EquationProcessor ──────────────────────────────────────────────────────────

class TestEquationProcessor:
    @pytest.mark.asyncio
    async def test_no_llm_returns_fallback(self) -> None:
        proc = EquationProcessor()
        block = _block(BlockType.EQUATION, r"E = mc^2")
        result = await proc.process(block, _ctx(block, llm_fn=None))
        desc = result.natural_language_description
        assert r"E = mc^2" in desc or "equation" in desc.lower()

    @pytest.mark.asyncio
    async def test_with_llm_returns_interpretation(self) -> None:
        proc = EquationProcessor()
        llm = _mock_llm({
            "interpretation": (
                "Mass-energy equivalence: energy equals mass times"
                " speed of light squared."
            ),
            "variables": [
                {"symbol": "E", "definition": "energy"},
                {"symbol": "m", "definition": "mass"},
            ],
            "domain": "physics",
            "complexity": "elementary",
            "entities": [{"name": "energy", "type": "CONCEPT"}],
        })
        block = _block(BlockType.EQUATION, r"E = mc^2")
        result = await proc.process(block, _ctx(block, llm_fn=llm))
        assert "energy" in result.natural_language_description.lower()
        assert result.structured_data["domain"] == "physics"

    @pytest.mark.asyncio
    async def test_latex_stored_in_structured_data(self) -> None:
        proc = EquationProcessor()
        block = _block(BlockType.EQUATION, r"\sum_{i=0}^{n} x_i")
        result = await proc.process(block, _ctx(block, llm_fn=None))
        assert result.structured_data["latex_source"] == r"\sum_{i=0}^{n} x_i"


# ── CodeProcessor ──────────────────────────────────────────────────────────────

class TestCodeProcessor:
    @pytest.mark.asyncio
    async def test_no_llm_uses_metadata(self) -> None:
        proc = CodeProcessor()
        block = _block(BlockType.CODE, "def foo(): pass",
                       meta={"language": "python", "name": "foo", "kind": "function",
                             "docstring": "Does nothing", "file_path": "/tmp/foo.py"})
        result = await proc.process(block, _ctx(block, llm_fn=None))
        assert "foo" in result.natural_language_description
        assert "python" in result.natural_language_description.lower()

    @pytest.mark.asyncio
    async def test_entity_from_function_name(self) -> None:
        proc = CodeProcessor()
        block = _block(BlockType.CODE, "def compute(): pass",
                       meta={"language": "python", "name": "compute", "kind": "function",
                             "docstring": "", "file_path": "/tmp/a.py"})
        result = await proc.process(block, _ctx(block, llm_fn=None))
        assert any(e.name == "compute" for e in result.entity_candidates)

    @pytest.mark.asyncio
    async def test_with_llm_returns_semantic_intent(self) -> None:
        proc = CodeProcessor()
        llm = _mock_llm({
            "semantic_intent": "Sorts a list using quicksort algorithm.",
            "algorithmic_category": "sorting",
            "complexity": "O(n log n)",
            "entities": [{"name": "quicksort", "type": "ALGORITHM"}],
            "tags": ["sorting", "divide-and-conquer"],
        })
        block = _block(BlockType.CODE, "def quicksort(arr): ...",
                       meta={"language": "python", "name": "quicksort", "kind": "function",
                             "docstring": "", "file_path": "/tmp/sort.py"})
        result = await proc.process(block, _ctx(block, llm_fn=llm))
        assert "quicksort" in result.natural_language_description.lower()
        assert result.structured_data["algorithmic_category"] == "sorting"

    @pytest.mark.asyncio
    async def test_no_name_produces_generic_description(self) -> None:
        proc = CodeProcessor()
        block = _block(BlockType.CODE, "x = 1 + 2",
                       meta={"language": "python", "name": "", "kind": "function",
                             "docstring": "", "file_path": ""})
        result = await proc.process(block, _ctx(block, llm_fn=None))
        assert "python" in result.natural_language_description.lower()
