"""Unit tests for the ContextEnricher."""

from __future__ import annotations

from openrag.models.content import BlockType, ContentBlock, ContentPayload, DocumentMeta
from openrag.models.processing import ContextWindowConfig
from openrag.pipeline.context_enricher import ContextEnricher


def _block(
    idx: int, text: str, page: int | None = None, btype: BlockType = BlockType.TEXT
) -> ContentBlock:
    return ContentBlock(
        document_id="doc1",
        block_id=f"b{idx}",
        block_type=btype,
        sequence_index=idx,
        raw_content=text,
        page_number=page,
    )


def _payload(*blocks: ContentBlock) -> ContentPayload:
    return ContentPayload(
        document_id="doc1",
        source_path="/tmp/doc1.pdf",
        tenant_id="t1",
        metadata=DocumentMeta(),
        blocks=list(blocks),
    )


class TestContextEnricher:
    def setup_method(self) -> None:
        self.enricher = ContextEnricher()

    def test_block_window_returns_neighbors(self) -> None:
        blocks = [_block(i, f"sentence {i}", page=0) for i in range(5)]
        payload = _payload(*blocks)
        target = blocks[2]
        config = ContextWindowConfig(strategy="block_window", window_size=1, max_tokens=2000)
        result = self.enricher.enrich(target, payload, config)
        assert "sentence 1" in result
        assert "sentence 3" in result
        assert "sentence 2" not in result  # target itself excluded

    def test_block_window_excludes_non_text(self) -> None:
        target = _block(1, "target text")
        img = ContentBlock(document_id="doc1", block_id="img", block_type=BlockType.IMAGE,
                           sequence_index=0, raw_content=b"bytes")
        payload = _payload(img, target)
        config = ContextWindowConfig(strategy="block_window", window_size=2, max_tokens=2000)
        result = self.enricher.enrich(target, payload, config)
        # Image block should not appear in context
        assert "bytes" not in result

    def test_page_window_returns_same_page_blocks(self) -> None:
        b0 = _block(0, "page 0 text A", page=0)
        b1 = _block(1, "page 1 text B", page=1)
        b2 = _block(2, "page 1 target", page=1)
        b3 = _block(3, "page 2 text C", page=2)
        payload = _payload(b0, b1, b2, b3)
        config = ContextWindowConfig(strategy="page_window", window_size=0, max_tokens=2000)
        result = self.enricher.enrich(b2, payload, config)
        assert "page 1 text B" in result
        assert "page 0 text A" not in result
        assert "page 2 text C" not in result

    def test_page_window_with_window_size(self) -> None:
        b0 = _block(0, "page 0 text", page=0)
        b1 = _block(1, "page 1 target", page=1)
        b2 = _block(2, "page 2 text", page=2)
        payload = _payload(b0, b1, b2)
        config = ContextWindowConfig(strategy="page_window", window_size=1, max_tokens=2000)
        result = self.enricher.enrich(b1, payload, config)
        assert "page 0 text" in result
        assert "page 2 text" in result

    def test_section_ancestor_finds_heading(self) -> None:
        h = _block(0, "# Introduction")
        p1 = _block(1, "First paragraph.")
        p2 = _block(2, "Second paragraph.")
        target = _block(3, "Target block.")
        payload = _payload(h, p1, p2, target)
        config = ContextWindowConfig(strategy="section_ancestor", max_tokens=2000)
        result = self.enricher.enrich(target, payload, config)
        assert "Introduction" in result
        assert "First paragraph" in result

    def test_truncate_to_max_tokens(self) -> None:
        long_text = " ".join([f"word{i}" for i in range(2000)])
        target = _block(0, "target")
        neighbor = _block(1, long_text)
        payload = _payload(target, neighbor)
        config = ContextWindowConfig(strategy="block_window", window_size=5, max_tokens=50)
        result = self.enricher.enrich(target, payload, config)
        word_count = len(result.split())
        # Should be well under 150 words (50 tokens * ~1.33 words/token)
        assert word_count < 150  # noqa: PLR2004

    def test_unknown_strategy_falls_back_to_block_window(self) -> None:
        b0 = _block(0, "before")
        target = _block(1, "target")
        b2 = _block(2, "after")
        payload = _payload(b0, target, b2)
        config = ContextWindowConfig(strategy="unknown_strategy", window_size=1, max_tokens=500)
        result = self.enricher.enrich(target, payload, config)
        assert "before" in result

    def test_page_window_without_page_numbers_falls_back(self) -> None:
        b0 = _block(0, "no page A")  # page_number=None
        target = _block(1, "no page target")
        payload = _payload(b0, target)
        config = ContextWindowConfig(strategy="page_window", window_size=1, max_tokens=500)
        # Should not raise even without page numbers
        result = self.enricher.enrich(target, payload, config)
        assert isinstance(result, str)
