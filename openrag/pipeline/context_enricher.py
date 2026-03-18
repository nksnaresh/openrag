"""Context enricher for the ingestion pipeline.

Provides surrounding textual context for a ContentBlock so that modality
processors (image, table, equation) have enough document context to produce
high-quality natural language descriptions.

Four builtin strategies:
    - page_window: All text blocks from pages within ±N pages
    - block_window: N blocks before and after by sequence index
    - section_ancestor: Walk backwards to the nearest heading block
    - semantic_neighbour: N adjacent blocks (alias for block_window in Phase 2)
"""

from __future__ import annotations

from openrag.models.content import BlockType, ContentBlock, ContentPayload
from openrag.models.processing import ContextWindowConfig

# Words approximating one token (rough estimate for truncation)
_WORDS_PER_TOKEN = 0.75


class ContextEnricher:
    """Extracts surrounding textual context for a given ContentBlock."""

    def enrich(
        self,
        block: ContentBlock,
        payload: ContentPayload,
        config: ContextWindowConfig,
    ) -> str:
        """Return a context string for the block, truncated to max_tokens.

        Dispatches to the strategy named by ``config.strategy``.
        """
        strategy_fn = {
            "page_window":       self._page_window,
            "block_window":      self._block_window,
            "section_ancestor":  self._section_ancestor,
            "semantic_neighbour": self._block_window,  # Phase 2 alias
        }.get(config.strategy, self._block_window)

        raw = strategy_fn(block, payload, config)
        return self._truncate(raw, config.max_tokens)

    # ── Strategies ─────────────────────────────────────────────────────────────

    def _page_window(
        self,
        block: ContentBlock,
        payload: ContentPayload,
        config: ContextWindowConfig,
    ) -> str:
        """Return text from blocks on pages within ±window_size of block's page."""
        page = block.page_number
        if page is None:
            return self._block_window(block, payload, config)

        texts: list[str] = []
        for b in payload.blocks:
            if b.block_id == block.block_id:
                continue
            if b.block_type not in (BlockType.TEXT,):
                continue
            if b.page_number is None:
                continue
            if abs(b.page_number - page) <= config.window_size:
                texts.append(str(b.raw_content))
        return " ".join(texts)

    def _block_window(
        self,
        block: ContentBlock,
        payload: ContentPayload,
        config: ContextWindowConfig,
    ) -> str:
        """Return text from ±window_size blocks by sequence_index."""
        idx = block.sequence_index
        lo = max(0, idx - config.window_size)
        hi = idx + config.window_size

        texts = [
            str(b.raw_content)
            for b in payload.blocks
            if b.block_id != block.block_id
            and b.block_type == BlockType.TEXT
            and lo <= b.sequence_index <= hi
        ]
        return " ".join(texts)

    def _section_ancestor(
        self,
        block: ContentBlock,
        payload: ContentPayload,
        config: ContextWindowConfig,  # noqa: ARG002
    ) -> str:
        """Return text from block back to the nearest heading-type block."""
        # Find blocks that precede this block by sequence index
        preceding = sorted(
            [b for b in payload.blocks if b.sequence_index < block.sequence_index],
            key=lambda b: b.sequence_index,
            reverse=True,
        )

        texts: list[str] = []
        for b in preceding:
            texts.insert(0, str(b.raw_content))
            # A heading is a TEXT block whose raw_content starts with '#' (markdown)
            if b.block_type == BlockType.TEXT and str(b.raw_content).startswith("#"):
                break

        return " ".join(texts)

    # ── Helpers ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _truncate(text: str, max_tokens: int) -> str:
        """Truncate text to approximately max_tokens by word count."""
        max_words = int(max_tokens / _WORDS_PER_TOKEN)
        words = text.split()
        if len(words) <= max_words:
            return text
        return " ".join(words[:max_words])
