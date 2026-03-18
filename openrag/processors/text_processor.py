"""Text modality processor.

Chunks text blocks into ProcessedBlocks using a recursive character
splitter strategy. Designed to be LLM-free: chunking and entity candidate
extraction are done without external API calls.

The processor is designed around the "embedding_text = chunk_text" principle
so that downstream embedding engines receive focused, coherent text.
"""

from __future__ import annotations

from openrag.models.content import BlockType, ContentBlock
from openrag.models.processing import ProcessedBlock, ProcessingContext
from openrag.processors.base import BaseModalityProcessor

# Separators used in recursive character splitting (highest priority first)
_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]


class TextProcessor(BaseModalityProcessor):
    """Chunk text blocks into semantically meaningful ProcessedBlocks.

    Default strategy: recursive character splitting with tiktoken budget.
    Entity candidates are stubbed as empty list (full NER lives in Phase 3).
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    async def process(
        self, block: ContentBlock, context: ProcessingContext
    ) -> ProcessedBlock:
        """Split a TEXT block into chunks and return the first ProcessedBlock.

        Note: in a real pipeline the DAG engine would receive multiple
        ProcessedBlocks per source block. For the Phase 2 interface
        (single return), we return the full text as one ProcessedBlock and
        attach the chunk list in structured_data so the orchestrator can
        explode them if needed.
        """
        text = str(block.raw_content)
        chunks = self._split(text, self._chunk_size, self._chunk_overlap)

        if not chunks:
            chunks = [text]

        # Primary block: first chunk
        return ProcessedBlock(
            source_block=block,
            natural_language_description=chunks[0],
            embedding_text=chunks[0],
            entity_candidates=[],  # Phase 3: spaCy NER
            structured_data={
                "chunks": chunks,
                "chunk_count": len(chunks),
                "strategy": "recursive_character",
            },
            confidence_score=1.0,
        )

    def supported_block_types(self) -> list[BlockType]:
        return [BlockType.TEXT]

    # ── Chunking ───────────────────────────────────────────────────────────────

    def _split(self, text: str, chunk_size: int, overlap: int) -> list[str]:
        """Recursive character split respecting a character budget."""
        return self._recursive_split(text, _SEPARATORS, chunk_size, overlap)

    def _recursive_split(
        self,
        text: str,
        separators: list[str],
        chunk_size: int,
        overlap: int,
    ) -> list[str]:
        if len(text) <= chunk_size:
            return [text] if text.strip() else []

        separator = separators[0] if separators else ""
        remaining_seps = separators[1:]

        if separator == "" or separator not in text:
            if remaining_seps:
                return self._recursive_split(text, remaining_seps, chunk_size, overlap)
            # Hard split by character budget
            return self._hard_split(text, chunk_size, overlap)

        splits = text.split(separator)
        good_splits: list[str] = []
        current = ""

        for split in splits:
            candidate = (current + separator + split).lstrip(separator) if current else split
            if len(candidate) <= chunk_size:
                current = candidate
            else:
                if current:
                    good_splits.append(current)
                    # carry overlap
                    words = current.split()
                    overlap_text = " ".join(words[-overlap // 5:]) if words else ""
                    current = (overlap_text + " " + split).strip() if overlap_text else split
                else:
                    # split is itself too long — recurse
                    if remaining_seps:
                        good_splits.extend(
                            self._recursive_split(split, remaining_seps, chunk_size, overlap)
                        )
                    else:
                        good_splits.extend(self._hard_split(split, chunk_size, overlap))
                    current = ""

        if current:
            good_splits.append(current)

        return [s for s in good_splits if s.strip()]

    @staticmethod
    def _hard_split(text: str, chunk_size: int, overlap: int) -> list[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start = end - overlap
        return chunks
