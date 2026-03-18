"""Abstract base class for all modal processors."""

from __future__ import annotations

from abc import ABC, abstractmethod

from openrag.models.content import BlockType, ContentBlock
from openrag.models.processing import ProcessedBlock, ProcessingContext


class BaseModalityProcessor(ABC):
    """
    Abstract interface for all content-modality processors.

    Each concrete processor handles one or more BlockTypes, calls
    LLM/VLM functions to generate semantic descriptions, and returns
    a ProcessedBlock ready for indexing.

    Extension guide:
        1. Subclass BaseModalityProcessor.
        2. Implement process() and supported_block_types().
        3. Register: AdapterRegistry.register_processor("my_type", MyProcessor)
    """

    @abstractmethod
    async def process(
        self,
        block: ContentBlock,
        context: ProcessingContext,
    ) -> ProcessedBlock:
        """
        Process a single content block and return an enriched result.

        Args:
            block:   The raw ContentBlock to process.
            context: Runtime context (full document payload, LLM/VLM funcs, config).

        Returns:
            ProcessedBlock with natural language description, embedding text,
            entity candidates, and optional structured data.
        """

    @abstractmethod
    def supported_block_types(self) -> list[BlockType]:
        """Return the BlockTypes this processor can handle."""

    # ── Helpers ────────────────────────────────────────────────────────────────

    def can_handle(self, block_type: BlockType) -> bool:
        return block_type in self.supported_block_types()

    @staticmethod
    def _strip_think_tags(text: str) -> str:
        """Strip chain-of-thought <think>…</think> wrapping from LLM output."""
        import re
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    @staticmethod
    def _extract_json(text: str) -> dict[str, object]:
        """
        Robustly extract a JSON object from an LLM response string.

        Tries four strategies in order:
          1. Direct parse (if the whole response is JSON)
          2. Extract from a fenced ```json … ``` block
          3. Extract from the first balanced { … } span
          4. Raise ValueError with the original text for debugging
        """
        import json
        import re

        text = BaseModalityProcessor._strip_think_tags(text)

        # Strategy 1 — full response is JSON
        try:
            return json.loads(text)  # type: ignore[no-any-return]
        except json.JSONDecodeError:
            pass

        # Strategy 2 — fenced code block
        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if m:
            try:
                return json.loads(m.group(1))  # type: ignore[no-any-return]
            except json.JSONDecodeError:
                pass

        # Strategy 3 — first balanced brace span
        start = text.find("{")
        if start != -1:
            depth = 0
            for i, ch in enumerate(text[start:], start=start):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start : i + 1])  # type: ignore[no-any-return]
                        except json.JSONDecodeError:
                            break

        raise ValueError(f"Could not extract JSON from LLM response:\n{text[:500]}")
