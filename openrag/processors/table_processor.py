"""Table modality processor.

Extracts schema statistics from a markdown/CSV/HTML table block,
then calls the LLM function for a natural language summary.
"""

from __future__ import annotations

from openrag.models.content import BlockType, ContentBlock
from openrag.models.processing import EntityCandidate, EntityType, ProcessedBlock, ProcessingContext
from openrag.pipeline.context_enricher import ContextEnricher
from openrag.processors.base import BaseModalityProcessor
from openrag.prompts.table import TABLE_ANALYSIS_PROMPT

_enricher = ContextEnricher()


class TableProcessor(BaseModalityProcessor):
    """Process TABLE blocks into summarized ProcessedBlocks.

    Works with markdown tables (produced by PyMuPDF and HTML parsers).
    Extracts column names + basic stats, then sends them with a context
    window to the LLM for a natural language summary.
    """

    async def process(
        self, block: ContentBlock, context: ProcessingContext
    ) -> ProcessedBlock:
        raw = str(block.raw_content)
        schema_summary, sample = self._parse_markdown_table(raw)
        surrounding = _enricher.enrich(block, context.payload, context.context_config)

        if schema_summary:
            fallback_desc = f"A table with columns: {schema_summary}"
        else:
            fallback_desc = "A table block."

        if context.llm_func is None:
            return ProcessedBlock(
                source_block=block,
                natural_language_description=fallback_desc,
                embedding_text=f"{fallback_desc}\n{sample}".strip(),
                entity_candidates=[],
                structured_data={"schema": schema_summary, "markdown": raw},
                confidence_score=0.5,
            )

        prompt = TABLE_ANALYSIS_PROMPT.format(
            table_markdown=sample or raw[:2000],
            schema_summary=schema_summary or "unknown",
            context=surrounding or "(no surrounding context)",
        )
        raw_response = await context.llm_func(prompt)
        parsed = self._extract_json(raw_response)

        summary = str(parsed.get("summary", fallback_desc))
        raw_ents = parsed.get("entities", None)
        ent_list = raw_ents if isinstance(raw_ents, list) else []
        entities = [
            EntityCandidate(
                name=str(e.get("name", "")),
                canonical_name=str(e.get("name", "")).lower(),
                entity_type=EntityType.CONCEPT,
                confidence=0.8,
            )
            for e in ent_list
            if isinstance(e, dict) and e.get("name")
        ]

        return ProcessedBlock(
            source_block=block,
            natural_language_description=summary,
            embedding_text=f"{summary}\n{sample}".strip(),
            entity_candidates=entities,
            structured_data={
                "schema": schema_summary,
                "domain_tags": parsed.get("domain_tags", []),
                "key_insights": parsed.get("key_insights", []),
                "markdown": raw,
            },
            confidence_score=0.9,
        )

    def supported_block_types(self) -> list[BlockType]:
        return [BlockType.TABLE]

    @staticmethod
    def _parse_markdown_table(md: str) -> tuple[str, str]:
        """Return (schema_summary, first-5-rows markdown subsample)."""
        lines = [line.strip() for line in md.strip().splitlines() if line.strip()]
        if len(lines) < 2:  # noqa: PLR2004
            return "", md[:500]

        header_cells = [c.strip() for c in lines[0].split("|") if c.strip()]
        schema = ", ".join(f'"{c}"' for c in header_cells)
        sample_lines = [lines[0], lines[1]] + lines[2:7]
        return schema, "\n".join(sample_lines)
