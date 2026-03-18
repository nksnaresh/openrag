"""Equation modality processor.

Sends LaTeX equation strings to the LLM for English interpretation,
variable definitions, and domain classification.
"""

from __future__ import annotations

from openrag.models.content import BlockType, ContentBlock
from openrag.models.processing import EntityCandidate, EntityType, ProcessedBlock, ProcessingContext
from openrag.pipeline.context_enricher import ContextEnricher
from openrag.processors.base import BaseModalityProcessor
from openrag.prompts.equation import EQUATION_ANALYSIS_PROMPT

_enricher = ContextEnricher()


class EquationProcessor(BaseModalityProcessor):
    """Process EQUATION blocks (LaTeX strings) via LLM interpretation."""

    async def process(
        self, block: ContentBlock, context: ProcessingContext
    ) -> ProcessedBlock:
        latex_src = str(block.raw_content)
        surrounding = _enricher.enrich(block, context.payload, context.context_config)

        fallback_desc = f"Mathematical equation: {latex_src[:200]}"

        if context.llm_func is None:
            return ProcessedBlock(
                source_block=block,
                natural_language_description=fallback_desc,
                embedding_text=fallback_desc,
                entity_candidates=[],
                structured_data={"latex_source": latex_src},
                confidence_score=0.5,
            )

        prompt = EQUATION_ANALYSIS_PROMPT.format(
            latex_source=latex_src,
            context=surrounding or "(no surrounding context)",
        )
        raw_response = await context.llm_func(prompt)
        parsed = self._extract_json(raw_response)

        interpretation = str(parsed.get("interpretation", fallback_desc))
        domain = str(parsed.get("domain", "mathematics"))
        raw_vars = parsed.get("variables", None)
        variables: list[dict[str, object]] = raw_vars if isinstance(raw_vars, list) else []

        raw_ents = parsed.get("entities", None)
        ent_list = raw_ents if isinstance(raw_ents, list) else []
        entities = [
            EntityCandidate(
                name=str(e.get("name", "")),
                canonical_name=str(e.get("name", "")).lower(),
                entity_type=EntityType.CONCEPT,
                confidence=0.85,
            )
            for e in ent_list
            if isinstance(e, dict) and e.get("name")
        ]

        var_text = " ".join(
            f"{v.get('symbol', '')} is {v.get('definition', '')}"
            for v in variables
            if isinstance(v, dict)
        )
        embedding_text = f"{interpretation} {var_text}".strip()

        return ProcessedBlock(
            source_block=block,
            natural_language_description=interpretation,
            embedding_text=embedding_text,
            entity_candidates=entities,
            structured_data={
                "latex_source": latex_src,
                "domain": domain,
                "complexity": parsed.get("complexity", "unknown"),
                "variables": variables,
            },
            confidence_score=0.9,
        )

    def supported_block_types(self) -> list[BlockType]:
        return [BlockType.EQUATION]
