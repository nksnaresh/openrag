"""Code modality processor.

Extracts AST metadata from CODE blocks and provides semantic intent
via LLM. The LLM call is optional — AST metadata alone is sufficient
for embedding without an LLM function.
"""

from __future__ import annotations

from openrag.models.content import BlockType, ContentBlock
from openrag.models.processing import EntityCandidate, EntityType, ProcessedBlock, ProcessingContext
from openrag.processors.base import BaseModalityProcessor
from openrag.prompts.code import CODE_ANALYSIS_PROMPT


class CodeProcessor(BaseModalityProcessor):
    """Process CODE blocks into ProcessedBlocks with semantic descriptions.

    Uses metadata injected by parsers (function name, docstring, language)
    to build a rich embedding_text. Optionally calls LLM for semantic intent.
    """

    async def process(
        self, block: ContentBlock, context: ProcessingContext
    ) -> ProcessedBlock:
        meta = block.metadata
        code_snippet = str(block.raw_content)
        language = str(meta.get("language", "unknown"))
        name = str(meta.get("name", ""))
        kind = str(meta.get("kind", "function"))
        docstring = str(meta.get("docstring", ""))
        file_path = str(meta.get("file_path", ""))

        # Build base description from parsed metadata (no LLM needed)
        base_desc = (
            f"{kind.capitalize()} `{name}` in {language}"
            if name
            else f"{language} code block"
        )
        if docstring:
            base_desc = f"{base_desc}: {docstring[:300]}"

        # Entity candidates from metadata (function/class names)
        entities: list[EntityCandidate] = []
        if name:
            etype = EntityType.CLASS if kind == "class" else EntityType.CONCEPT
            entities.append(EntityCandidate(
                name=name,
                canonical_name=name.lower(),
                entity_type=etype,
                confidence=1.0,
            ))

        if context.llm_func is None:
            return ProcessedBlock(
                source_block=block,
                natural_language_description=base_desc,
                embedding_text=f"{base_desc}\n{code_snippet[:500]}".strip(),
                entity_candidates=entities,
                structured_data={
                    "language": language,
                    "name": name,
                    "kind": kind,
                    "docstring": docstring,
                },
                confidence_score=0.85,
            )

        prompt = CODE_ANALYSIS_PROMPT.format(
            language=language,
            file_path=file_path or "unknown",
            code_snippet=code_snippet[:2000],
            docstring=docstring or "(none)",
        )
        raw_response = await context.llm_func(prompt)
        parsed = self._extract_json(raw_response)

        semantic_intent = str(parsed.get("semantic_intent", base_desc))
        algo_category = str(parsed.get("algorithmic_category", "other"))

        raw_ents = parsed.get("entities", None)
        for e in (raw_ents if isinstance(raw_ents, list) else []):
            if isinstance(e, dict) and e.get("name"):
                entities.append(EntityCandidate(
                    name=str(e["name"]),
                    canonical_name=str(e["name"]).lower(),
                    entity_type=EntityType.CONCEPT,
                    confidence=0.8,
                ))

        embedding_text = (
            f"{semantic_intent}\n"
            f"Category: {algo_category}\n"
            f"{code_snippet[:500]}"
        ).strip()

        return ProcessedBlock(
            source_block=block,
            natural_language_description=semantic_intent,
            embedding_text=embedding_text,
            entity_candidates=entities,
            structured_data={
                "language": language,
                "name": name,
                "kind": kind,
                "algorithmic_category": algo_category,
                "complexity": parsed.get("complexity", "unknown"),
                "tags": parsed.get("tags", []),
            },
            confidence_score=0.9,
        )

    def supported_block_types(self) -> list[BlockType]:
        return [BlockType.CODE]
