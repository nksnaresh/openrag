"""LLM-based relationship extraction logic."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from openrag.models.content import BlockType

if TYPE_CHECKING:
    from openrag.models.processing import EntityCandidate, ProcessedBlock


REL_PROMPT_TEMPLATE = """
You are an expert knowledge graph engineer. Given a text snippet and a set of identified entities, your task is to identify key relationships between them.

TEXT SNIPPET:
{text}

ENTITIES:
{entities}

Identify relationships and return them in JSON format:
{{
  "relationships": [
    {{
      "source": "Entity Name 1",
      "target": "Entity Name 2",
      "type": "ONE_OF: [REFERENCES, DEFINES, IMPLEMENTS, CITES, PROVES, ILLUSTRATES, PART_OF, RELATED_TO]",
      "description": "Short explanation of the relationship"
    }}
  ]
}}
"""


class RelationshipExtractor:
    """Extracts typed relationships between entities using LLM calls.

    Can operate on a single ProcessedBlock or a set of blocks.
    """

    def __init__(self, llm_func: Any = None) -> None:  # noqa: ANN401
        self._llm_func = llm_func

    async def extract_from_block(
        self,
        block: ProcessedBlock,
        entities: list[dict[str, Any]],
        llm_func: Any = None,  # noqa: ANN401
    ) -> list[dict[str, Any]]:
        """Identify relationships between entities in a single block context."""
        func = llm_func or self._llm_func
        if not func:
            return []

        if not entities or len(entities) < 2:
            return []

        # 1. Prepare context
        text = block.natural_language_description or ""
        ent_list = [e["name"] for e in entities]
        
        prompt = REL_PROMPT_TEMPLATE.format(
            text=text,
            entities=", ".join(ent_list),
        )

        # 2. Call LLM
        try:
            response = await func(prompt)
            # Basic JSON extraction (naive)
            # Find first { and last }
            start = response.find("{")
            end = response.rfind("}")
            if start != -1 and end != -1:
                data = json.loads(response[start : end + 1])
                return data.get("relationships", [])
        except (Exception, json.JSONDecodeError):  # noqa: BLE001
            # Graceful fallback: return empty list on failure
            return []

        return []
