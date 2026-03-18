"""Image modality processor.

Encodes image bytes into base64 and sends them to a VLM (Vision Language
Model) function injected via ProcessingContext. Returns a ProcessedBlock
with a natural language description, OCR text, and entity candidates.

The VLM function is expected to match the signature:
    async def vlm_func(prompt: str, image_b64: str) -> str
"""

from __future__ import annotations

import base64

from openrag.models.content import BlockType, ContentBlock
from openrag.models.processing import EntityCandidate, EntityType, ProcessedBlock, ProcessingContext
from openrag.pipeline.context_enricher import ContextEnricher
from openrag.processors.base import BaseModalityProcessor
from openrag.prompts.image import IMAGE_ANALYSIS_PROMPT, IMAGE_FALLBACK_DESCRIPTION

_enricher = ContextEnricher()


class ImageProcessor(BaseModalityProcessor):
    """Process IMAGE blocks via a VLM function.

    Falls back to a placeholder description if no VLM function is available
    (``context.vlm_func is None``).
    """

    async def process(
        self, block: ContentBlock, context: ProcessingContext
    ) -> ProcessedBlock:
        raw = block.raw_content
        img_bytes: bytes = bytes(raw) if isinstance(raw, (bytes, bytearray)) else b""

        if not img_bytes or context.vlm_func is None:
            return ProcessedBlock(
                source_block=block,
                natural_language_description=IMAGE_FALLBACK_DESCRIPTION,
                embedding_text=IMAGE_FALLBACK_DESCRIPTION,
                entity_candidates=[],
                confidence_score=0.0,
            )

        # Enrich with surrounding context
        surrounding = _enricher.enrich(block, context.payload, context.context_config)
        prompt = IMAGE_ANALYSIS_PROMPT.format(context=surrounding or "(no surrounding context)")
        b64 = base64.b64encode(img_bytes).decode("ascii")

        raw_response = await context.vlm_func(prompt, image_b64=b64)
        parsed = self._extract_json(raw_response)

        description = str(parsed.get("description", IMAGE_FALLBACK_DESCRIPTION))
        ocr_text = str(parsed.get("ocr_text", ""))
        _conf = parsed.get("confidence", 0.9)
        confidence = float(_conf) if isinstance(_conf, (int, float)) else 0.9
        embedding_text = f"{description} {ocr_text}".strip()

        raw_ents = parsed.get("entities", None)
        ent_list = raw_ents if isinstance(raw_ents, list) else []
        entities = [
            EntityCandidate(
                name=str(e.get("name", "")),
                canonical_name=str(e.get("name", "")).lower(),
                entity_type=self._map_type(str(e.get("type", "CONCEPT"))),
                confidence=confidence,
            )
            for e in ent_list
            if isinstance(e, dict) and e.get("name")
        ]

        return ProcessedBlock(
            source_block=block,
            natural_language_description=description,
            embedding_text=embedding_text,
            entity_candidates=entities,
            structured_data={"ocr_text": ocr_text},
            confidence_score=confidence,
        )

    def supported_block_types(self) -> list[BlockType]:
        return [BlockType.IMAGE]

    @staticmethod
    def _map_type(raw: str) -> EntityType:
        mapping = {
            "PERSON": EntityType.PERSON,
            "ORG": EntityType.ORGANIZATION,
            "ORGANIZATION": EntityType.ORGANIZATION,
            "PRODUCT": EntityType.PRODUCT,
            "LOCATION": EntityType.LOCATION,
            "CONCEPT": EntityType.CONCEPT,
        }
        return mapping.get(raw.upper(), EntityType.CONCEPT)
