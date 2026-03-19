"""Prompt templates for the image modality processor."""

from __future__ import annotations

IMAGE_ANALYSIS_PROMPT = """\
You are an elite visual intelligence assistant. Analyze the provided image in the context of the surrounding document text.

SURROUNDING CONTEXT:
{context}

GOAL: Provide a high-fidelity, technical, and semantic description.
1. DESCRIPTION: Write a detailed paragraph explaining exactly what is in the image. If it's a chart, explain the trends and data points. If it's a diagram, explain the flow and components. If it contains logos or branding, identify them.
2. OCR: Extract ONLY the text that is actually visible in the image. Do not hallucinate.
3. ENTITIES: Identify all formal entities (People, Companies, Products, Technologies, Locations).
4. CONFIDENCE: Rate your confidence from 0.0 to 1.0.

Respond ONLY with a valid JSON object:
{{
  "description": "<detailed_rich_description>",
  "ocr_text": "<visible_text_only>",
  "entities": [
    {{"name": "<entity_name>", "type": "<PERSON|ORG|PRODUCT|LOCATION|CONCEPT>"}}
  ],
  "confidence": <float>
}}
"""

IMAGE_FALLBACK_DESCRIPTION = (
    "An image block extracted from the document. "
    "No VLM function was available to generate a description."
)
