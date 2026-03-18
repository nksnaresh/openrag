"""Prompt templates for the image modality processor."""

from __future__ import annotations

IMAGE_ANALYSIS_PROMPT = """\
You are an expert visual analyst processing a document image.
Analyze the image and the surrounding document context carefully.

SURROUNDING CONTEXT:
{context}

Respond ONLY with a valid JSON object in this exact schema:
{{
  "description": "<rich natural-language description of the image content>",
  "ocr_text": "<any visible text in the image, empty string if none>",
  "entities": [
    {{"name": "<entity name>", "type": "<PERSON|ORG|PRODUCT|LOCATION|CONCEPT>"}}
  ],
  "confidence": <float 0.0-1.0>
}}
"""

IMAGE_FALLBACK_DESCRIPTION = (
    "An image block extracted from the document. "
    "No VLM function was available to generate a description."
)
