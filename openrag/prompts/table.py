"""Prompt templates for the table modality processor."""

from __future__ import annotations

TABLE_ANALYSIS_PROMPT = """\
You are an expert data analyst. Analyze the following table extracted from a document.

TABLE (markdown format):
{table_markdown}

COLUMN SCHEMA:
{schema_summary}

SURROUNDING CONTEXT:
{context}

Respond ONLY with a valid JSON object:
{{
  "summary": "<natural-language description of what this table shows>",
  "domain_tags": ["<tag1>", "<tag2>"],
  "key_insights": ["<insight1>", "<insight2>"],
  "entities": [
    {{"name": "<entity>", "type": "<METRIC|PRODUCT|ORG|LOCATION|CONCEPT>"}}
  ]
}}
"""
