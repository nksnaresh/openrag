"""Prompt templates for the table modality processor."""

from __future__ import annotations

TABLE_ANALYSIS_PROMPT = """\
You are a senior data scientist and document analyst. Analyze the following table extracted from a professional document.

TABLE (markdown format):
{table_markdown}

COLUMN SCHEMA:
{schema_summary}

SURROUNDING CONTEXT:
{context}

GOAL: Synthesize the table data into high-value insights.
1. SUMMARY: Provide a comprehensive 2-3 sentence summary of what this table represents and its importance in the document.
2. INSIGHTS: List specific data-driven insights (e.g., "Company X grew by 20%").
3. TAGS: Provide domain-specific tags (e.g., "Financials", "Q4", "Telecomm").
4. ENTITIES: Identify key metrics or organizations mentioned in the table.

Respond ONLY with a valid JSON object:
{{
  "summary": "<rich_comprehensive_summary>",
  "domain_tags": ["<tag1>", "<tag2>"],
  "key_insights": ["<insight1>", "<insight2>"],
  "entities": [
    {{"name": "<entity>", "type": "<METRIC|PRODUCT|ORG|LOCATION|CONCEPT>"}}
  ]
}}
"""
