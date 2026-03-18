"""Prompt templates for the code modality processor."""

from __future__ import annotations

CODE_ANALYSIS_PROMPT = """\
You are an expert software engineer. Analyze the following code block.

LANGUAGE: {language}
FILE: {file_path}

CODE:
{code_snippet}

DOCSTRING / COMMENTS:
{docstring}

Respond ONLY with a valid JSON object:
{{
  "semantic_intent": "<what this code does in plain English>",
  "algorithmic_category": "<sorting|searching|parsing|networking|ML|data-processing|other>",
  "complexity": "<O(1)|O(log n)|O(n)|O(n log n)|O(n²)|unknown>",
  "entities": [
    {{"name": "<entity>", "type": "<FUNCTION|CLASS|MODULE|ALGORITHM|LIBRARY>"}}
  ],
  "tags": ["<tag1>", "<tag2>"]
}}
"""
