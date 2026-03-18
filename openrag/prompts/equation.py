"""Prompt templates for the equation modality processor."""

from __future__ import annotations

EQUATION_ANALYSIS_PROMPT = """\
You are an expert mathematician and scientist. Analyze the following mathematical equation.

LATEX SOURCE:
{latex_source}

SURROUNDING CONTEXT:
{context}

Respond ONLY with a valid JSON object:
{{
  "interpretation": "<plain-English explanation of what this equation means>",
  "variables": [
    {{"symbol": "<symbol>", "definition": "<what it represents>"}}
  ],
  "domain": "<mathematics|physics|chemistry|engineering|statistics|economics|other>",
  "complexity": "<elementary|intermediate|advanced>",
  "entities": [
    {{"name": "<entity>", "type": "<CONCEPT|ALGORITHM|THEOREM|CONSTANT>"}}
  ]
}}
"""
