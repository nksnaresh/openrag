# Component Architecture: Ingestion Pipeline

The Ingestion Pipeline is responsible for transforming raw document files into structured, searchable knowledge stored in vector and graph formats.

## 1. Orchestration (`openrag/ingestion/orchestrator.py`)
The `IngestionOrchestrator` manages the end-to-end lifecycle of a document:
- **Deduplication**: Checks content hashes to prevent redundant processing.
- **Parsing**: Dispatches to specific adapters (PDF, Text, Code) based on file type.
- **Task Scheduling**: Uses an internal DAG engine to parallelize modality-agnostic steps.

## 2. DAG Pipeline Engine (`openrag/pipeline/dag_engine.py`)
The pipeline is modeled as a Directed Acyclic Graph.
- **Modality Splitting**: Documents are broken into blocks (Text, Image, Table, etc.).
- **Parallel Processing**: Different processors (e.g., `ImageProcessor` and `TextProcessor`) run concurrently if hardware/API limits allow.

## 3. Modality Processors
Each processor is specialized for a content type:
- **Vision (Gemini)**: Extracts semantic meaning and OCR from images.
- **Table**: Converts visual/textual tables into rich Markdown and structured summaries.
- **Equation**: Simplifies LaTeX into natural language descriptions for embedding.

## 4. Context Enrichment
Before embedding, each chunk is enriched with its "surroundings" (e.g., current section heading, adjacent paragraphs) to ensure the vector representation captures the full semantic context.

---
> **File Reference**: [openrag/ingestion/orchestrator.py](file:///Users/nareshsingh/MEGA-P/dev/openrag/openrag/ingestion/orchestrator.py)
