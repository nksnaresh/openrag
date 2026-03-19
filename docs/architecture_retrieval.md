# Component Architecture: Retrieval & Reasoning

The Retrieval Layer turns indexed data back into intelligent answers. It is powered by a high-performance Hybrid Search engine and Gemini synthesis.

## 1. Query Orchestration (`openrag/query/orchestrator.py`)
The `QueryOrchestrator` coordinates the search process:
- **Preprocessing**: Handles query expansion and HyDE (Hypothetical Document Embeddings).
- **Hybrid Retrieval**: Executes concurrent searches across multiple engines.
- **Synthesis**: Combines retrieved chunks into a prompt for Gemini to generate the final answer.

## 2. Hybrid Search Engine (`openrag/search/hybrid_search.py`)
The system uses **Reciprocal Rank Fusion (RRF)** to combine three distinct search signals:
1. **Vector (Dense)**: Semantic similarity using Gemini embeddings.
2. **BM25 (Sparse)**: Keyword-level precision using document token indices.
3. **Graph (Structural)**: Entity-hop traversals via the Knowledge Graph.

## 3. Reranker (`openrag/search/reranker.py`)
Top-K candidates from the hybrid search are passed through a Cross-Encoder reranker. This provides a final "relevance score" to ensure the most critical context is presented first to the LLM.

## 4. Synthesis & Citations
The answer synthesizer generates natural language responses while maintaining strict provenance. Every fact is mapped back to its source `document_id` and `page_number`, resulting in numbered citations (`[1], [2]`) in the output.

---
> **File Reference**: [openrag/query/orchestrator.py](file:///Users/nareshsingh/MEGA-P/dev/openrag/openrag/query/orchestrator.py)
