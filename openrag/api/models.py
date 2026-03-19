from __future__ import annotations

from pydantic import BaseModel, Field
from typing import List, Optional

class QueryRequest(BaseModel):
    text: str = Field(..., description="The question to ask the RAG engine.")
    namespace: str = Field("default", description="The namespace to search within.")
    top_k: int = Field(5, description="Number of blocks to retrieve.")

class Citation(BaseModel):
    source_path: str
    score: float
    content: str
    block_type: str
    page_number: Optional[int] = None

class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]
    latency_ms: float

class IngestResponse(BaseModel):
    document_id: str
    status: str
    block_count: int
