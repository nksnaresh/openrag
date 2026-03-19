from __future__ import annotations

import time
import os
import shutil
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Request, Response
from typing import List

from openrag.core import OpenRAG
from openrag.api.models import QueryRequest, QueryResponse, IngestResponse, Citation
from openrag.observability.metrics import MetricsManager

router = APIRouter()

# Dependency to get the OpenRAG instance from app state
def get_rag(request: Request):
    rag = getattr(request.app.state, "rag", None)
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")
    return rag

@router.get("/metrics")
async def get_metrics():
    """Endpoint for Prometheus scraping."""
    return Response(content=MetricsManager.get_latest(), media_type=MetricsManager.content_type())

@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest, rag: OpenRAG = Depends(get_rag)):
    start_time = time.time()
    print(f"DEBUG: API Querying: {request.text}")
    try:
        result = await rag.query(
            text=request.text, 
            namespace=request.namespace,
            top_k=request.top_k
        )
        print(f"DEBUG: Query success. Answer length: {len(result.answer)}")
        
        # Convert core citations to API citations
        citations = [
            Citation(
                source_path=c.document_title,
                score=c.score,
                content=c.excerpt,
                block_type=c.block_type.value if hasattr(c.block_type, "value") else str(c.block_type),
                page_number=c.page_number
            )
            for c in result.citations
        ]
        
        return QueryResponse(
            answer=result.answer,
            citations=citations,
            latency_ms=(time.time() - start_time) * 1000
        )
    except Exception as e:
        import traceback
        print(f"ERROR: Query failed: {e}")
        traceback.print_exc()
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ingest", response_model=IngestResponse)
async def ingest_file(file: UploadFile = File(...), rag: OpenRAG = Depends(get_rag)):
    start_time = time.time()
    print(f"DEBUG: API Ingesting {file.filename}")
    try:
        # Save uploaded file to a temporary directory for processing
        temp_dir = Path("temp_uploads").absolute()
        temp_dir.mkdir(exist_ok=True)
        file_path = temp_dir / file.filename
        
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"DEBUG: Saved to {file_path}")
        
        res = await rag.ingest(str(file_path))
        print(f"DEBUG: Engine ingest result: {res.status}")
        
        if res.status == "failed":
            print(f"DEBUG: Ingest failure reason: {res.reason}")
            raise HTTPException(status_code=500, detail=res.reason)
            
        return IngestResponse(
            document_id=res.document_id or "",
            status=res.status,
            block_count=res.block_count
        )
    except Exception as e:
        import traceback
        print(f"ERROR: Ingestion failed: {e}")
        traceback.print_exc()
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup temp file
        if 'file_path' in locals() and file_path.exists():
            file_path.unlink()

@router.get("/health")
async def health():
    return {"status": "ok"}
