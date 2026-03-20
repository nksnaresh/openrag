from __future__ import annotations

import time
import os
import shutil
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Request, Response, status
from fastapi.security import OAuth2PasswordRequestForm
from typing import List, Optional
from datetime import timedelta
from openrag.api.auth import (
    Token, User, create_access_token, ACCESS_TOKEN_EXPIRE_MINUTES, 
    FAKE_USERS_DB, verify_password, get_current_user, check_admin_role
)

from openrag.core import OpenRAG
from openrag.api.models import QueryRequest, QueryResponse, IngestResponse, Citation
from openrag.observability.metrics import MetricsManager
from fastapi.responses import StreamingResponse
import json
import asyncio

router = APIRouter()

# Dependency to get the OpenRAG instance from app state
def get_rag(request: Request):
    rag = getattr(request.app.state, "rag", None)
    if rag is None:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")
    return rag

@router.get("/metrics")
async def get_metrics(current_user: User = Depends(check_admin_role)):
    """Endpoint for Prometheus scraping."""
    return Response(content=MetricsManager.get_latest(), media_type=MetricsManager.content_type())

@router.post("/auth/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user_dict = FAKE_USERS_DB.get(form_data.username)
    if not user_dict or not verify_password(form_data.password, user_dict["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user_dict["username"]}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest, rag: OpenRAG = Depends(get_rag), current_user: User = Depends(get_current_user)):
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

@router.post("/query/stream")
async def query_stream(request: QueryRequest, rag: OpenRAG = Depends(get_rag), current_user: User = Depends(get_current_user)):
    async def event_generator():
        queue = asyncio.Queue()
        
        async def on_progress(msg: str):
            await queue.put(msg)
            
        task = asyncio.create_task(rag.query(
            text=request.text,
            namespace=request.namespace,
            top_k=request.top_k,
            on_progress=on_progress
        ))
        
        # Yield progress from queue
        while not task.done() or not queue.empty():
            try:
                # Use a small timeout so we can check if task is done
                msg = await asyncio.wait_for(queue.get(), timeout=0.1)
                yield f"{msg}\n"
            except asyncio.TimeoutError:
                continue
        
        result = await task
        # Final JSON payload
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
        final_data = {
            "type": "result",
            "answer": result.answer,
            "citations": [c.dict() for c in citations],
            "latency_ms": result.latency_ms
        }
        yield f"RESULT:{json.dumps(final_data)}\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@router.post("/ingest", response_model=IngestResponse)
async def ingest_file(file: UploadFile = File(...), rag: OpenRAG = Depends(get_rag), current_user: User = Depends(check_admin_role)):
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

@router.post("/ingest/stream")
async def ingest_stream(file: UploadFile = File(...), rag: OpenRAG = Depends(get_rag), current_user: User = Depends(check_admin_role)):
    # Save to temp file first
    temp_dir = Path("temp_uploads").absolute()
    temp_dir.mkdir(exist_ok=True)
    file_path = temp_dir / file.filename
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    async def event_generator():
        queue = asyncio.Queue()
        
        async def on_progress(msg: str):
            await queue.put(msg)
            
        try:
            task = asyncio.create_task(rag.ingest(str(file_path), on_progress=on_progress))
            
            while not task.done() or not queue.empty():
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=0.1)
                    yield f"{msg}\n"
                except asyncio.TimeoutError:
                    continue
            
            res = await task
            final_data = {
                "type": "result",
                "document_id": res.document_id or "",
                "status": res.status.value if hasattr(res.status, 'value') else str(res.status),
                "block_count": getattr(res, "block_count", 0),
                "reason": getattr(res, "reason", None) or ""
            }
            yield f"RESULT:{json.dumps(final_data)}\n"
        finally:
            if file_path.exists():
                file_path.unlink()

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@router.get("/documents")
async def get_documents(namespace: str = "default", limit: int = 100, rag: OpenRAG = Depends(get_rag), current_user: User = Depends(get_current_user)):
    if hasattr(rag, "_core_components"):
        # Access from components dict or registry
        # Let's cleanly reach into the initialized state
        pass
    
    # Try looking for _doc_store on the orchestrator or initialized rag instance
    store = getattr(rag, "_doc_store", None)
    if not store and hasattr(rag, "config"):
        # Let's get it by initializing
        pass
        
    try:
        # Internal access hack for phase 5 API since OpenRAG doesn't expose doc_store via a standard getter yet.
        # It's injected into the IngestionOrchestrator by the private _get_orchestrator().
        orch = rag._get_orchestrator()
        docs = await orch._doc_store.list_documents(namespace, limit=limit)
        
        # Fallback disk check for older documents that didn't persist size to vectors
        for d in docs:
            if "metadata" not in d or d["metadata"] is None:
                d["metadata"] = {}
            if not d["metadata"].get("file_size_bytes"):
                if "source_path" in d and os.path.exists(d["source_path"]):
                    try:
                        d["metadata"]["file_size_bytes"] = os.path.getsize(d["source_path"])
                    except Exception:
                        pass
                        
        return {"documents": docs}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/documents/{document_id}")
async def delete_document(document_id: str, namespace: str = "default", rag: OpenRAG = Depends(get_rag), current_user: User = Depends(check_admin_role)):
    try:
        orch = rag._get_orchestrator()
        await orch._doc_store.delete_document(document_id, namespace)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health():
    return {"status": "ok"}

@router.get("/observability/logs")
async def get_system_logs(limit: int = 50, current_user: User = Depends(check_admin_role)):
    log_file = Path("/tmp/openrag_server.log")
    if not log_file.exists():
        return {"logs": ["[INFO] System log file not found."]}
    with log_file.open("r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
        return {"logs": [line.strip() for line in lines[-limit:] if line.strip()]}

@router.delete("/observability/logs")
async def clear_system_logs(current_user: User = Depends(check_admin_role)):
    log_file = Path("/tmp/openrag_server.log")
    if log_file.exists():
        log_file.open("w").close()
    return {"status": "cleared"}

@router.get("/observability/traces")
async def get_recent_traces(rag: OpenRAG = Depends(get_rag)):
    # Standard dummy trace payload that simulates a system execution
    import time
    now = time.time()
    
    def ts(offset):
        return time.strftime("%H:%M:%S", time.localtime(now - offset)) + f".{int((now - offset) * 1000) % 1000:03d}"
        
    return {"trace": [
        {"time": ts(2.5), "content": "<strong>Query Received:</strong> System trace initiated.", "meta": ""},
        {"time": ts(2.1), "content": "<strong>Vector Retrieval:</strong> Searched Vector DB + BM25.", "meta": "Duration: 105ms"},
        {"time": ts(0.8), "content": "<strong>Reranking:</strong> Reranker filtered to top chunks.", "meta": "Duration: 62ms"},
        {"time": ts(0.1), "content": "<strong>LLM Synthesis:</strong> Generated final answers.", "meta": "Tokens: 254"}
    ]}
