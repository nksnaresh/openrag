import os
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI
import google.generativeai as genai
from dotenv import load_dotenv

# Ensure logs flush immediately
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Load environment variables from .env file
load_dotenv()

from openrag.core import OpenRAG
from openrag.config import OpenRAGConfig
from openrag.api.routes import router
from openrag.registry import AdapterRegistry
from openrag.observability.tracing import setup_tracing
from openrag.observability.metrics import MetricsManager
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

# Import Gemini adapter to register it
import openrag.embeddings.gemini

# Global RAG instance
rag_instance: OpenRAG | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 0. Setup Tracing
    setup_tracing("openrag-api")
    
    # 1. Configuration Check
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("WARNING: GEMINI_API_KEY not found in environment. Real AI features will fail.")
    else:
        genai.configure(api_key=api_key)

    working_dir = os.path.abspath("./openrag_data")
    os.makedirs(working_dir, exist_ok=True)
    db_path = os.path.join(working_dir, "openrag_store.db")

    config = OpenRAGConfig(
        namespace="default",
        working_dir=working_dir
    )

    # Configure open source offline embedding providers — use NPZ vector DB for persistence
    config.embedding.provider = "huggingface"
    config.embedding.model = "BAAI/bge-base-en-v1.5"
    config.embedding.dimensions = 768
    config.llm.provider = "gemini"
    config.llm.model = "gemini-flash-latest"
    config.rerank.enabled = False
    config.vector_db.adapter = "npz"  # ← persistent across restarts

    # Ensure Doc Store is inside working_dir
    config.document_store.url = f"sqlite+aiosqlite:///{db_path}"

    # ── Cache the HuggingFace embedding adapter at startup (not per-call) ──────────
    _emb_adapter = AdapterRegistry.get_embedding(config.embedding.provider)(config.embedding)

    async def hf_emb_func(texts: list[str]) -> list[list[float]]:
        vectors = await _emb_adapter.embed(texts)
        return [v.tolist() for v in vectors]

    async def gemini_llm_func(prompt: str) -> str:
        if not api_key:
            return "[Error: GEMINI_API_KEY not set]"
        model = genai.GenerativeModel(config.llm.model)
        response = await model.generate_content_async(prompt)
        
        # Record Metrics
        if hasattr(response, "usage_metadata"):
            usage = response.usage_metadata
            MetricsManager.TOKEN_USAGE.labels(
                tenant=config.tenant_id,
                model=config.llm.model,
                type="prompt"
            ).inc(usage.prompt_token_count)
            MetricsManager.TOKEN_USAGE.labels(
                tenant=config.tenant_id,
                model=config.llm.model,
                type="completion"
            ).inc(usage.candidates_token_count)
            
        return response.text

    # ── Quick Gemini Vision function for image/table processors ───────────────
    async def gemini_vlm_func(prompt: str, image_b64: str) -> str:
        if not api_key:
            return "[Error: GEMINI_API_KEY not set]"
        import base64
        model = genai.GenerativeModel("gemini-flash-latest")
        image_part = {"mime_type": "image/png", "data": base64.b64decode(image_b64)}
        response = await model.generate_content_async([prompt, image_part])
        return response.text

    # 2. Initialize
    rag = OpenRAG(
        config,
        embedding_func=hf_emb_func,
        llm_func=gemini_llm_func,
        vlm_func=gemini_vlm_func,
    )
    await rag.initialize()

    app.state.rag = rag
    global rag_instance
    rag_instance = rag

    print(f"INFO: OpenRAG initialized (NPZ vector DB at {working_dir}/vectors/, Offline HuggingFace Embeddings, Gemini LLM)")

    yield

    # 3. Shutdown
    if rag:
        await rag.close()
        print("INFO: OpenRAG shut down")

app = FastAPI(title="OpenRAG API", lifespan=lifespan)

# Enable automatic tracing for FastAPI
FastAPIInstrumentor.instrument_app(app)

app.include_router(router, prefix="/api/v1")

print("DEBUG: Registered Routes:")
for route in app.routes:
    print(f"  {route.path} [{route.methods if hasattr(route, 'methods') else 'N/A'}]")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
