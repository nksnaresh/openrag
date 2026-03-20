"""Google Gemini Embedding Adapter."""

from __future__ import annotations

import os
import asyncio
import time
from typing import TYPE_CHECKING
import numpy as np

try:
    import google.generativeai as genai
except ImportError:
    genai = None

from openrag.embeddings.base import BaseEmbeddingAdapter
from openrag.registry import AdapterRegistry

if TYPE_CHECKING:
    from openrag.config import EmbeddingConfig


class GeminiEmbeddingAdapter(BaseEmbeddingAdapter):
    """Embedding adapter using Google Gemini API."""

    def __init__(self, config: EmbeddingConfig) -> None:
        self.config = config
        # Default to gemini-embedding-001 if not specified or if an OpenAI model is leaked in
        if config.model and (config.model.startswith("models/") or "embedding" in config.model.lower()) and "text-embedding-3" not in config.model:
            self.model_name = config.model
        else:
            self.model_name = "models/gemini-embedding-001"
        
        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)

    async def embed(self, texts: list[str]) -> list[np.ndarray]:
        """Generate embeddings using google-generativeai."""
        if genai is None:
            raise ImportError("google-generativeai not installed.")
        
        # genai.embed_content is blocking, so we run it in a thread
        def _get_embeddings():
            max_retries = 5
            base_delay = 2.0
            
            for attempt in range(max_retries):
                try:
                    result = genai.embed_content(
                        model=self.model_name,
                        content=texts,
                        task_type="retrieval_document"
                    )
                    return [np.array(e) for e in result["embedding"]]
                except Exception as e:
                    if "429" in str(e) or "quota" in str(e).lower() or "exhausted" in str(e).lower():
                        if attempt == max_retries - 1:
                            raise
                        time.sleep(base_delay * (2 ** attempt))
                    else:
                        raise
            return []
            
        # Add a tiny base spacing to prevent immediate overwhelming
        await asyncio.sleep(0.5)
        return await asyncio.to_thread(_get_embeddings)

    def dimension(self) -> int:
        # text-embedding-004 defaults to 768
        return self.config.dimensions if self.config.dimensions else 768

# Register the adapter
AdapterRegistry.register_embedding("gemini", GeminiEmbeddingAdapter)
