"""
OpenRAG — Open-Source Universal Knowledge Intelligence Framework.

Provides a full-stack multimodal RAG platform with:
- Six native content modalities (text, image, table, equation, code, audio/video)
- Built-in REST + GraphQL + WebSocket API server
- Declarative YAML pipeline configuration
- Pluggable storage adapters (vector DB, graph DB, document store)
- Multi-tenant architecture with document-level ACLs
- OpenTelemetry tracing + Prometheus metrics
"""

__version__ = "0.1.0"
__author__ = "OpenRAG Contributors"
__license__ = "Apache-2.0"

from openrag.config import OpenRAGConfig
from openrag.core import OpenRAG

__all__ = [
    "__version__",
    "OpenRAG",
    "OpenRAGConfig",
]

# ── Built-in adapter auto-registration ────────────────────────────────────────
# Register the bundled in-process adapters so that the default config values
# ("in_memory", "networkx", "sqlite") resolve without any user configuration.

from openrag.registry import AdapterRegistry  # noqa: E402
from openrag.storage.document.sqlite import SQLiteDocumentAdapter  # noqa: E402
from openrag.storage.graph.networkx import NetworkXAdapter  # noqa: E402
from openrag.storage.vector.in_memory import InMemoryVectorAdapter  # noqa: E402

from openrag.storage.vector.npz import NPZVectorAdapter  # noqa: E402

AdapterRegistry.register_vector_db("in_memory", InMemoryVectorAdapter)
AdapterRegistry.register_vector_db("npz", NPZVectorAdapter)
AdapterRegistry.register_graph_db("networkx", NetworkXAdapter)
AdapterRegistry.register_doc_store("sqlite", SQLiteDocumentAdapter)

# ── Parser auto-registration (Phase 2) ───────────────────────────────────────
from openrag.parsers.code import PlainCodeAdapter  # noqa: E402
from openrag.parsers.html import HTMLAdapter  # noqa: E402
from openrag.parsers.pdf import PyMuPDFAdapter  # noqa: E402
from openrag.parsers.text import PlainTextAdapter  # noqa: E402

AdapterRegistry.register_parser(".pdf",  PyMuPDFAdapter)
AdapterRegistry.register_parser(".txt",  PlainTextAdapter)
AdapterRegistry.register_parser(".md",   PlainTextAdapter)
AdapterRegistry.register_parser(".html", HTMLAdapter)
AdapterRegistry.register_parser(".htm",  HTMLAdapter)
for _ext in [".py", ".js", ".ts", ".go", ".java", ".rs", ".cpp", ".c"]:
    AdapterRegistry.register_parser(_ext, PlainCodeAdapter)

# Docling optional — only register if library is installed
from openrag.parsers.docling import DoclingAdapter  # noqa: E402

if DoclingAdapter.health_check():
    for _ext in [".docx", ".pptx", ".doc"]:
        AdapterRegistry.register_parser(_ext, DoclingAdapter)
# ── Embedding auto-registration (Phase 3) ────────────────────────────────────
from openrag.embeddings.huggingface import (  # noqa: E402
    HuggingFaceEmbeddingAdapter,
)
from openrag.embeddings.ollama import OllamaEmbeddingAdapter  # noqa: E402
from openrag.embeddings.openai import OpenAIEmbeddingAdapter  # noqa: E402

AdapterRegistry.register_embedding("openai", OpenAIEmbeddingAdapter)
AdapterRegistry.register_embedding("ollama", OllamaEmbeddingAdapter)
try:
    from openrag.embeddings.gemini import GeminiEmbeddingAdapter  # noqa: E402
    AdapterRegistry.register_embedding("gemini", GeminiEmbeddingAdapter)
except (ImportError, TypeError):
    pass

try:
    import sentence_transformers  # noqa: F401
    AdapterRegistry.register_embedding("huggingface", HuggingFaceEmbeddingAdapter)
except ImportError:
    pass
