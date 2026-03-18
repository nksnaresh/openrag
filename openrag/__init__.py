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

AdapterRegistry.register_vector_db("in_memory", InMemoryVectorAdapter)
AdapterRegistry.register_graph_db("networkx", NetworkXAdapter)
AdapterRegistry.register_doc_store("sqlite", SQLiteDocumentAdapter)
