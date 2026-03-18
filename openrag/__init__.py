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
