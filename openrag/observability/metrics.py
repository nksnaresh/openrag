"""Metrics management for OpenRAG using Prometheus."""

from prometheus_client import Counter, Histogram, CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST

class MetricsManager:
    _registry = CollectorRegistry()

    # Ingestion Metrics
    INGEST_DURATION = Histogram(
        "openrag_ingest_duration_seconds",
        "Time spent in document ingestion",
        ["tenant", "status"],
        registry=_registry
    )
    INGEST_DOCS = Counter(
        "openrag_ingest_documents_total",
        "Total documents ingested",
        ["tenant", "status"],
        registry=_registry
    )
    INGEST_BLOCKS = Counter(
        "openrag_ingest_blocks_total",
        "Total content blocks processed",
        ["tenant", "type"],
        registry=_registry
    )

    # Query Metrics
    QUERY_DURATION = Histogram(
        "openrag_query_duration_seconds",
        "Time spent in query execution",
        ["tenant", "mode"],
        registry=_registry
    )
    QUERY_REQUESTS = Counter(
        "openrag_query_requests_total",
        "Total query requests",
        ["tenant", "mode", "status"],
        registry=_registry
    )
    TOKEN_USAGE = Counter(
        "openrag_llm_tokens_total",
        "Total LLM tokens consumed",
        ["tenant", "model", "type"],   # type: prompt or completion
        registry=_registry
    )

    @classmethod
    def get_latest(cls) -> bytes:
        return generate_latest(cls._registry)

    @classmethod
    def content_type(cls) -> str:
        return CONTENT_TYPE_LATEST
