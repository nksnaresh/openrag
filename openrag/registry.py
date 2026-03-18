"""
Adapter Registry — central registry for all pluggable OpenRAG adapters.

Every swappable component (parsers, vector DBs, graph DBs, document stores,
embedding providers) registers itself here. The registry is populated at
package import time by the built-in adapters, and can be extended by
third-party plugins.

Usage::

    from openrag.registry import AdapterRegistry

    # Register a custom vector DB adapter
    AdapterRegistry.register_vector_db("my_db", MyVectorDBAdapter)

    # Retrieve an adapter class
    cls = AdapterRegistry.get_vector_db("my_db")
    adapter = cls(config)
"""

from __future__ import annotations


class RegistryError(Exception):
    """Raised when an adapter name is not found in the registry."""


class AdapterRegistry:
    """Central registry for all pluggable OpenRAG adapters."""

    _parsers: dict[str, type] = {}
    _vector_dbs: dict[str, type] = {}
    _graph_dbs: dict[str, type] = {}
    _doc_stores: dict[str, type] = {}
    _embeddings: dict[str, type] = {}
    _processors: dict[str, type] = {}

    # ── Parser adapters ────────────────────────────────────────────────────────

    @classmethod
    def register_parser(cls, name: str, adapter_cls: type) -> None:
        """Register a parser adapter under a name (usually a file extension)."""
        cls._parsers[name.lower().lstrip(".")] = adapter_cls

    @classmethod
    def get_parser(cls, name: str) -> type:
        key = name.lower().lstrip(".")
        if key not in cls._parsers:
            raise RegistryError(
                f"No parser registered for '{key}'. "
                f"Available: {sorted(cls._parsers)}"
            )
        return cls._parsers[key]

    @classmethod
    def list_parsers(cls) -> list[str]:
        return sorted(cls._parsers)

    # ── Vector DB adapters ─────────────────────────────────────────────────────

    @classmethod
    def register_vector_db(cls, name: str, adapter_cls: type) -> None:
        cls._vector_dbs[name.lower()] = adapter_cls

    @classmethod
    def get_vector_db(cls, name: str) -> type:
        key = name.lower()
        if key not in cls._vector_dbs:
            raise RegistryError(
                f"No vector DB adapter registered for '{key}'. "
                f"Available: {sorted(cls._vector_dbs)}"
            )
        return cls._vector_dbs[key]

    @classmethod
    def list_vector_dbs(cls) -> list[str]:
        return sorted(cls._vector_dbs)

    # ── Graph DB adapters ──────────────────────────────────────────────────────

    @classmethod
    def register_graph_db(cls, name: str, adapter_cls: type) -> None:
        cls._graph_dbs[name.lower()] = adapter_cls

    @classmethod
    def get_graph_db(cls, name: str) -> type:
        key = name.lower()
        if key not in cls._graph_dbs:
            raise RegistryError(
                f"No graph DB adapter registered for '{key}'. "
                f"Available: {sorted(cls._graph_dbs)}"
            )
        return cls._graph_dbs[key]

    @classmethod
    def list_graph_dbs(cls) -> list[str]:
        return sorted(cls._graph_dbs)

    # ── Document store adapters ────────────────────────────────────────────────

    @classmethod
    def register_doc_store(cls, name: str, adapter_cls: type) -> None:
        cls._doc_stores[name.lower()] = adapter_cls

    @classmethod
    def get_doc_store(cls, name: str) -> type:
        key = name.lower()
        if key not in cls._doc_stores:
            raise RegistryError(
                f"No document store adapter registered for '{key}'. "
                f"Available: {sorted(cls._doc_stores)}"
            )
        return cls._doc_stores[key]

    # ── Embedding adapters ─────────────────────────────────────────────────────

    @classmethod
    def register_embedding(cls, name: str, adapter_cls: type) -> None:
        cls._embeddings[name.lower()] = adapter_cls

    @classmethod
    def get_embedding(cls, name: str) -> type:
        key = name.lower()
        if key not in cls._embeddings:
            raise RegistryError(
                f"No embedding adapter registered for '{key}'. "
                f"Available: {sorted(cls._embeddings)}"
            )
        return cls._embeddings[key]

    # ── Modal processor adapters ───────────────────────────────────────────────

    @classmethod
    def register_processor(cls, block_type: str, processor_cls: type) -> None:
        cls._processors[block_type.lower()] = processor_cls

    @classmethod
    def get_processor(cls, block_type: str) -> type:
        key = block_type.lower()
        if key not in cls._processors:
            raise RegistryError(
                f"No processor registered for block type '{key}'. "
                f"Available: {sorted(cls._processors)}"
            )
        return cls._processors[key]

    @classmethod
    def list_processors(cls) -> list[str]:
        return sorted(cls._processors)

    # ── Diagnostics ────────────────────────────────────────────────────────────

    @classmethod
    def summary(cls) -> dict[str, list[str]]:
        """Return a summary of all registered adapters for health checks."""
        return {
            "parsers": cls.list_parsers(),
            "vector_dbs": cls.list_vector_dbs(),
            "graph_dbs": cls.list_graph_dbs(),
            "doc_stores": sorted(cls._doc_stores),
            "embeddings": sorted(cls._embeddings),
            "processors": cls.list_processors(),
        }
