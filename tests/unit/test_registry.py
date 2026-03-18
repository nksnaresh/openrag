"""Unit tests for the adapter registry."""

from __future__ import annotations

import pytest

from openrag.registry import AdapterRegistry, RegistryError


class _FakeVectorDB:
    pass


class _FakeGraphDB:
    pass


class _FakeParser:
    pass


class TestAdapterRegistry:
    def setup_method(self) -> None:
        """Snapshot and clear the registry before each test."""
        self._saved = {
            "_parsers": dict(AdapterRegistry._parsers),
            "_vector_dbs": dict(AdapterRegistry._vector_dbs),
            "_graph_dbs": dict(AdapterRegistry._graph_dbs),
            "_doc_stores": dict(AdapterRegistry._doc_stores),
            "_embeddings": dict(AdapterRegistry._embeddings),
            "_processors": dict(AdapterRegistry._processors),
        }

    def teardown_method(self) -> None:
        """Restore registry state after each test."""
        for key, val in self._saved.items():
            setattr(AdapterRegistry, key, val)

    def test_register_and_get_parser(self) -> None:
        AdapterRegistry.register_parser(".pdf", _FakeParser)
        cls = AdapterRegistry.get_parser(".pdf")
        assert cls is _FakeParser

    def test_get_parser_strips_dot(self) -> None:
        AdapterRegistry.register_parser("pdf", _FakeParser)
        # Both "pdf" and ".pdf" should resolve
        assert AdapterRegistry.get_parser("pdf") is _FakeParser
        assert AdapterRegistry.get_parser(".pdf") is _FakeParser

    def test_get_parser_missing_raises(self) -> None:
        with pytest.raises(RegistryError, match="No parser registered"):
            AdapterRegistry.get_parser("xyz_unknown")

    def test_register_and_get_vector_db(self) -> None:
        AdapterRegistry.register_vector_db("fake_db", _FakeVectorDB)
        assert AdapterRegistry.get_vector_db("fake_db") is _FakeVectorDB

    def test_summary_keys(self) -> None:
        summary = AdapterRegistry.summary()
        assert set(summary.keys()) == {
            "parsers", "vector_dbs", "graph_dbs", "doc_stores", "embeddings", "processors"
        }

    def test_case_insensitive(self) -> None:
        AdapterRegistry.register_vector_db("MyDB", _FakeVectorDB)
        assert AdapterRegistry.get_vector_db("mydb") is _FakeVectorDB
        assert AdapterRegistry.get_vector_db("MYDB") is _FakeVectorDB
