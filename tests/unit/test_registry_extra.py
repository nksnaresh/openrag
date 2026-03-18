"""Unit tests for the AdapterRegistry."""

import pytest
from unittest.mock import MagicMock
from openrag.registry import AdapterRegistry


def test_adapter_registry_full_cycle() -> None:
    registry = AdapterRegistry()
    
    # Test Parser registration
    mock_parser = MagicMock()
    registry.register_parser(".test", mock_parser)
    assert registry.get_parser(".test") == mock_parser
    
    # Test Vector DB registration
    mock_vdb = MagicMock()
    registry.register_vector_db("test_vdb", mock_vdb)
    assert registry.get_vector_db("test_vdb") == mock_vdb
    
    # Test Graph DB registration
    mock_gdb = MagicMock()
    registry.register_graph_db("test_gdb", mock_gdb)
    assert registry.get_graph_db("test_gdb") == mock_gdb
    
    # Test Doc Store registration
    mock_doc = MagicMock()
    registry.register_doc_store("test_doc", mock_doc)
    assert registry.get_doc_store("test_doc") == mock_doc
    
    # Test Embedding registration
    mock_emb = MagicMock()
    registry.register_embedding("test_emb", mock_emb)
    assert registry.get_embedding("test_emb") == mock_emb
