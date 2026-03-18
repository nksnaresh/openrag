"""Shared pytest fixtures for all OpenRAG tests."""

from __future__ import annotations

import pytest

from openrag.config import OpenRAGConfig
from openrag.models.content import (
    BlockType,
    ContentBlock,
    ContentPayload,
    DocumentMeta,
)


@pytest.fixture
def base_config(tmp_path) -> OpenRAGConfig:
    """Minimal in-memory config suitable for unit tests."""
    return OpenRAGConfig(
        namespace="test-ns",
        tenant_id="test-tenant",
        working_dir=str(tmp_path / "openrag_storage"),
    )


@pytest.fixture
def sample_text_block() -> ContentBlock:
    return ContentBlock(
        block_id="block-text-001",
        block_type=BlockType.TEXT,
        sequence_index=0,
        raw_content="OpenRAG is a multimodal RAG framework.",
        page_number=1,
    )


@pytest.fixture
def sample_image_block() -> ContentBlock:
    return ContentBlock(
        block_id="block-img-001",
        block_type=BlockType.IMAGE,
        sequence_index=1,
        raw_content=b"\xff\xd8\xff\xe0",  # JPEG magic bytes
        page_number=1,
        metadata={"caption": "Figure 1: Architecture overview"},
    )


@pytest.fixture
def sample_payload(sample_text_block, sample_image_block) -> ContentPayload:
    return ContentPayload(
        document_id="sha256-test-doc",
        source_path="/tmp/test_doc.pdf",
        tenant_id="test-tenant",
        metadata=DocumentMeta(title="Test Document", page_count=5),
        blocks=[sample_text_block, sample_image_block],
    )
