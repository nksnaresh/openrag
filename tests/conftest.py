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
from openrag.models.processing import ProcessedBlock


_DOC_ID = "sha256-test-doc"


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
        document_id=_DOC_ID,
        block_id="block-text-001",
        block_type=BlockType.TEXT,
        sequence_index=0,
        raw_content="OpenRAG is a multimodal RAG framework.",
        page_number=1,
    )


@pytest.fixture
def sample_image_block() -> ContentBlock:
    return ContentBlock(
        document_id=_DOC_ID,
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
        document_id=_DOC_ID,
        source_path="/tmp/test_doc.pdf",
        tenant_id="test-tenant",
        metadata=DocumentMeta(title="Test Document", page_count=5),
        blocks=[sample_text_block, sample_image_block],
    )


@pytest.fixture
def sample_processed_block(sample_text_block) -> ProcessedBlock:
    """A ready-made ProcessedBlock for tests that need one."""
    return ProcessedBlock(
        source_block=sample_text_block,
        natural_language_description="OpenRAG is a multimodal RAG framework.",
        embedding_text="OpenRAG is a multimodal RAG framework.",
        confidence_score=1.0,
    )
