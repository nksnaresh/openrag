"""Unit tests for the deduplicator."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import pytest

from openrag.ingestion.deduplicator import compute_file_hash, is_duplicate
from openrag.storage.document.sqlite import SQLiteDocumentAdapter
from tests.unit.test_storage import _make_payload


class TestComputeFileHash:
    @pytest.mark.asyncio
    async def test_returns_sha256_hex(self) -> None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as f:
            f.write(b"hello openrag")
            path = f.name
        result = await compute_file_hash(path)
        expected = hashlib.sha256(b"hello openrag").hexdigest()
        assert result == expected

    @pytest.mark.asyncio
    async def test_empty_file_returns_hash(self) -> None:
        with tempfile.NamedTemporaryFile(delete=False) as f:
            path = f.name
        result = await compute_file_hash(path)
        expected = hashlib.sha256(b"").hexdigest()
        assert result == expected

    @pytest.mark.asyncio
    async def test_different_content_different_hash(self) -> None:
        with tempfile.NamedTemporaryFile(delete=False) as f1:
            f1.write(b"file A")
            p1 = f1.name
        with tempfile.NamedTemporaryFile(delete=False) as f2:
            f2.write(b"file B")
            p2 = f2.name
        h1 = await compute_file_hash(p1)
        h2 = await compute_file_hash(p2)
        assert h1 != h2


class TestIsDuplicate:
    @pytest.fixture()
    def adapter(self, tmp_path: Path) -> SQLiteDocumentAdapter:
        class _Cfg:
            url = f"sqlite+aiosqlite:///{tmp_path}/dup_test.db"
        return SQLiteDocumentAdapter(_Cfg())

    @pytest.mark.asyncio
    async def test_returns_false_for_new_hash(
        self, adapter: SQLiteDocumentAdapter
    ) -> None:
        await adapter.initialize()
        result = await is_duplicate("newhash123", "ns1", adapter)
        assert result is False

    @pytest.mark.asyncio
    async def test_returns_true_for_existing_hash(
        self, adapter: SQLiteDocumentAdapter
    ) -> None:
        await adapter.initialize()
        payload = _make_payload("existinghash")
        await adapter.save_document(payload, "ns1")
        result = await is_duplicate("existinghash", "ns1", adapter)
        assert result is True

    @pytest.mark.asyncio
    async def test_namespace_isolated(
        self, adapter: SQLiteDocumentAdapter
    ) -> None:
        await adapter.initialize()
        payload = _make_payload("sharedhash")
        await adapter.save_document(payload, "ns1")
        # Different namespace should not match
        result = await is_duplicate("sharedhash", "ns2", adapter)
        assert result is False
