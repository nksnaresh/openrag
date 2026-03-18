"""Document deduplicator for the ingestion pipeline.

Checks whether a document has already been ingested by comparing its
SHA-256 content hash against the document store.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from openrag.storage.base import BaseDocumentAdapter


async def compute_file_hash(path: str | Path) -> str:
    """Return the SHA-256 hex digest of a file's raw bytes.

    Args:
        path: Absolute path to the file on disk.

    Returns:
        64-character lowercase hex string.
    """
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


async def is_duplicate(
    content_hash: str,
    namespace: str,
    doc_store: BaseDocumentAdapter,
) -> bool:
    """Return True if a document with this hash already exists in the namespace.

    Args:
        content_hash: SHA-256 hex digest of the file bytes.
        namespace:    Tenant namespace to search.
        doc_store:    Initialized document store adapter.
    """
    existing = await doc_store.find_by_hash(namespace, content_hash)
    return existing is not None
