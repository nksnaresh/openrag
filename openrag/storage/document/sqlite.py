"""SQLite-backed document store adapter using aiosqlite for async I/O.

Stores document records, BM25 tokens, and conversation sessions in a
single SQLite database file. Register via:

    AdapterRegistry.register_doc_store("sqlite", SQLiteDocumentAdapter)
"""

from __future__ import annotations

import json
from typing import Any

import aiosqlite

from openrag.models.content import ContentPayload
from openrag.storage.base import BaseDocumentAdapter

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS documents (
    document_id   TEXT NOT NULL,
    namespace     TEXT NOT NULL,
    source_path   TEXT NOT NULL,
    content_hash  TEXT NOT NULL,
    metadata_json TEXT,
    ingested_at   TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (document_id, namespace)
);

CREATE INDEX IF NOT EXISTS idx_docs_ns_hash
    ON documents (namespace, content_hash);

CREATE TABLE IF NOT EXISTS bm25_tokens (
    chunk_id    TEXT NOT NULL,
    namespace   TEXT NOT NULL,
    tokens_json TEXT NOT NULL,
    PRIMARY KEY (namespace, chunk_id)
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT NOT NULL PRIMARY KEY,
    turns_json TEXT NOT NULL,
    updated_at TEXT DEFAULT (datetime('now'))
);
"""


class SQLiteDocumentAdapter(BaseDocumentAdapter):
    """Async SQLite document store for document records, BM25, and sessions.

    Args:
        config: expects a config object with a `url` attribute of the form
                ``sqlite+aiosqlite:///path/to/db.sqlite`` (the
                ``sqlite+aiosqlite://`` prefix is stripped internally).
                Falls back to ``./openrag_store.db`` if config is None.
    """

    def __init__(self, config: object = None, **kwargs: Any) -> None:
        raw_url: str = "sqlite+aiosqlite:///./openrag_store.db"
        if config is not None and hasattr(config, "url"):
            raw_url = config.url
        # Strip SQLAlchemy-style prefix to get a plain filesystem path
        self._db_path = raw_url.replace("sqlite+aiosqlite:///", "")

    async def initialize(self) -> None:
        """Create tables and indices if they don't exist."""
        async with aiosqlite.connect(self._db_path) as db:
            await db.executescript(_SCHEMA)
            await db.commit()

    # ── Document operations ────────────────────────────────────────────────────

    async def save_document(self, payload: ContentPayload, namespace: str) -> None:
        """Persist a document record from a ContentPayload."""
        meta_json = json.dumps(
            {
                "title": payload.metadata.title,
                "author": payload.metadata.author,
                "language": payload.metadata.language,
                "page_count": payload.metadata.page_count,
                "source_url": payload.metadata.source_url,
                "file_size_bytes": payload.metadata.file_size_bytes,
                "mime_type": payload.metadata.mime_type,
                "custom": payload.metadata.custom,
            }
        )
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                INSERT INTO documents
                    (document_id, namespace, source_path, content_hash, metadata_json)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT (document_id, namespace) DO UPDATE SET
                    source_path   = excluded.source_path,
                    content_hash  = excluded.content_hash,
                    metadata_json = excluded.metadata_json,
                    ingested_at   = datetime('now')
                """,
                (
                    payload.document_id,
                    namespace,
                    payload.source_path,
                    payload.document_id,  # document_id IS the sha256 hash
                    meta_json,
                ),
            )
            await db.commit()

    async def get_document(
        self, document_id: str, namespace: str
    ) -> dict[str, Any] | None:
        """Retrieve a document record by (document_id, namespace)."""
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM documents WHERE document_id=? AND namespace=?",
                (document_id, namespace),
            ) as cursor:
                row = await cursor.fetchone()
                if row is None:
                    return None
                return self._row_to_dict(row)

    async def find_by_hash(
        self, namespace: str, content_hash: str
    ) -> dict[str, Any] | None:
        """Return the first document whose content_hash matches in the namespace."""
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM documents WHERE namespace=? AND content_hash=? LIMIT 1",
                (namespace, content_hash),
            ) as cursor:
                row = await cursor.fetchone()
                if row is None:
                    return None
                return self._row_to_dict(row)

    async def list_documents(
        self, namespace: str, limit: int = 100, offset: int = 0
    ) -> list[dict[str, Any]]:
        """List documents in a namespace with pagination."""
        async with aiosqlite.connect(self._db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM documents WHERE namespace=? "
                "ORDER BY ingested_at DESC LIMIT ? OFFSET ?",
                (namespace, limit, offset),
            ) as cursor:
                rows = await cursor.fetchall()
                return [self._row_to_dict(r) for r in rows]

    async def close(self) -> None: pass

    async def delete_document(self, document_id: str, namespace: str) -> None:
        """Delete a document and all its BM25 tokens."""
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                "DELETE FROM documents WHERE document_id=? AND namespace=?",
                (document_id, namespace),
            )
            await db.execute(
                "DELETE FROM bm25_tokens WHERE namespace=? AND chunk_id LIKE ?",
                (namespace, f"{document_id}%"),
            )
            await db.commit()

    # ── BM25 token operations ──────────────────────────────────────────────────

    async def save_bm25_tokens(
        self, namespace: str, chunk_id: str, tokens: list[str]
    ) -> None:
        """Persist tokenised content for BM25 indexing."""
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                INSERT INTO bm25_tokens (chunk_id, namespace, tokens_json)
                VALUES (?, ?, ?)
                ON CONFLICT (namespace, chunk_id) DO UPDATE SET
                    tokens_json = excluded.tokens_json
                """,
                (chunk_id, namespace, json.dumps(tokens)),
            )
            await db.commit()

    async def get_all_bm25_tokens(
        self, namespace: str
    ) -> list[tuple[str, list[str]]]:
        """Return all (chunk_id, tokens) pairs for BM25 corpus construction."""
        async with aiosqlite.connect(self._db_path) as db, db.execute(
            "SELECT chunk_id, tokens_json FROM bm25_tokens WHERE namespace=?",
            (namespace,),
        ) as cursor:
            rows = await cursor.fetchall()
            return [(row[0], json.loads(row[1])) for row in rows]

    # ── Session operations ─────────────────────────────────────────────────────

    async def save_session(
        self, session_id: str, turns: list[dict[str, str]]
    ) -> None:
        """Persist or overwrite all turns for a conversation session."""
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                INSERT INTO sessions (session_id, turns_json)
                VALUES (?, ?)
                ON CONFLICT (session_id) DO UPDATE SET
                    turns_json = excluded.turns_json,
                    updated_at = datetime('now')
                """,
                (session_id, json.dumps(turns)),
            )
            await db.commit()

    async def get_session(self, session_id: str) -> list[dict[str, str]]:
        """Retrieve all turns for a session. Returns [] if not found."""
        async with aiosqlite.connect(self._db_path) as db, db.execute(
            "SELECT turns_json FROM sessions WHERE session_id=?",
            (session_id,),
        ) as cursor:
            row = await cursor.fetchone()
            if row is None:
                return []
            return json.loads(row[0])  # type: ignore[no-any-return]

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _row_to_dict(row: aiosqlite.Row) -> dict[str, Any]:
        """Convert a sqlite Row to a plain dict."""
        d = dict(row)
        if "metadata_json" in d and d["metadata_json"]:
            d["metadata"] = json.loads(d.pop("metadata_json"))
        return d
