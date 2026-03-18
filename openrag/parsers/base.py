"""Abstract base class for all document parser adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from openrag.models.content import ContentPayload


@dataclass
class ParseOptions:
    """Options passed to every parser adapter."""

    tenant_id: str = "default"
    namespace: str = "default"
    ocr_enabled: bool = True
    language: str = "en"
    start_page: int | None = None
    end_page: int | None = None
    extra: dict[str, object] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.extra is None:
            self.extra = {}


class BaseParserAdapter(ABC):
    """
    Abstract interface for all document parser adapters.

    Implementors must extract typed ContentBlocks from a source document
    and return them as a ContentPayload. The contract enforces:
      - document_id  = SHA-256 hex of the file bytes
      - source_path  = absolute path or URL
      - ordered blocks with unique block_ids
    """

    @abstractmethod
    async def parse(self, file_path: str | Path, options: ParseOptions) -> ContentPayload:
        """
        Parse a document and return its typed content blocks.

        Args:
            file_path: Absolute path (or URL) to the source document.
            options: Parsing configuration (OCR, pages, language, etc.)

        Returns:
            ContentPayload with all extracted ContentBlocks.
        """

    @abstractmethod
    def supported_extensions(self) -> list[str]:
        """
        Return the list of file extensions this adapter can handle.

        Extensions must be lowercase, including the leading dot (e.g. ".pdf").
        """

    @classmethod
    def health_check(cls) -> bool:
        """
        Return True if the parser backend is installed and reachable.

        Default implementation always returns True; override for adapters
        that depend on external binaries or services.
        """
        return True

    # ── Shared utility ────────────────────────────────────────────────────────

    @staticmethod
    def _sha256_file(path: str | Path) -> str:
        """Compute the SHA-256 hex digest of a file's raw bytes."""
        import hashlib
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def _make_block_id(document_id: str, sequence_index: int) -> str:
        """Generate a stable block ID from the document hash and index."""
        import hashlib
        key = f"{document_id}:{sequence_index}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]
