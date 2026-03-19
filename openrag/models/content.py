"""Content block data contracts — core document representation types."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class BlockType(str, Enum):
    """Supported content modality types."""

    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    EQUATION = "equation"
    CODE = "code"
    AUDIO_TRANSCRIPT = "audio_transcript"
    VIDEO_FRAME = "video_frame"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class BoundingBox:
    """Pixel/point coordinates of a content block on a page."""

    x0: float
    y0: float
    x1: float
    y1: float
    page: int

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass
class ContentBlock:
    """
    A single typed content unit extracted from a document.

    This is the normalized output format for all parser adapters.
    Every piece of content — a paragraph, an image, a table, a code
    snippet, an equation — becomes one ContentBlock.
    """

    block_id: str
    """Stable unique identifier: sha256(document_id + sequence_index)."""

    document_id: str
    """SHA-256 hex digest of the source document."""

    block_type: BlockType
    """Modality of this content block."""

    sequence_index: int
    """0-based order within the document (for context windowing)."""

    raw_content: Any
    """
    Type-specific payload:
    - TEXT:             str
    - IMAGE:            bytes (raw image bytes)
    - TABLE:            str  (Markdown or HTML table)
    - EQUATION:         str  (LaTeX or MathML string)
    - CODE:             str  (source code string)
    - AUDIO_TRANSCRIPT: list[dict] with keys "start", "end", "text"
    - VIDEO_FRAME:      bytes (JPEG/PNG frame bytes)
    """

    page_number: int | None = None
    """1-based page number; None for non-paged sources."""

    bounding_box: BoundingBox | None = None
    """Spatial location of the block on the page, if available."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """
    Modality-specific extra fields, e.g.:
    - caption, alt_text, language, footnote, mime_type,
      table_format, equation_format, function_name, timestamp_start
    """


@dataclass
class DocumentMeta:
    """Rich metadata describing the source document."""

    title: str | None = None
    author: str | None = None
    language: str = "en"
    page_count: int | None = None
    word_count: int | None = None
    source_url: str | None = None
    created_at: str | None = None    # ISO-8601
    modified_at: str | None = None   # ISO-8601
    file_size_bytes: int | None = None
    mime_type: str | None = None
    custom: dict[str, Any] = field(default_factory=dict)


@dataclass
class ContentPayload:
    """
    Normalized output of a parser adapter run.

    One ContentPayload is produced per ingested document and carries
    all extracted ContentBlocks plus document-level metadata.
    """

    document_id: str
    """SHA-256 hex digest of the raw file bytes. Used for deduplication."""

    source_path: str
    """Absolute file path or URL of the original document."""

    tenant_id: str
    """Namespace/tenant this document belongs to."""

    metadata: DocumentMeta
    """Document-level metadata."""

    blocks: list[ContentBlock] = field(default_factory=list)
    """Ordered list of content blocks, by sequence_index."""

    acl: dict[str, list[str]] = field(default_factory=lambda: {"read": [], "write": []})
    """Access control policy: {"read": ["role:analysts"], "write": ["role:admins"]}"""

    def blocks_of_type(self, block_type: BlockType) -> list[ContentBlock]:
        """Return all blocks of a given modality."""
        return [b for b in self.blocks if b.block_type == block_type]

    def __len__(self) -> int:
        return len(self.blocks)
