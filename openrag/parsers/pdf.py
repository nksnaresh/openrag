"""PDF parser adapter backed by PyMuPDF (fitz).

Extracts text spans, embedded images, and tables from PDF files.
Register via:

    AdapterRegistry.register_parser(".pdf", PyMuPDFAdapter)
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from openrag.models.content import (
    BlockType,
    BoundingBox,
    ContentBlock,
    ContentPayload,
    DocumentMeta,
)
from openrag.parsers.base import BaseParserAdapter, ParseOptions


class PyMuPDFAdapter(BaseParserAdapter):
    """Parse PDF files into typed ContentBlocks using PyMuPDF (fitz).

    Extracts:
        - Text spans → TEXT blocks
        - Embedded images → IMAGE blocks (raw bytes)
        - Tables (if fitz.Page.find_tables is available) → TABLE blocks (markdown)
    """

    async def parse(self, file_path: str | Path, options: ParseOptions) -> ContentPayload:
        """Parse a PDF and return a ContentPayload.

        Args:
            file_path: Path to the PDF file.
            options:   Parsing configuration (page range, tenant, etc.)

        Returns:
            ContentPayload with all extracted ContentBlocks.
        """
        import fitz  # PyMuPDF

        path = Path(file_path)
        raw_bytes = path.read_bytes()
        document_id = hashlib.sha256(raw_bytes).hexdigest()

        doc = fitz.open(str(path))

        meta = DocumentMeta(
            title=doc.metadata.get("title") or path.stem,
            author=doc.metadata.get("author"),
            page_count=doc.page_count,
        )

        blocks: list[ContentBlock] = []
        seq = 0

        start_page = options.start_page or 0
        end_page = (options.end_page or doc.page_count - 1) + 1
        end_page = min(end_page, doc.page_count)

        for page_num in range(start_page, end_page):
            page = doc[page_num]

            # ── Text blocks ──────────────────────────────────────────────────
            text_dict = page.get_text("dict")
            page_text_parts: list[str] = []
            for bk in text_dict.get("blocks", []):
                if bk.get("type") == 0:  # text block
                    for line in bk.get("lines", []):
                        for span in line.get("spans", []):
                            text = span.get("text", "").strip()
                            if text:
                                page_text_parts.append(text)

            if page_text_parts:
                full_text = " ".join(page_text_parts)
                rect = page.rect
                blocks.append(ContentBlock(
                    block_id=f"{document_id[:8]}-text-{seq}",
                    block_type=BlockType.TEXT,
                    sequence_index=seq,
                    raw_content=full_text,
                    page_number=page_num,
                    bounding_box=BoundingBox(
                        x0=rect.x0, y0=rect.y0,
                        x1=rect.x1, y1=rect.y1,
                        page=page_num,
                    ),
                ))
                seq += 1

            # ── Image blocks ─────────────────────────────────────────────────
            for img_info in page.get_images(full=True):
                xref = img_info[0]
                img_data = doc.extract_image(xref)
                img_bytes = img_data.get("image", b"")
                if img_bytes:
                    blocks.append(ContentBlock(
                        block_id=f"{document_id[:8]}-img-{seq}",
                        block_type=BlockType.IMAGE,
                        sequence_index=seq,
                        raw_content=img_bytes,
                        page_number=page_num,
                        metadata={"ext": img_data.get("ext", "png")},
                    ))
                    seq += 1

            # ── Table blocks ─────────────────────────────────────────────────
            if hasattr(page, "find_tables"):
                for table in page.find_tables():
                    markdown = self._table_to_markdown(table)
                    if markdown:
                        blocks.append(ContentBlock(
                            block_id=f"{document_id[:8]}-tbl-{seq}",
                            block_type=BlockType.TABLE,
                            sequence_index=seq,
                            raw_content=markdown,
                            page_number=page_num,
                        ))
                        seq += 1

        doc.close()

        return ContentPayload(
            document_id=document_id,
            source_path=str(path),
            tenant_id=options.tenant_id,
            metadata=meta,
            blocks=blocks,
        )

    def supported_extensions(self) -> list[str]:
        return [".pdf"]

    @classmethod
    def health_check(cls) -> bool:
        try:
            import fitz  # noqa: F401
            return True
        except ImportError:
            return False

    @staticmethod
    def _table_to_markdown(table: object) -> str:
        """Convert a fitz table object to a markdown string."""
        try:
            _extract = getattr(table, "extract", None)
            rows = _extract() if callable(_extract) else None
        except Exception:  # noqa: BLE001
            return ""
        if not rows:
            return ""
        lines: list[str] = []
        header = rows[0]
        lines.append("| " + " | ".join(str(c or "") for c in header) + " |")
        lines.append("|" + "|".join("---" for _ in header) + "|")
        for row in rows[1:]:
            lines.append("| " + " | ".join(str(c or "") for c in row) + " |")
        return "\n".join(lines)
