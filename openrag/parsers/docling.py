"""Docling parser adapter for DOCX, PPTX, and structured HTML files.

Uses the `docling` library to parse rich document formats into
ContentPayload with typed ContentBlocks.

Register via:

    AdapterRegistry.register_parser(".docx", DoclingAdapter)
    AdapterRegistry.register_parser(".pptx", DoclingAdapter)
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from openrag.models.content import BlockType, ContentBlock, ContentPayload, DocumentMeta
from openrag.parsers.base import BaseParserAdapter, ParseOptions


class DoclingAdapter(BaseParserAdapter):
    """Parse DOCX/PPTX files using docling.

    Falls back gracefully to plain-text extraction if docling is not
    installed (health_check() will return False and the parser won't be
    registered automatically).
    """

    async def parse(self, file_path: str | Path, options: ParseOptions) -> ContentPayload:
        from docling.document_converter import DocumentConverter

        path = Path(file_path)
        raw_bytes = path.read_bytes()
        document_id = hashlib.sha256(raw_bytes).hexdigest()

        converter = DocumentConverter()
        result = converter.convert(str(path))
        doc = result.document

        meta = DocumentMeta(
            title=getattr(doc, "name", path.stem),
            page_count=None,
        )

        blocks: list[ContentBlock] = []
        seq = 0

        # Export to markdown and split by elements
        md_text = doc.export_to_markdown()
        if not md_text:
            # Fallback: bare text
            md_text = getattr(doc, "text", "") or ""

        for element_text in self._split_markdown(md_text):
            if not element_text.strip():
                continue
            btype = self._classify(element_text)
            blocks.append(ContentBlock(
                document_id=document_id,
                block_id=f"{document_id[:8]}-{seq}",
                block_type=btype,
                sequence_index=seq,
                raw_content=element_text,
            ))
            seq += 1

        return ContentPayload(
            document_id=document_id,
            source_path=str(path),
            tenant_id=options.tenant_id,
            metadata=meta,
            blocks=blocks,
        )

    def supported_extensions(self) -> list[str]:
        return [".docx", ".pptx", ".doc"]

    @classmethod
    def health_check(cls) -> bool:
        try:
            from docling.document_converter import DocumentConverter  # noqa: F401
            return True
        except ImportError:
            return False

    @staticmethod
    def _split_markdown(md: str) -> list[str]:
        """Split markdown into logical elements (paragraphs, headings, code, tables)."""
        chunks: list[str] = []
        current: list[str] = []
        in_code_block = False

        for line in md.splitlines(keepends=True):
            if line.startswith("```"):
                in_code_block = not in_code_block
                current.append(line)
                if not in_code_block:
                    chunks.append("".join(current))
                    current = []
            elif in_code_block:
                current.append(line)
            elif line.strip() == "" and current:
                chunks.append("".join(current))
                current = []
            else:
                current.append(line)

        if current:
            chunks.append("".join(current))
        return chunks

    @staticmethod
    def _classify(text: str) -> BlockType:
        """Classify a markdown chunk into a BlockType."""
        stripped = text.strip()
        if stripped.startswith("#"):
            return BlockType.TEXT
        if stripped.startswith("```"):
            return BlockType.CODE
        if re.match(r"^\|.*\|", stripped):
            return BlockType.TABLE
        if re.match(r"^\$\$", stripped) or stripped.startswith("\\["):
            return BlockType.EQUATION
        return BlockType.TEXT
