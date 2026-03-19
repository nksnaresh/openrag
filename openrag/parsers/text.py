"""Plain-text parser adapter."""

from __future__ import annotations

from pathlib import Path

from openrag.models.content import BlockType, ContentBlock, ContentPayload, DocumentMeta
from openrag.parsers.base import BaseParserAdapter, ParseOptions


class PlainTextAdapter(BaseParserAdapter):
    """Adapter for reading raw .txt files."""

    async def parse(self, file_path: str | Path, options: ParseOptions) -> ContentPayload:
        path = Path(file_path)
        document_id = self._sha256_file(path)
        raw_text = path.read_text(encoding="utf-8", errors="replace")

        meta = DocumentMeta(title=path.name)
        
        # Simple chunking: one block for the whole file for now
        blocks = [ContentBlock(
            block_id=self._make_block_id(document_id, 0),
            document_id=document_id,
            block_type=BlockType.TEXT,
            sequence_index=0,
            raw_content=raw_text,
            metadata={"file_path": str(path)},
        )]

        return ContentPayload(
            document_id=document_id,
            source_path=str(path.absolute()),
            tenant_id=options.tenant_id,
            metadata=meta,
            blocks=blocks,
        )

    def supported_extensions(self) -> list[str]:
        return [".txt", ".md", ".rst", ".log"]
