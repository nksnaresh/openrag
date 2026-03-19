"""Image parser adapter — handles standalone image files (.jpg, .png, etc.)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from openrag.models.content import BlockType, ContentBlock, ContentPayload, DocumentMeta
from openrag.parsers.base import BaseParserAdapter

if TYPE_CHECKING:
    from openrag.parsers.base import ParseOptions


class ImageParserAdapter(BaseParserAdapter):
    """
    Parser for standalone image files.
    
    Wraps the entire image file into a single ContentBlock of type IMAGE.
    """

    async def parse(self, file_path: str | Path, options: ParseOptions) -> ContentPayload:
        path = Path(file_path)
        with path.open("rb") as f:
            image_bytes = f.read()

        # Compute document ID (hash of bytes)
        import hashlib
        doc_id = hashlib.sha256(image_bytes).hexdigest()

        # Create the IMAGE block
        block = ContentBlock(
            block_id=f"{doc_id}_0",
            document_id=doc_id,
            block_type=BlockType.IMAGE,
            sequence_index=0,
            raw_content=image_bytes,
            metadata={
                "file_name": path.name,
                "extension": path.suffix.lower(),
                "mime_type": f"image/{path.suffix.lower().lstrip('.')}"
            }
        )

        return ContentPayload(
            document_id=doc_id,
            source_path=str(path.absolute()),
            tenant_id=options.tenant_id,
            metadata=DocumentMeta(
                title=path.name,
                mime_type=f"image/{path.suffix.lower().lstrip('.')}",
                file_size_bytes=len(image_bytes)
            ),
            blocks=[block]
        )

    def supported_extensions(self) -> list[str]:
        return [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"]
