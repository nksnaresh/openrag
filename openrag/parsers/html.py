"""HTML / URL parser adapter.

Supports:
    - Local HTML files (.html, .htm)
    - Remote URLs (http:// or https://)

Strips navigation, footer, and script elements, extracting article-body
text, headings, code blocks, and tables.

Register via:

    AdapterRegistry.register_parser(".html", HTMLAdapter)
    AdapterRegistry.register_parser(".htm",  HTMLAdapter)
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from openrag.models.content import (
    BlockType,
    ContentBlock,
    ContentPayload,
    DocumentMeta,
)
from openrag.parsers.base import BaseParserAdapter, ParseOptions

_STRIP_TAGS = {"nav", "footer", "header", "script", "style", "aside", "noscript"}


class HTMLAdapter(BaseParserAdapter):
    """Parse HTML files or URLs into typed ContentBlocks with BeautifulSoup."""

    async def parse(self, file_path: str | Path, options: ParseOptions) -> ContentPayload:
        """Parse an HTML file or URL.

        If ``file_path`` starts with ``http://`` or ``https://``, the page
        is fetched asynchronously via httpx. Otherwise the file is read
        from disk.
        """
        from bs4 import BeautifulSoup

        src = str(file_path)
        if src.startswith(("http://", "https://")):
            html = await self._fetch_url(src)
            document_id = hashlib.sha256(html.encode()).hexdigest()
            source_path = src
        else:
            path = Path(file_path)
            raw = path.read_bytes()
            document_id = hashlib.sha256(raw).hexdigest()
            html = raw.decode("utf-8", errors="replace")
            source_path = str(path)

        soup = BeautifulSoup(html, "lxml")

        # Strip unwanted elements
        for tag in soup.find_all(_STRIP_TAGS):
            tag.decompose()

        title_tag = soup.find("title")
        page_title = title_tag.get_text(strip=True) if title_tag else Path(source_path).stem

        meta = DocumentMeta(title=page_title, source_url=source_path)
        blocks: list[ContentBlock] = []
        seq = 0

        body = soup.body
        for element in body.descendants if body is not None else []:
            tag_name = getattr(element, "name", None)
            if tag_name is None:
                continue  # NavigableString

            if tag_name in {"p", "li", "dd", "dt", "blockquote", "article"}:
                text = element.get_text(separator=" ", strip=True)
                if len(text) > 20:
                    blocks.append(ContentBlock(
                        document_id=document_id,
                        block_id=f"{document_id[:8]}-text-{seq}",
                        block_type=BlockType.TEXT,
                        sequence_index=seq,
                        raw_content=text,
                    ))
                    seq += 1

            elif re.match(r"h[1-6]", tag_name):
                text = "# " + element.get_text(strip=True)
                if text.strip():
                    blocks.append(ContentBlock(
                        document_id=document_id,
                        block_id=f"{document_id[:8]}-head-{seq}",
                        block_type=BlockType.TEXT,
                        sequence_index=seq,
                        raw_content=text,
                        metadata={"heading_level": tag_name},
                    ))
                    seq += 1

            elif tag_name in {"pre", "code"}:
                code = element.get_text()
                if len(code.strip()) > 10:
                    blocks.append(ContentBlock(
                        document_id=document_id,
                        block_id=f"{document_id[:8]}-code-{seq}",
                        block_type=BlockType.CODE,
                        sequence_index=seq,
                        raw_content=code,
                    ))
                    seq += 1

            elif tag_name == "table":
                md = self._table_to_markdown(element)
                if md:
                    blocks.append(ContentBlock(
                        document_id=document_id,
                        block_id=f"{document_id[:8]}-tbl-{seq}",
                        block_type=BlockType.TABLE,
                        sequence_index=seq,
                        raw_content=md,
                    ))
                    seq += 1

        return ContentPayload(
            document_id=document_id,
            source_path=source_path,
            tenant_id=options.tenant_id,
            metadata=meta,
            blocks=blocks,
        )

    def supported_extensions(self) -> list[str]:
        return [".html", ".htm"]

    @classmethod
    def health_check(cls) -> bool:
        try:
            from bs4 import BeautifulSoup  # noqa: F401
            return True
        except ImportError:
            return False

    @staticmethod
    async def _fetch_url(url: str) -> str:
        """Fetch HTML from a URL using httpx."""
        import httpx
        async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
            resp = await client.get(url, headers={"User-Agent": "OpenRAG/0.1"})
            resp.raise_for_status()
            return resp.text

    @staticmethod
    def _table_to_markdown(table_tag: object) -> str:
        """Convert a BS4 table element to markdown."""
        from bs4 import Tag
        if not isinstance(table_tag, Tag):
            return ""
        rows = table_tag.find_all("tr")
        if not rows:
            return ""
        lines: list[str] = []
        for i, row in enumerate(rows):
            cells = [c.get_text(strip=True) for c in row.find_all(["th", "td"])]
            lines.append("| " + " | ".join(cells) + " |")
            if i == 0:
                lines.append("|" + "|".join("---" for _ in cells) + "|")
        return "\n".join(lines)
