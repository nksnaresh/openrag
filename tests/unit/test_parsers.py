"""Unit tests for parser adapters.

These tests use real parsing logic on synthetic files created in tmp_path.
No network traffic, no external services.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from openrag.models.content import BlockType
from openrag.parsers.base import ParseOptions
from openrag.parsers.code import PlainCodeAdapter
from openrag.parsers.html import HTMLAdapter


def _opts(tmp_path: Path) -> ParseOptions:
    return ParseOptions(tenant_id="t1", namespace="ns1")


class TestPlainCodeAdapter:
    @pytest.mark.asyncio
    async def test_python_file_extracts_functions(self, tmp_path: Path) -> None:
        code = textwrap.dedent("""\
            def greet(name: str) -> str:
                '''Say hello.'''
                return f"Hello, {name}"

            def farewell(name: str) -> str:
                return f"Goodbye, {name}"
        """)
        p = tmp_path / "greet.py"
        p.write_text(code)
        adapter = PlainCodeAdapter()
        payload = await adapter.parse(p, _opts(tmp_path))
        assert len(payload.blocks) >= 2  # noqa: PLR2004
        assert all(b.block_type == BlockType.CODE for b in payload.blocks)
        names = [b.metadata.get("name") for b in payload.blocks]
        assert "greet" in names
        assert "farewell" in names

    @pytest.mark.asyncio
    async def test_python_file_extracts_classes(self, tmp_path: Path) -> None:
        code = textwrap.dedent("""\
            class Animal:
                '''Base animal class.'''
                def speak(self): pass
        """)
        p = tmp_path / "animals.py"
        p.write_text(code)
        adapter = PlainCodeAdapter()
        payload = await adapter.parse(p, _opts(tmp_path))
        kinds = [b.metadata.get("kind") for b in payload.blocks]
        assert "class" in kinds

    @pytest.mark.asyncio
    async def test_python_invalid_syntax_fallback(self, tmp_path: Path) -> None:
        bad_code = "def foo(:\n    pass"
        p = tmp_path / "bad.py"
        p.write_text(bad_code)
        adapter = PlainCodeAdapter()
        payload = await adapter.parse(p, _opts(tmp_path))
        # Should fall back to single block
        assert len(payload.blocks) == 1
        assert payload.blocks[0].block_type == BlockType.CODE

    @pytest.mark.asyncio
    async def test_document_id_is_sha256(self, tmp_path: Path) -> None:
        import hashlib
        content = b"def foo(): pass"
        p = tmp_path / "sh.py"
        p.write_bytes(content)
        adapter = PlainCodeAdapter()
        payload = await adapter.parse(p, _opts(tmp_path))
        expected = hashlib.sha256(content).hexdigest()
        assert payload.document_id == expected

    @pytest.mark.asyncio
    async def test_supported_extensions(self) -> None:
        adapter = PlainCodeAdapter()
        exts = adapter.supported_extensions()
        assert ".py" in exts
        assert ".js" in exts
        assert ".go" in exts

    @pytest.mark.asyncio
    async def test_empty_file_produces_single_block(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.py"
        p.write_text("")
        adapter = PlainCodeAdapter()
        payload = await adapter.parse(p, _opts(tmp_path))
        assert len(payload.blocks) == 1


class TestHTMLAdapter:
    @pytest.mark.asyncio
    async def test_local_html_extracts_paragraphs(self, tmp_path: Path) -> None:
        html = """<!DOCTYPE html>
        <html><head><title>Test Page</title></head>
        <body>
          <p>This is the first paragraph with enough words to pass the filter.</p>
          <p>This is the second paragraph about testing OpenRAG ingestion pipeline.</p>
        </body>
        </html>"""
        p = tmp_path / "test.html"
        p.write_text(html)
        adapter = HTMLAdapter()
        payload = await adapter.parse(p, _opts(tmp_path))
        assert len(payload.blocks) >= 2  # noqa: PLR2004
        assert all(b.block_type == BlockType.TEXT for b in payload.blocks)

    @pytest.mark.asyncio
    async def test_html_strips_script_and_nav(self, tmp_path: Path) -> None:
        html = """<html><body>
        <nav>Navigation menu go here</nav>
        <p>This is the main article content paragraph with many words.</p>
        <script>var x = 1;</script>
        </body></html>"""
        p = tmp_path / "strip.html"
        p.write_text(html)
        adapter = HTMLAdapter()
        payload = await adapter.parse(p, _opts(tmp_path))
        all_text = " ".join(str(b.raw_content) for b in payload.blocks)
        assert "Navigation menu" not in all_text
        assert "var x = 1" not in all_text
        assert "main article" in all_text

    @pytest.mark.asyncio
    async def test_html_code_block_extracted(self, tmp_path: Path) -> None:
        html = """<html><body>
        <pre><code>def hello(): return "hi"</code></pre>
        </body></html>"""
        p = tmp_path / "code.html"
        p.write_text(html)
        adapter = HTMLAdapter()
        payload = await adapter.parse(p, _opts(tmp_path))
        code_blocks = [b for b in payload.blocks if b.block_type == BlockType.CODE]
        assert len(code_blocks) >= 1

    @pytest.mark.asyncio
    async def test_html_heading_gets_prefix(self, tmp_path: Path) -> None:
        html = """<html><body><h1>Introduction</h1><p>Some content</p></body></html>"""
        p = tmp_path / "heading.html"
        p.write_text(html)
        adapter = HTMLAdapter()
        payload = await adapter.parse(p, _opts(tmp_path))
        headings = [b for b in payload.blocks if str(b.raw_content).startswith("#")]
        assert len(headings) >= 1
        assert "Introduction" in headings[0].raw_content

    @pytest.mark.asyncio
    async def test_html_table_extracted(self, tmp_path: Path) -> None:
        html = """<html><body>
        <table>
          <tr><th>Name</th><th>Score</th></tr>
          <tr><td>Alice</td><td>95</td></tr>
        </table>
        </body></html>"""
        p = tmp_path / "table.html"
        p.write_text(html)
        adapter = HTMLAdapter()
        payload = await adapter.parse(p, _opts(tmp_path))
        tables = [b for b in payload.blocks if b.block_type == BlockType.TABLE]
        assert len(tables) >= 1
        assert "Name" in str(tables[0].raw_content)

    @pytest.mark.asyncio
    async def test_html_health_check(self) -> None:
        assert HTMLAdapter.health_check() is True

    @pytest.mark.asyncio
    async def test_html_supported_extensions(self) -> None:
        exts = HTMLAdapter().supported_extensions()
        assert ".html" in exts
        assert ".htm" in exts
