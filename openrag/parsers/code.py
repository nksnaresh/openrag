"""Plain-code parser adapter.

Parses source code files into CODE ContentBlocks. Uses Python's stdlib
``ast`` module for .py files and regex-based extraction for other languages.

Register via:

    for ext in [".py", ".js", ".ts", ".go", ".java", ".rs", ".cpp", ".c"]:
        AdapterRegistry.register_parser(ext, PlainCodeAdapter)
"""

from __future__ import annotations

import ast
import hashlib
import re
from pathlib import Path

from openrag.models.content import BlockType, ContentBlock, ContentPayload, DocumentMeta
from openrag.parsers.base import BaseParserAdapter, ParseOptions

# Language detection by file extension
_EXT_TO_LANG: dict[str, str] = {
    ".py": "python", ".js": "javascript", ".ts": "typescript",
    ".go": "go", ".java": "java", ".rs": "rust",
    ".cpp": "cpp", ".cc": "cpp", ".c": "c",
    ".rb": "ruby", ".php": "php",
}

# Regex patterns for non-Python function/class extraction
_FUNCTION_RE = re.compile(
    r"(?:^|\n)"                          # start of line
    r"(?:pub\s+)?(?:async\s+)?"          # optional pub/async (Rust/JS)
    r"(?:func|function|def|fn|void|int|string|bool|auto)\s+"
    r"(\w+)\s*\(",
    re.MULTILINE,
)
_CLASS_RE = re.compile(
    r"(?:^|\n)(?:pub\s+)?(?:class|struct|interface|trait|type)\s+(\w+)",
    re.MULTILINE,
)


class PlainCodeAdapter(BaseParserAdapter):
    """Parse source code files into CODE ContentBlocks.

    For Python files, uses the stdlib ``ast`` module to extract individual
    function and class definitions. For all other languages, falls back to
    regex-based extraction. If no functions/classes are found, the entire
    file becomes a single CODE block.
    """

    async def parse(self, file_path: str | Path, options: ParseOptions) -> ContentPayload:
        path = Path(file_path)
        raw = path.read_bytes()
        document_id = hashlib.sha256(raw).hexdigest()
        source_code = raw.decode("utf-8", errors="replace")
        lang = _EXT_TO_LANG.get(path.suffix.lower(), "unknown")

        meta = DocumentMeta(title=path.name)
        blocks: list[ContentBlock] = []

        if lang == "python":
            blocks = self._parse_python(source_code, document_id, str(path))
        else:
            blocks = self._parse_generic(source_code, document_id, lang, str(path))

        if not blocks:
            # Fallback: whole file as one block
            blocks = [ContentBlock(
                block_id=f"{document_id[:8]}-code-0",
                document_id=document_id,
                block_type=BlockType.CODE,
                sequence_index=0,
                raw_content=source_code,
                metadata={"language": lang, "file_path": str(path)},
            )]

        return ContentPayload(
            document_id=document_id,
            source_path=str(path),
            tenant_id=options.tenant_id,
            metadata=meta,
            blocks=blocks,
        )

    def supported_extensions(self) -> list[str]:
        return list(_EXT_TO_LANG.keys())

    # ── Python AST extraction ──────────────────────────────────────────────────

    @staticmethod
    def _parse_python(
        source: str, document_id: str, file_path: str
    ) -> list[ContentBlock]:
        """Extract functions and classes using ast.parse."""
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return []

        blocks: list[ContentBlock] = []
        lines = source.splitlines()
        seq = 0

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            start = node.lineno - 1
            end = node.end_lineno or start + 1
            snippet = "\n".join(lines[start:end])
            docstring = ast.get_docstring(node) or ""
            kind = "class" if isinstance(node, ast.ClassDef) else "function"
            blocks.append(ContentBlock(
                block_id=f"{document_id[:8]}-code-{seq}",
                document_id=document_id,
                block_type=BlockType.CODE,
                sequence_index=seq,
                raw_content=snippet,
                metadata={
                    "language": "python",
                    "name": node.name,
                    "kind": kind,
                    "docstring": docstring,
                    "line_start": node.lineno,
                    "line_end": node.end_lineno,
                    "file_path": file_path,
                },
            ))
            seq += 1

        return blocks

    @staticmethod
    def _parse_generic(
        source: str, document_id: str, lang: str, file_path: str
    ) -> list[ContentBlock]:
        """Extract functions and classes via regex for non-Python files."""
        spans: list[tuple[int, str, str]] = []  # (pos, kind, name)

        for m in _FUNCTION_RE.finditer(source):
            spans.append((m.start(), "function", m.group(1)))
        for m in _CLASS_RE.finditer(source):
            spans.append((m.start(), "class", m.group(1)))

        spans.sort(key=lambda x: x[0])
        if not spans:
            return []

        blocks: list[ContentBlock] = []
        lines = source.splitlines()

        for seq, (pos, kind, name) in enumerate(spans):
            line_no = source[:pos].count("\n")
            # Take up to 60 lines as a reasonable snippet
            snippet = "\n".join(lines[line_no: line_no + 60])
            blocks.append(ContentBlock(
                block_id=f"{document_id[:8]}-code-{seq}",
                document_id=document_id,
                block_type=BlockType.CODE,
                sequence_index=seq,
                raw_content=snippet,
                metadata={
                    "language": lang,
                    "name": name,
                    "kind": kind,
                    "file_path": file_path,
                },
            ))

        return blocks
