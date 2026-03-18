"""BM25 indexer that tokenizes content and stores it in the Document Store.

This provides the 'sparse' component of the hybrid search system. 
Actual scoring is typically done at retrieval time using rank-bm25 or a 
full-text search engine (Elasticsearch/Solr).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openrag.models.processing import ProcessedBlock
    from openrag.storage.base import BaseDocumentAdapter


class BM25Indexer:
    """Tokenizes processed blocks and persists tokens to the document store.

    Args:
        doc_store: The document store adapter where tokens will be persisted.
    """

    def __init__(self, doc_store: BaseDocumentAdapter) -> None:
        self._doc_store = doc_store
        # Basic alphanumeric tokenizer
        self._tokenizer_regex = re.compile(r"[\w']+")

    def tokenize(self, text: str) -> list[str]:
        """Simple regex-based tokenizer that lowercases and splits.

        In a production setting, this would include stemmers (Porter/Snowball)
        and more sophisticated stop-word filtering.
        """
        if not text:
            return []
        return self._tokenizer_regex.findall(text.lower())

    async def index_blocks(
        self,
        blocks: list[ProcessedBlock],
        namespace: str,
    ) -> None:
        """Tokenize each block's embedding_text and save to the store."""
        if not blocks:
            return

        for block in blocks:
            text = block.embedding_text
            if not text:
                # Fallback to description if embedding_text is empty
                text = block.natural_language_description or ""
            
            if not text:
                continue

            tokens = self.tokenize(text)
            if tokens:
                await self._doc_store.save_bm25_tokens(
                    namespace=namespace,
                    chunk_id=block.source_block.block_id,
                    tokens=tokens,
                )
