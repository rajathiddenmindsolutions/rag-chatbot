"""Fixed-size token window chunking strategy."""

from docling_core.types.doc import DoclingDocument
from langchain_text_splitters import TokenTextSplitter

from ragchat.chunking.base import Chunker, ChunkResult


class FixedSizeChunker(Chunker):
    """Chunks text into fixed size token windows using tiktoken."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Default encoding is gpt2/gpt-4 compatible
        self.splitter = TokenTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

    def chunk(self, docling_document: DoclingDocument, markdown_text: str) -> list[ChunkResult]:
        """Split document text into fixed size chunks."""
        # We split the raw markdown text directly
        raw_chunks = self.splitter.split_text(markdown_text)
        
        chunks = []
        for text in raw_chunks:
            # We estimate token count based on typical 4 chars/token or standard split
            # TokenTextSplitter splits by tokens, so we can calculate it exactly by split size
            token_count = len(text.split())  # raw fallback, but we can set chunk_size approx
            chunks.append(ChunkResult(
                text=text.strip(),
                section_path="root",
                token_count=token_count,
            ))
        return chunks
