"""Base chunker protocol and interface definition."""

from typing import Protocol, runtime_checkable
from docling_core.types.doc import DoclingDocument


class ChunkResult:
    def __init__(self, text: str, section_path: str = "", token_count: int = 0):
        self.text = text
        self.section_path = section_path
        self.token_count = token_count


@runtime_checkable
class Chunker(Protocol):
    """Protocol for chunking strategies."""

    def chunk(self, docling_document: DoclingDocument, markdown_text: str) -> list[ChunkResult]:
        """Split a Docling parsed document into structured ChunkResult objects."""
        ...
