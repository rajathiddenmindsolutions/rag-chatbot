"""Markdown and recursive character boundary-aware chunking strategy."""

from docling_core.types.doc import DoclingDocument
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from ragchat.chunking.base import Chunker, ChunkResult


class RecursiveMarkdownChunker(Chunker):
    """Chunks markdown by header hierarchy and then subdivides large blocks recursively."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Split on standard Markdown headers
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
        ]
        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on,
            strip_headers=False,
        )
        
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )

    def chunk(self, docling_document: DoclingDocument, markdown_text: str) -> list[ChunkResult]:
        """Split text by headers and sub-split large sections recursively."""
        if not markdown_text.strip():
            return []

        # Step 1: Split by markdown headers
        header_documents = self.header_splitter.split_text(markdown_text)
        
        chunks = []
        for doc in header_documents:
            # Construct a section path from metadata
            headers = []
            for h_key in ["Header 1", "Header 2", "Header 3", "Header 4"]:
                if h_key in doc.metadata:
                    headers.append(doc.metadata[h_key])
            section_path = " > ".join(headers) if headers else "root"
            
            # Step 2: If the text is larger than chunk_size, split recursively
            if len(doc.page_content) > self.chunk_size:
                sub_texts = self.recursive_splitter.split_text(doc.page_content)
                for sub_text in sub_texts:
                    token_count = len(sub_text.split())  # simple word count estimate
                    chunks.append(ChunkResult(
                        text=sub_text.strip(),
                        section_path=section_path,
                        token_count=token_count,
                    ))
            else:
                token_count = len(doc.page_content.split())
                chunks.append(ChunkResult(
                    text=doc.page_content.strip(),
                    section_path=section_path,
                    token_count=token_count,
                ))
                
        return chunks
