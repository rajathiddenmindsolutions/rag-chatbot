"""Custom structural chunking strategy using Docling's document structure."""

import pandas as pd
from docling_core.types.doc import DoclingDocument
from ragchat.chunking.base import Chunker, ChunkResult


class StructuralChunker(Chunker):
    """Chunks documents using heading boundaries and structural blocks (tables, sections)."""

    def __init__(self, target_chunk_size: int = 1200):
        self.target_chunk_size = target_chunk_size

    def chunk(self, docling_document: DoclingDocument, markdown_text: str) -> list[ChunkResult]:
        """Iterates through Docling document items to split structurally."""
        chunks = []
        
        # Maintain active headings at each level
        active_headings: dict[int, str] = {}
        
        # Accumulate paragraph text for the current section
        current_text_buffer = []
        current_buffer_length = 0

        def get_section_path() -> str:
            sorted_levels = sorted(active_headings.keys())
            path_parts = [active_headings[lvl] for lvl in sorted_levels if active_headings[lvl]]
            return " > ".join(path_parts) if path_parts else "root"

        def flush_buffer():
            nonlocal current_text_buffer, current_buffer_length
            if current_text_buffer:
                combined_text = "\n".join(current_text_buffer).strip()
                if combined_text:
                    token_count = len(combined_text.split())
                    chunks.append(ChunkResult(
                        text=combined_text,
                        section_path=get_section_path(),
                        token_count=token_count,
                    ))
                current_text_buffer = []
                current_buffer_length = 0

        # Check if the docling_document is valid and has iterate_items
        if not docling_document or not hasattr(docling_document, "iterate_items"):
            # Fallback: if document structure is not present, return empty or fallback
            return []

        for item, level in docling_document.iterate_items():
            label = getattr(item, "label", "").lower()
            text = getattr(item, "text", "").strip()

            if label == "heading":
                # Flush existing buffer before changing heading path
                flush_buffer()
                # Clear all headings at the same or deeper levels
                levels_to_clear = [l for l in active_headings.keys() if l >= level]
                for l in levels_to_clear:
                    active_headings.pop(l, None)
                active_headings[level] = text

            elif label == "table":
                # Flush text buffer first so table is separate
                flush_buffer()
                
                # Try to serialize table to markdown
                table_md = ""
                try:
                    # In docling v2, table has export_to_dataframe
                    if hasattr(item, "export_to_dataframe"):
                        df = item.export_to_dataframe(doc=docling_document)
                        table_md = df.to_markdown()
                    else:
                        table_md = text
                except Exception:
                    table_md = text if text else "[Table]"

                if table_md:
                    token_count = len(table_md.split())
                    chunks.append(ChunkResult(
                        text=table_md,
                        section_path=get_section_path(),
                        token_count=token_count,
                    ))

            elif label in ("paragraph", "list_item", "caption", "text"):
                if not text:
                    continue
                current_text_buffer.append(text)
                current_buffer_length += len(text)
                
                if current_buffer_length >= self.target_chunk_size:
                    flush_buffer()
            else:
                # Other items (e.g. check for generic text)
                if text and len(text) > 20:
                    current_text_buffer.append(text)
                    current_buffer_length += len(text)
                    if current_buffer_length >= self.target_chunk_size:
                        flush_buffer()

        # Flush any remaining text in the buffer
        flush_buffer()

        return chunks
