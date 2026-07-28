"""Semantic similarity-based chunking strategy using sentence embeddings."""

import re
import numpy as np
from docling_core.types.doc import DoclingDocument
from sentence_transformers import SentenceTransformer

from ragchat.chunking.base import Chunker, ChunkResult
from ragchat.config import settings


class SemanticChunker(Chunker):
    """Splits text into chunks at sentence boundaries when semantic similarity drops."""

    def __init__(
        self,
        embedding_model_name: str = settings.embedding_model,
        similarity_threshold_percentile: float = 40.0,
        max_chunk_size: int = 1500,
    ):
        self.embedding_model_name = embedding_model_name
        self.similarity_threshold_percentile = similarity_threshold_percentile
        self.max_chunk_size = max_chunk_size
        
        # Load the model lazily
        self._model = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.embedding_model_name)
        return self._model

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences using simple regex parsing."""
        # Split on sentence boundaries (., !, ?) followed by whitespace
        sentence_end = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s')
        sentences = sentence_end.split(text)
        return [s.strip() for s in sentences if s.strip()]

    def chunk(self, docling_document: DoclingDocument, markdown_text: str) -> list[ChunkResult]:
        """Split markdown text based on semantic similarity of sentences."""
        if not markdown_text.strip():
            return []

        sentences = self._split_into_sentences(markdown_text)
        if len(sentences) <= 1:
            return [ChunkResult(text=markdown_text, section_path="root", token_count=len(markdown_text.split()))]

        # 1. Compute embeddings for all sentences
        embeddings = self.model.encode(sentences, convert_to_numpy=True)

        # 2. Compute cosine similarity between consecutive sentences
        similarities = []
        for i in range(len(embeddings) - 1):
            vec1 = embeddings[i]
            vec2 = embeddings[i + 1]
            
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 > 0 and norm2 > 0:
                sim = np.dot(vec1, vec2) / (norm1 * norm2)
            else:
                sim = 0.0
            similarities.append(sim)

        # 3. Determine threshold below which we split
        if similarities:
            # We want to split where similarity is LOW, i.e. drops below a percentile threshold.
            threshold = np.percentile(similarities, self.similarity_threshold_percentile)
        else:
            threshold = 0.5

        # 4. Build chunks
        chunks = []
        current_sentences = [sentences[0]]
        current_len = len(sentences[0])

        for i, sim in enumerate(similarities):
            next_sentence = sentences[i + 1]
            
            # Split if similarity is below threshold OR we exceed max_chunk_size
            should_split = (sim < threshold) or (current_len + len(next_sentence) > self.max_chunk_size)
            
            if should_split and current_sentences:
                chunk_text = " ".join(current_sentences)
                chunks.append(ChunkResult(
                    text=chunk_text,
                    section_path="root",
                    token_count=len(chunk_text.split()),
                ))
                current_sentences = [next_sentence]
                current_len = len(next_sentence)
            else:
                current_sentences.append(next_sentence)
                current_len += len(next_sentence) + 1  # count the space

        if current_sentences:
            chunk_text = " ".join(current_sentences)
            chunks.append(ChunkResult(
                text=chunk_text,
                section_path="root",
                token_count=len(chunk_text.split()),
            ))

        return chunks
