"""Semantic similarity-based chunking strategy using sentence embeddings."""

import re
import numpy as np
from docling_core.types.doc import DoclingDocument

from ragchat.chunking.base import Chunker, ChunkResult
from ragchat.config import settings

import structlog
logger = structlog.get_logger(__name__)


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
        self._model = None

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.embedding_model_name)
            except Exception as exc:
                logger.warning("sentence_transformers_not_available_for_semantic_chunker", error=str(exc))
                self._model = "lightweight"
        return self._model

    def _split_into_sentences(self, text: str) -> list[str]:
        sentence_end = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s')
        sentences = sentence_end.split(text)
        return [s.strip() for s in sentences if s.strip()]

    def chunk(self, docling_document: DoclingDocument, markdown_text: str) -> list[ChunkResult]:
        if not markdown_text.strip():
            return []

        sentences = self._split_into_sentences(markdown_text)
        if len(sentences) <= 1:
            return [ChunkResult(text=markdown_text, section_path="root", token_count=len(markdown_text.split()))]

        model = self.model
        if hasattr(model, "encode"):
            embeddings = model.encode(sentences, convert_to_numpy=True)
            similarities = []
            for i in range(len(embeddings) - 1):
                vec1 = embeddings[i]
                vec2 = embeddings[i + 1]
                norm1 = np.linalg.norm(vec1)
                norm2 = np.linalg.norm(vec2)
                sim = float(np.dot(vec1, vec2) / (norm1 * norm2)) if (norm1 > 0 and norm2 > 0) else 0.0
                similarities.append(sim)

            threshold = float(np.percentile(similarities, self.similarity_threshold_percentile)) if similarities else 0.0

            chunks = []
            current_sentences = [sentences[0]]
            for i, sim in enumerate(similarities):
                if sim < threshold or len(" ".join(current_sentences)) > self.max_chunk_size:
                    chunk_text = " ".join(current_sentences)
                    chunks.append(ChunkResult(text=chunk_text, section_path="root", token_count=len(chunk_text.split())))
                    current_sentences = [sentences[i + 1]]
                else:
                    current_sentences.append(sentences[i + 1])

            if current_sentences:
                chunk_text = " ".join(current_sentences)
                chunks.append(ChunkResult(text=chunk_text, section_path="root", token_count=len(chunk_text.split())))

            return chunks

        # Fallback simple sentence window chunker
        chunks = []
        step = 5
        for i in range(0, len(sentences), step):
            chunk_text = " ".join(sentences[i:i+step])
            chunks.append(ChunkResult(text=chunk_text, section_path="root", token_count=len(chunk_text.split())))
        return chunks
