"""SentenceTransformers local embedding wrapper conforming to LangChain Embeddings.

Key design: The SentenceTransformer model is loaded ONCE at module level as a
process-wide singleton. This avoids the 6-7 second reload penalty that occurred
when LocalEmbeddings() was instantiated per-request inside retrieve_node.
"""

from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

from ragchat.config import settings

import structlog

logger = structlog.get_logger(__name__)

# ─── Process-wide singleton ───────────────────────────────────────────────────
# Loaded once when the module is first imported (at server startup).
# All subsequent calls to LocalEmbeddings.embed_query / embed_documents reuse
# this single in-memory model — eliminating the per-request HuggingFace reload.
_EMBEDDING_MODEL: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    """Return the shared SentenceTransformer model, loading it once if needed."""
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        logger.info("loading_embedding_model", model=settings.embedding_model)
        _EMBEDDING_MODEL = SentenceTransformer(settings.embedding_model)
        logger.info("embedding_model_loaded", model=settings.embedding_model)
    return _EMBEDDING_MODEL


# ─────────────────────────────────────────────────────────────────────────────


class LocalEmbeddings(Embeddings):
    """Wrapper around the shared SentenceTransformer singleton with Layer 2 caching."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of document strings."""
        if not texts:
            return []
        model = get_embedding_model()
        embeddings = model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string with Layer 2 caching."""
        from ragchat.cache.rag_cache import rag_cache

        cached_vector = rag_cache.get_embedding(text)
        if cached_vector is not None:
            return cached_vector

        model = get_embedding_model()
        embedding = model.encode(text, convert_to_numpy=True).tolist()
        rag_cache.set_embedding(text, embedding)
        return embedding
