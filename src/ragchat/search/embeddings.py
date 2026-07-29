"""Embedding module supporting Local BAAI/bge-small-en-v1.5 and Production FastEmbed 384d Embeddings.
"""

from typing import List
import structlog
from langchain_core.embeddings import Embeddings

from ragchat.config import settings

logger = structlog.get_logger(__name__)

_LOCAL_MODEL = None
_FASTEMBED_MODEL = None


def get_embedding_model():
    """Return local SentenceTransformer model singleton."""
    global _LOCAL_MODEL
    if _LOCAL_MODEL is None:
        from sentence_transformers import SentenceTransformer
        logger.info("loading_local_bge_embedding_model", model=settings.embedding_model)
        _LOCAL_MODEL = SentenceTransformer(settings.embedding_model)
        logger.info("local_bge_embedding_model_loaded", model=settings.embedding_model)
    return _LOCAL_MODEL


def get_fastembed_model():
    """Return lightweight 384d FastEmbed model instance for Qdrant compatibility."""
    global _FASTEMBED_MODEL
    if _FASTEMBED_MODEL is None:
        from fastembed import TextEmbedding
        logger.info("loading_fastembed_384d_model")
        _FASTEMBED_MODEL = TextEmbedding("BAAI/bge-small-en-v1.5")
    return _FASTEMBED_MODEL


class LocalEmbeddings(Embeddings):
    """Router for local BAAI embeddings vs Production 384d FastEmbed API embeddings."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        # Production Cloud Routing (384d FastEmbed matching Qdrant)
        if getattr(settings, "app_env", "dev").lower() in ("prod", "production"):
            try:
                model = get_fastembed_model()
                return [list(vec) for vec in model.embed(texts)]
            except Exception as exc:
                logger.warning("fastembed_failed_using_local_fallback", error=str(exc))

        # Local Execution (Default BAAI/bge-small-en-v1.5)
        model = get_embedding_model()
        embeddings = model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        from ragchat.cache.rag_cache import rag_cache

        cached_vector = rag_cache.get_embedding(text)
        if cached_vector is not None:
            return cached_vector

        # Production Cloud Routing (384d FastEmbed matching Qdrant)
        if getattr(settings, "app_env", "dev").lower() in ("prod", "production"):
            try:
                model = get_fastembed_model()
                embedding = list(next(model.embed([text])))
                rag_cache.set_embedding(text, embedding)
                return embedding
            except Exception as exc:
                logger.warning("fastembed_failed_using_local_fallback", error=str(exc))

        # Local Execution (Default BAAI/bge-small-en-v1.5)
        model = get_embedding_model()
        embedding = model.encode(text, convert_to_numpy=True).tolist()
        rag_cache.set_embedding(text, embedding)
        return embedding
