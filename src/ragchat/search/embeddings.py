"""Embedding module supporting Local BAAI/bge-small-en-v1.5 and Production Google Gemini Embedding 2 (512d).
"""

from typing import List
import structlog
from langchain_core.embeddings import Embeddings

from ragchat.config import settings

logger = structlog.get_logger(__name__)

_LOCAL_MODEL = None


def get_embedding_model():
    """Return local SentenceTransformer model singleton."""
    global _LOCAL_MODEL
    if _LOCAL_MODEL is None:
        from sentence_transformers import SentenceTransformer
        logger.info("loading_local_bge_embedding_model", model=settings.embedding_model)
        _LOCAL_MODEL = SentenceTransformer(settings.embedding_model)
        logger.info("local_bge_embedding_model_loaded", model=settings.embedding_model)
    return _LOCAL_MODEL


class LocalEmbeddings(Embeddings):
    """Router for local BAAI embeddings vs Production Google Gemini Embedding 2 (512d)."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        # Production Cloud Routing (Google Gemini Embedding 2 with 512d matching Qdrant)
        if getattr(settings, "app_env", "dev").lower() in ("prod", "production"):
            try:
                from langchain_google_genai import GoogleGenerativeAIEmbeddings
                api_key = settings.google_api_key or getattr(settings, "gemni_api_key", None)
                gemini_embed = GoogleGenerativeAIEmbeddings(
                    model="gemini-embedding-2",
                    google_api_key=api_key,
                    output_dimensionality=512,
                )
                return gemini_embed.embed_documents(texts)
            except Exception as exc:
                logger.warning("gemini_embedding_failed_using_local_fallback", error=str(exc))

        # Local Execution (Default BAAI/bge-small-en-v1.5)
        model = get_embedding_model()
        embeddings = model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        from ragchat.cache.rag_cache import rag_cache

        cached_vector = rag_cache.get_embedding(text)
        if cached_vector is not None:
            return cached_vector

        # Production Cloud Routing (Google Gemini Embedding 2 with 512d matching Qdrant)
        if getattr(settings, "app_env", "dev").lower() in ("prod", "production"):
            try:
                from langchain_google_genai import GoogleGenerativeAIEmbeddings
                api_key = settings.google_api_key or getattr(settings, "gemni_api_key", None)
                gemini_embed = GoogleGenerativeAIEmbeddings(
                    model="gemini-embedding-2",
                    google_api_key=api_key,
                    output_dimensionality=512,
                )
                formatted_query = f"task: search result | query: {text}"
                embedding = gemini_embed.embed_query(formatted_query)
                rag_cache.set_embedding(text, embedding)
                return embedding
            except Exception as exc:
                logger.warning("gemini_embedding_failed_using_local_fallback", error=str(exc))

        # Local Execution (Default BAAI/bge-small-en-v1.5)
        model = get_embedding_model()
        embedding = model.encode(text, convert_to_numpy=True).tolist()
        rag_cache.set_embedding(text, embedding)
        return embedding
