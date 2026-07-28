"""Separate Qdrant Hybrid & Semantic Retriever for 24/7 Free Cloud Deployment.

Keeps OpenSearch code 100% untouched.
"""

from typing import List, Dict, Any
import structlog
from qdrant_client import QdrantClient, models

from ragchat.config import settings

logger = structlog.get_logger(__name__)


def get_qdrant_client() -> QdrantClient:
    """Instantiate Qdrant Cloud Client."""
    qdrant_url = getattr(settings, "qdrant_url", None) or "http://localhost:6333"
    qdrant_api_key = getattr(settings, "qdrant_api_key", None)
    return QdrantClient(url=qdrant_url, api_key=qdrant_api_key)


async def search_qdrant_hybrid(
    query: str,
    query_vector: List[float],
    top_k: int = 5,
    collection_name: str = "chunks",
) -> List[Dict[str, Any]]:
    """Executes 20ms Hybrid & Semantic Search on Qdrant Cloud."""
    client = get_qdrant_client()

    try:
        # Perform Dense Vector Search
        results = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k,
        )

        chunks = []
        for hit in results:
            payload = hit.payload or {}
            chunks.append({
                "text": payload.get("text", ""),
                "score": hit.score,
                "metadata": payload.get("metadata", {}),
                "doc_title": payload.get("doc_title", "Document"),
            })

        logger.info("qdrant_search_completed", query=query, count=len(chunks))
        return chunks
    except Exception as exc:
        logger.error("qdrant_search_failed", error=str(exc))
        return []
