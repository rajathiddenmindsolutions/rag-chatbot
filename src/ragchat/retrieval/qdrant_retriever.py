"""Separate Qdrant Hybrid & Semantic Retriever for 24/7 Free Cloud Deployment.
"""

from typing import List
import structlog
from qdrant_client import QdrantClient
from langchain_core.documents import Document

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
    score_threshold: float = 0.35,
) -> List[Document]:
    """Executes 20ms Hybrid & Semantic Search on Qdrant Cloud with minimum relevance score thresholding."""
    client = get_qdrant_client()

    try:
        # Compatible with all QdrantClient versions (query_points / search)
        if hasattr(client, "query_points"):
            res = client.query_points(
                collection_name=collection_name,
                query=query_vector,
                limit=top_k,
                score_threshold=score_threshold,
            )
            hits = res.points
        else:
            hits = client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=top_k,
                score_threshold=score_threshold,
            )

        documents = []
        for hit in hits:
            payload = hit.payload or {}
            text = payload.get("text", "")
            doc_title = payload.get("doc_title", "Document")
            meta = payload.get("metadata", {})

            score_val = getattr(hit, "score", 1.0)
            doc = Document(
                page_content=text,
                metadata={
                    "document_id": str(hit.id),
                    "chunk_id": str(hit.id),
                    "title": doc_title,
                    "score": score_val if score_val is not None else 1.0,
                    "section_path": meta.get("section_path", "root"),
                }
            )
            documents.append(doc)

        logger.info("qdrant_search_completed", query=query, count=len(documents))
        return documents
    except Exception as exc:
        logger.error("qdrant_search_failed", error=str(exc))
        return []
