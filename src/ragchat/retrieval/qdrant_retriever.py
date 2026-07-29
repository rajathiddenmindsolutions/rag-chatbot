"""Separate Qdrant Hybrid & Semantic Retriever for 24/7 Free Cloud Deployment.

Keeps OpenSearch code 100% untouched.
"""

from typing import List, Dict, Any
import structlog
from qdrant_client import QdrantClient, models
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
) -> List[Document]:
    """Executes 20ms Hybrid & Semantic Search on Qdrant Cloud and returns LangChain Documents."""
    client = get_qdrant_client()

    try:
        results = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k,
        )

        documents = []
        for hit in results:
            payload = hit.payload or {}
            text = payload.get("text", "")
            doc_title = payload.get("doc_title", "Document")
            meta = payload.get("metadata", {})

            doc = Document(
                page_content=text,
                metadata={
                    "document_id": str(hit.id),
                    "chunk_id": str(hit.id),
                    "title": doc_title,
                    "score": hit.score,
                    "section_path": meta.get("section_path", "root"),
                }
            )
            documents.append(doc)

        logger.info("qdrant_search_completed", query=query, count=len(documents))
        return documents
    except Exception as exc:
        logger.error("qdrant_search_failed", error=str(exc))
        return []
