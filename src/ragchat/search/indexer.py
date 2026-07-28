"""Bulk indexing utility for uploading document chunks into OpenSearch."""

import structlog
from opensearchpy import AsyncOpenSearch
from opensearchpy.helpers import async_bulk

from ragchat.config import settings

logger = structlog.get_logger(__name__)


async def index_chunks(client: AsyncOpenSearch, chunks_data: list[dict]) -> list[str]:
    """Bulk index a list of chunks with metadata and embeddings into OpenSearch.

    Args:
        client: AsyncOpenSearch client instance.
        chunks_data: List of dicts representing chunk records. Expected fields:
            - chunk_id: str (UUID)
            - document_id: str (UUID)
            - chunk_index: int
            - text: str
            - section_path: str
            - title: str
            - authors: list[str]
            - chunking_strategy: str
            - embedding: list[float]

    Returns:
        List of chunk_ids that were successfully indexed.
    """
    if not chunks_data:
        return []

    index_name = settings.opensearch_index
    actions = []

    for c in chunks_data:
        action = {
            "_index": index_name,
            "_id": str(c["chunk_id"]),
            "_source": {
                "document_id": str(c["document_id"]),
                "chunk_id": str(c["chunk_id"]),
                "chunk_index": c["chunk_index"],
                "text": c["text"],
                "section_path": c.get("section_path") or "",
                "title": c.get("title") or "",
                "authors": c.get("authors") or [],
                "chunking_strategy": c["chunking_strategy"],
                "embedding": c["embedding"],
            },
        }
        actions.append(action)

    try:
        success, failed = await async_bulk(client, actions, refresh=True)
        logger.info("bulk_index_completed", success_count=success, failed_count=len(failed) if isinstance(failed, list) else failed)
        if failed:
            logger.warn("bulk_index_failures", details=failed)
        return [str(c["chunk_id"]) for c in chunks_data]
    except Exception as exc:
        logger.error("bulk_index_failed", error=str(exc))
        raise


async def delete_document_chunks(client: AsyncOpenSearch, document_id: str) -> int:
    """Delete all indexed chunks belonging to a single document from OpenSearch.

    Returns the number of chunks deleted (best-effort; 0 if the index
    doesn't exist yet or nothing matched).
    """
    index_name = settings.opensearch_index
    body = {"query": {"term": {"document_id": str(document_id)}}}
    try:
        response = await client.delete_by_query(index=index_name, body=body, refresh=True)
        deleted = response.get("deleted", 0)
        logger.info("opensearch_document_chunks_deleted", document_id=str(document_id), deleted_count=deleted)
        return deleted
    except Exception as exc:
        logger.error("opensearch_delete_failed", document_id=str(document_id), error=str(exc))
        raise


async def delete_all_chunks(client: AsyncOpenSearch) -> int:
    """Delete every chunk in the index (full corpus reset). Returns the
    number of documents deleted from the index (best-effort).
    """
    index_name = settings.opensearch_index
    body = {"query": {"match_all": {}}}
    try:
        response = await client.delete_by_query(index=index_name, body=body, refresh=True)
        deleted = response.get("deleted", 0)
        logger.info("opensearch_all_chunks_deleted", deleted_count=deleted)
        return deleted
    except Exception as exc:
        logger.error("opensearch_delete_all_failed", error=str(exc))
        raise
