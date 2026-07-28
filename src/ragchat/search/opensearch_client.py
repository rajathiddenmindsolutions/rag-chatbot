"""Async OpenSearch client wrapper with index mapping initialization."""

from collections.abc import AsyncGenerator
import structlog
from opensearchpy import AsyncOpenSearch
from ragchat.config import settings

logger = structlog.get_logger(__name__)

CHUNKS_INDEX_MAPPING = {
    "settings": {
        "index.knn": True,
        "number_of_shards": 1,
        "number_of_replicas": 0,
    },
    "mappings": {
        "properties": {
            "document_id": {"type": "keyword"},
            "chunk_id": {"type": "keyword"},
            "chunk_index": {"type": "integer"},
            "text": {"type": "text", "analyzer": "english"},
            "section_path": {
                "type": "text",
                "fields": {"raw": {"type": "keyword"}},
            },
            "title": {
                "type": "text",
                "fields": {"raw": {"type": "keyword"}},
            },
            "authors": {"type": "keyword"},
            "chunking_strategy": {"type": "keyword"},
            "embedding": {
                "type": "knn_vector",
                "dimension": 384,
                "method": {
                    "name": "hnsw",
                    "space_type": "cosinesimil",
                    "engine": "lucene",
                    "parameters": {
                        "ef_construction": 128,
                        "m": 16,
                    },
                },
            },
        }
    },
}


def get_opensearch_client() -> AsyncOpenSearch:
    """Instantiate and return the async OpenSearch client."""
    client = AsyncOpenSearch(
        hosts=[{"host": settings.opensearch_host, "port": settings.opensearch_port}],
        use_ssl=settings.opensearch_use_ssl,
        verify_certs=False,
        ssl_show_warn=False,
    )
    return client


async def ensure_index_exists(client: AsyncOpenSearch) -> None:
    """Ensure the chunks index exists with proper knn_vector mappings."""
    index_name = settings.opensearch_index
    try:
        exists = await client.indices.exists(index=index_name)
        if not exists:
            await client.indices.create(index=index_name, body=CHUNKS_INDEX_MAPPING)
            logger.info("opensearch_index_created", index=index_name)
        else:
            logger.info("opensearch_index_already_exists", index=index_name)
    except Exception as exc:
        logger.error("opensearch_index_check_failed", error=str(exc))


async def get_os_client_dep() -> AsyncGenerator[AsyncOpenSearch, None]:
    """Dependency injection yield for OpenSearch client."""
    client = get_opensearch_client()
    try:
        yield client
    finally:
        await client.close()
