"""Main FastAPI application entrypoint."""

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ragchat.api.routers.documents import router as documents_router
from ragchat.api.routers.health import router as health_router
from ragchat.api.routers.ingest import router as ingest_router
from ragchat.api.routers.query import router as query_router
from ragchat.api.routers.openai_compat import router as openai_compat_router
from ragchat.config import settings
from ragchat.logging_conf import configure_logging
from ragchat.search.embeddings import get_embedding_model

# Configure logging at start
configure_logging()
logger = structlog.get_logger(__name__)

app = FastAPI(
    title="RAG Chatbot API",
    description="Docling + Postgres + OpenSearch Hybrid RAG Chatbot Backend",
    version="0.1.0",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Wire routers
app.include_router(health_router, prefix="/api", tags=["Health"])
app.include_router(documents_router, prefix="/api", tags=["Documents"])
app.include_router(ingest_router, prefix="/api", tags=["Ingestion"])
app.include_router(query_router, prefix="/api", tags=["Query"])
app.include_router(openai_compat_router, tags=["OpenAI Compatibility"])


@app.on_event("startup")
async def on_startup():
    logger.info("api_startup_sequence_initiated", host=settings.api_host, port=settings.api_port)
    import asyncio

    # Try pre-warming local embedding model if available
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, get_embedding_model)
        logger.info("embedding_model_warmed_up")
    except Exception as emb_exc:
        logger.warning("embedding_model_warmup_skipped", error=str(emb_exc))

    # Try ensuring OpenSearch index exists if running locally
    try:
        from ragchat.search.opensearch_client import get_opensearch_client, ensure_index_exists
        os_client = get_opensearch_client()
        await ensure_index_exists(os_client)
        await os_client.close()
    except Exception as os_exc:
        logger.warning("opensearch_startup_check_skipped", error=str(os_exc))


@app.on_event("shutdown")
async def on_shutdown():
    logger.info("api_shutdown_sequence_initiated")
