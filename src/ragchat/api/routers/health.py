"""API Router for health checks."""

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from opensearchpy import AsyncOpenSearch
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from ragchat.api.deps import get_db, get_search_client

logger = structlog.get_logger(__name__)
router = APIRouter()


@router.get("/health", status_code=status.HTTP_200_OK)
async def health_check(
    db: AsyncSession = Depends(get_db),
    os_client: AsyncOpenSearch = Depends(get_search_client),
) -> dict:
    """Consolidated health check verifying database and search engine connectivity."""
    db_healthy = False
    os_healthy = False

    # Check Database
    try:
        await db.execute(text("SELECT 1;"))
        db_healthy = True
    except Exception as exc:
        logger.error("database_health_check_failed", error=str(exc))

    # Check OpenSearch
    try:
        health = await os_client.cluster.health()
        if health.get("status") in ("green", "yellow"):
            os_healthy = True
    except Exception as exc:
        logger.error("opensearch_health_check_failed", error=str(exc))

    if not db_healthy or not os_healthy:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "status": "unhealthy",
                "database": "ok" if db_healthy else "failed",
                "opensearch": "ok" if os_healthy else "failed",
            },
        )

    return {
        "status": "healthy",
        "database": "ok",
        "opensearch": "ok",
    }
