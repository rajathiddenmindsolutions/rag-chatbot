"""Langfuse observability and tracing configuration."""

from typing import Optional
import structlog
from langfuse.callback import CallbackHandler
from ragchat.config import settings

logger = structlog.get_logger(__name__)


def get_langfuse_callback() -> Optional[CallbackHandler]:
    """Instantiate and return the Langfuse callback handler for LangChain/LangGraph.

    Returns None if Langfuse is not configured.
    """
    if (
        not settings.langfuse_public_key
        or settings.langfuse_public_key == "changeme"
        or not settings.langfuse_secret_key
        or settings.langfuse_secret_key == "changeme"
    ):
        return None

    try:
        handler = CallbackHandler(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
        logger.info("langfuse_callback_configured", host=settings.langfuse_host)
        return handler
    except Exception as exc:
        logger.error("langfuse_callback_config_failed", error=str(exc))
        return None
