"""FastAPI dependency injection utilities."""

from typing import AsyncGenerator
from opensearchpy import AsyncOpenSearch
from sqlalchemy.ext.asyncio import AsyncSession

from ragchat.search.opensearch_client import get_os_client_dep
from ragchat.storage.db import get_db_session

# Re-expose session and client dependencies for easy router imports
get_db = get_db_session
get_search_client = get_os_client_dep
