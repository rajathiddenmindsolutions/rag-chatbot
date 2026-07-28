"""PostgreSQL Database Session & Engine setup."""

from collections.abc import AsyncGenerator
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from ragchat.config import settings

# Create asynchronous engine.
# We set pool_pre_ping=True to automatically reconnect if the database restarts.
engine = create_async_engine(
    settings.postgres_dsn,
    pool_pre_ping=True,
    future=True,
    echo=False,
)

# Async session factory
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency / context manager yielding an active async database session.

    Rolls back automatically on unhandled exceptions and closes session.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
