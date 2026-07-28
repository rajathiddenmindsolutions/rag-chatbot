"""Verify Postgres and OpenSearch are up and reachable with the current config.

Usage:
    uv run python scripts/check_health.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import psycopg2  # noqa: E402
import structlog  # noqa: E402
from opensearchpy import OpenSearch  # noqa: E402

from ragchat.config import settings  # noqa: E402
from ragchat.logging_conf import configure_logging  # noqa: E402

configure_logging()
logger = structlog.get_logger(__name__)


def check_postgres() -> bool:
    try:
        conn = psycopg2.connect(
            host=settings.postgres_host,
            port=settings.postgres_port,
            dbname=settings.postgres_db,
            user=settings.postgres_user,
            password=settings.postgres_password,
        )
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM documents;")
            (count,) = cur.fetchone()
        conn.close()
        logger.info("postgres_ok", documents_table_row_count=count)
        return True
    except Exception as exc:
        logger.error("postgres_failed", error=str(exc))
        return False


def check_opensearch() -> bool:
    try:
        client = OpenSearch(
            hosts=[{"host": settings.opensearch_host, "port": settings.opensearch_port}],
            use_ssl=settings.opensearch_use_ssl,
            verify_certs=False,
        )
        health = client.cluster.health()
        logger.info("opensearch_ok", status=health.get("status"))
        return True
    except Exception as exc:
        logger.error("opensearch_failed", error=str(exc))
        return False


def main() -> None:
    pg_ok = check_postgres()
    os_ok = check_opensearch()
    if pg_ok and os_ok:
        logger.info("all_systems_healthy")
        sys.exit(0)
    else:
        logger.error("health_check_failed", postgres=pg_ok, opensearch=os_ok)
        sys.exit(1)


if __name__ == "__main__":
    main()
