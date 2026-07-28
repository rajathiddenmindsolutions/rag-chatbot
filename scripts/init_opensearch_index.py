"""Create the OpenSearch chunks index from infra/opensearch/index_template.json.

Usage:
    uv run python scripts/init_opensearch_index.py
"""

import json
import sys
from pathlib import Path

from opensearchpy import OpenSearch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ragchat.config import settings  # noqa: E402
from ragchat.logging_conf import configure_logging  # noqa: E402
import structlog  # noqa: E402

configure_logging()
logger = structlog.get_logger(__name__)

TEMPLATE_PATH = Path(__file__).parent.parent / "infra" / "opensearch" / "index_template.json"


def main() -> None:
    client = OpenSearch(
        hosts=[{"host": settings.opensearch_host, "port": settings.opensearch_port}],
        use_ssl=settings.opensearch_use_ssl,
        verify_certs=False,
    )

    index_name = settings.opensearch_index
    body = json.loads(TEMPLATE_PATH.read_text())

    if client.indices.exists(index=index_name):
        logger.info("deleting_existing_index", index=index_name)
        client.indices.delete(index=index_name)

    client.indices.create(index=index_name, body=body)
    logger.info("index_created", index=index_name)


if __name__ == "__main__":
    main()
