# RAG Chatbot

Docling → PostgreSQL → OpenSearch (hybrid BM25 + dense) → RAG (query expansion + retrieval + prompting) → Groq → Langfuse → Gradio + FastAPI.

See `DESIGN.md` for the full architecture write-up.

## Status

**Step 1 & 2 complete:** repo skeleton + tooling, Postgres + OpenSearch running locally via Docker Compose.
**Step 3 (next):** Docling ingestion of sample PDFs into Postgres.

## Setup

```bash
# 1. Install uv if you don't have it: https://docs.astral.sh/uv/getting-started/installation/

# 2. Install dependencies (creates .venv automatically)
uv sync --extra dev

# 3. Copy env file and adjust if needed
cp .env.example .env

# 4. Start Postgres + OpenSearch
make up

# 5. Wait ~10-20s for OpenSearch to be healthy, then create the index
make init-index

# 6. Confirm everything is reachable
make check-health

# 7. Install pre-commit hooks
uv run pre-commit install
```

## Useful commands

```bash
make up             # start postgres + opensearch (+ dashboards)
make down           # stop everything
make logs           # tail container logs
make ps             # container status
make init-index     # create the OpenSearch chunks index
make check-health   # verify postgres + opensearch connectivity
make test           # run pytest
make lint           # ruff check
make format         # ruff format
```

OpenSearch Dashboards (optional UI): http://localhost:5601
OpenSearch API: http://localhost:9200
