.PHONY: up down logs ps init-index check-health test lint format

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f

ps:
	docker compose ps

init-index:
	uv run python scripts/init_opensearch_index.py

check-health:
	uv run python scripts/check_health.py

test:
	uv run pytest

lint:
	uv run ruff check .

format:
	uv run ruff format .
