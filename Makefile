.PHONY: publish test test-integration lint

publish:
	uv build
	@export $$(grep UV_PUBLISH_TOKEN .env | xargs) && uv publish

test:
	uv run pytest

test-integration:
	uv run pytest -m integration

lint:
	uv run ruff check src tests && uv run ruff format --check src tests
