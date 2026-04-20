.PHONY: publish test test-core test-integration lint lint-core

publish:
	rm -rf dist/
	uv build
	@export $$(grep UV_PUBLISH_TOKEN .env | xargs) && uv publish

test:
	uv run pytest tests

test-core:
	uv run pytest tests

test-integration:
	uv run pytest -m integration

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

lint-core:
	uv run ruff check src tests && uv run ruff format --check src tests
