.PHONY: publish test test-core test-integration lint lint-core models-table docs-serve

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

models-table:
	uv run python scripts/generate_verified_models_table.py

docs-serve:
	uv run mkdocs serve -a localhost:8001
