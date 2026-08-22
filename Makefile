.PHONY: build check publish test test-core test-integration lint lint-core models-table health-report discover-models probe-quirks docs-build docs-serve

build:
	uv build

check: lint test docs-build build

publish:
	rm -rf dist/
	$(MAKE) build
	@export $$(grep UV_PUBLISH_TOKEN .env | xargs) && uv publish

test test-core:
	uv run pytest

test-integration:
	uv run pytest -m integration

lint lint-core:
	uv run ruff check src tests scripts
	uv run ruff format --check src tests scripts

models-table:
	uv run python scripts/generate_verified_models_table.py

health-report:
	uv run python scripts/model_reliability.py --markdown --output "docs/usage/Provider Health.md"
	uv run python scripts/model_reliability.py --html --output "dist/provider-health.html"
	open "dist/provider-health.html"

discover-models:
	uv run python scripts/discover_new_models.py

probe-quirks:
	uv run python scripts/probe_model_quirks.py
	uv run python scripts/generate_verified_models_table.py

docs-build:
	uv run mkdocs build --strict

docs-serve:
	uv run mkdocs serve -a localhost:8001
