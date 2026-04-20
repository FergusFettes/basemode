.PHONY: publish publish-loom test test-core test-loom test-integration lint lint-core lint-loom

publish:
	rm -rf dist/
	uv build
	@export $$(grep UV_PUBLISH_TOKEN .env | xargs) && uv publish

publish-loom:
	cd packages/basemode-loom && rm -rf dist/ && uv build
	cd packages/basemode-loom && export $$(grep UV_PUBLISH_TOKEN ../../.env | xargs) && uv publish

test:
	uv run pytest tests
	cd packages/basemode-loom && PYTHONPATH=../../src:src uv run pytest tests

test-core:
	uv run pytest tests

test-loom:
	cd packages/basemode-loom && PYTHONPATH=../../src:src uv run pytest tests

test-integration:
	uv run pytest -m integration

lint:
	uv run ruff check src tests packages/basemode-loom/src packages/basemode-loom/tests
	uv run ruff format --check src tests packages/basemode-loom/src packages/basemode-loom/tests

lint-core:
	uv run ruff check src tests && uv run ruff format --check src tests

lint-loom:
	uv run ruff check packages/basemode-loom/src packages/basemode-loom/tests
	uv run ruff format --check packages/basemode-loom/src packages/basemode-loom/tests
