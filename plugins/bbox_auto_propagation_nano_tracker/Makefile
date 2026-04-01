.PHONY: format type-check

format:
	uv run ruff format .
	uv run ruff check --fix .

type-check:
	uv run mypy --config-file ../../mypy.ini .
