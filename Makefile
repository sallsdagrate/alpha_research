.PHONY: help test lint type check cli

help:
	@echo "Targets: cli, test, lint, type, check"

cli:
	PYTHONPATH=src python -m alpha_research --help

test:
	pytest

lint:
	ruff check .

type:
	pyright

check: lint type test
