.PHONY: help install lint check test clean

UV := $(shell command -v uv 2> /dev/null)

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  install   Install dependencies"
	@echo "  lint      Format and autofix, then type-check"
	@echo "  check     Verify formatting and lint (no changes), then type-check"
	@echo "  test      Run the CHAP integration tests (requires the chap CLI)"
	@echo "  clean     Remove caches and run artifacts"

install:
	@echo ">>> Installing dependencies"
	@$(UV) sync

lint:
	@echo ">>> Formatting and autofixing"
	@$(UV) run ruff format .
	@$(UV) run ruff check . --fix
	@echo ">>> Type checking"
	@$(UV) run mypy train.py predict.py model.py tests
	@$(UV) run pyright

check:
	@echo ">>> Checking formatting and lint"
	@$(UV) run ruff format --check .
	@$(UV) run ruff check .
	@echo ">>> Type checking"
	@$(UV) run mypy train.py predict.py model.py tests
	@$(UV) run pyright

test:
	@echo ">>> Running CHAP integration tests"
	@$(UV) run pytest -q

clean:
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@rm -rf runs output *.nc

.DEFAULT_GOAL := help
