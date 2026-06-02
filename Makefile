.PHONY: help install lint check test eval clean

UV := $(shell command -v uv 2> /dev/null)
AR_N_ITER ?= 30

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  install   Install dependencies"
	@echo "  lint      Format and autofix, then type-check"
	@echo "  check     Verify formatting and lint (no changes), then type-check"
	@echo "  test      Run the self-contained train/predict pipeline tests"
	@echo "  eval      Run a chap eval backtest (chap CLI via uvx; no chap-core dep)"
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
	@echo ">>> Running pipeline tests"
	@$(UV) run pytest -q

eval:
	@echo ">>> Running chap eval (chap CLI via uvx)"
	@AR_N_ITER=$(AR_N_ITER) uvx --from chap-core chap eval \
		--model-name . \
		--dataset-csv input/trainData.csv \
		--output-file eval.nc \
		--backtest-params.n-splits 2 \
		--backtest-params.n-periods 1
	@echo ">>> Verifying chap eval output (eval.nc)"
	@$(UV) run --with xarray --with netcdf4 python scripts/verify_eval.py eval.nc

clean:
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@rm -rf runs output *.nc

.DEFAULT_GOAL := help
