PYTHON ?= python
# CPU wheels keep installs small; point this at a CUDA index (e.g. .../whl/cu124) for a GPU build.
TORCH_INDEX ?= https://download.pytorch.org/whl/cpu
STEPS ?= 2000

.DEFAULT_GOAL := help
.PHONY: help install run view dev test lint build clean

help: ## List available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-8s\033[0m %s\n", $$1, $$2}'

install: ## Install PyTorch (CPU), the engine with dev/report extras, and the web viewer
	$(PYTHON) -m pip install torch --index-url $(TORCH_INDEX)
	$(PYTHON) -m pip install -e ".[dev]" matplotlib imageio imageio-ffmpeg
	cd web && npm ci --no-audit --no-fund

run: ## Headless run that prints population stats (STEPS=2000)
	$(PYTHON) -m primordial run --steps $(STEPS)

view: ## Interactive pygame viewer
	$(PYTHON) -m primordial view

dev: ## Browser version with hot reload (Vite dev server)
	cd web && npm run dev

test: ## pytest suite (engine) + vitest suite (web)
	$(PYTHON) -m pytest -q
	cd web && npm test

lint: ## ruff on the Python code, tsc --noEmit on the web code
	$(PYTHON) -m ruff check .
	cd web && npm run typecheck

build: ## Single-file browser build into web/dist/index.html
	cd web && npm run build

clean: ## Remove caches and build output
	rm -rf .pytest_cache .ruff_cache primordial.egg-info web/dist
	find . -name __pycache__ -type d -not -path './.venv/*' -not -path './web/node_modules/*' -prune -exec rm -rf {} +
