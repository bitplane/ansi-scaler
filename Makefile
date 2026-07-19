.PHONY: help all install dev lock upgrade test coverage lint format clean \
	deps catalog prompts generate rembg lod pyramid classify verify review corpus smoke

RUN_CONFIG ?= configs/runs/first.yaml
RETRY_ERRORS ?= 0

all: dev test lint  ## Prepare the project and run all local checks

install: .venv/.installed  ## Create the locked runtime environment

dev: .venv/.installed-dev  ## Create the locked development environment

lock:  ## Refresh uv.lock without upgrading dependencies
	uv lock

upgrade:  ## Upgrade dependencies and refresh uv.lock
	uv lock --upgrade

test: .venv/.installed-dev  ## Run unit tests (no model downloads or GPU required)
	scripts/test.sh

coverage: .venv/.installed-dev  ## Run tests and build an HTML coverage report
	scripts/coverage.sh

lint: .venv/.installed-dev  ## Check formatting and lint Python code
	scripts/lint.sh

format: .venv/.installed-dev  ## Format and auto-fix Python code
	scripts/format.sh

clean:  ## Remove the venv and local caches, but never corpus data
	scripts/clean.sh

deps: .venv/.installed  ## Check host build tools, Python headers, CUDA, and GPU runtimes
	scripts/check-deps.sh

catalog: .venv/.installed  ## Validate the committed concept catalogue
	scripts/catalog.sh $(RUN_CONFIG)

prompts: .venv/.installed  ## Build a deterministic prompt manifest
	scripts/prompts.sh $(RUN_CONFIG)

generate: deps  ## Generate Sana rasters (downloads model; CUDA required)
	scripts/generate.sh $(RUN_CONFIG)

rembg: .venv/.installed  ## Remove raster backgrounds with the configured model
	scripts/rembg.sh $(RUN_CONFIG)

lod: .venv/.installed  ## Build SVG LODs and PNG previews with VTracer
	scripts/lod.sh $(RUN_CONFIG)

pyramid: .venv/.installed .tools/.chuda-0.1.1  ## Build compressed ANSI pyramids with pinned Chuda
	scripts/pyramid.sh $(RUN_CONFIG)

classify: .venv/.installed  ## Classify cutouts with the configured Ollama VLM
	scripts/classify.sh $(RUN_CONFIG)

verify: .venv/.installed  ## Verify VLM classifications with the configured Ollama LLM
	scripts/verify.sh $(RUN_CONFIG)

review: .venv/.installed  ## Open the local corpus review and annotation UI
	scripts/review.sh $(RUN_CONFIG)

corpus: deps .tools/.chuda-0.1.1  ## Resume the configured corpus through ANSI pyramids
	RETRY_ERRORS=$(RETRY_ERRORS) scripts/corpus.sh $(RUN_CONFIG)

smoke: deps .tools/.chuda-0.1.1  ## Run ten real samples through all stages (downloads models; CUDA required)
	RETRY_ERRORS=$(RETRY_ERRORS) scripts/corpus.sh configs/runs/smoke.yaml

.venv/.installed: pyproject.toml uv.lock .venv/.created scripts/install.sh
	scripts/install.sh

.venv/.installed-dev: pyproject.toml uv.lock .venv/.created scripts/install-dev.sh
	scripts/install-dev.sh

.tools/.chuda-0.1.1: scripts/install-chuda.sh
	scripts/install-chuda.sh 0.1.1

.venv/.created: .python-version scripts/venv.sh
	scripts/venv.sh

help:  ## Show this help
	@awk 'BEGIN {FS = ":.*?## "}; /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
