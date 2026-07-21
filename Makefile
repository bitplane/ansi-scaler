.PHONY: help all install dev lock upgrade test coverage lint format clean \
	deps content prompts generate background lod pyramid classify verify review corpus smoke gc \
	dataset-plan dataset dataset-validate refiner-smoke refiner-train refiner-demo

RUN_CONFIG ?= configs/runs/first.yaml
RETRY_ERRORS ?= 0
GC_CONFIRM ?= 0
DATASET_RECIPE ?= configs/datasets/first.yaml
DATASET_LIMIT ?=
TRAINING_CONFIG ?= configs/training/refiner-first.yaml
OBJECT ?= wooden treasure chest
WIDTH ?= 40
CHECKPOINT ?=

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

gc: .venv/.installed  ## Compact superseded corpus data after showing a deletion report
	GC_CONFIRM=$(GC_CONFIRM) scripts/gc.sh $(RUN_CONFIG)

dataset-plan: .venv/.installed  ## Preview dataset selection, splits, cell count, and size
	DATASET_LIMIT=$(DATASET_LIMIT) scripts/dataset-plan.sh $(DATASET_RECIPE)

dataset: .venv/.installed  ## Compile resumable immutable safetensors dataset shards
	DATASET_LIMIT=$(DATASET_LIMIT) scripts/dataset.sh $(DATASET_RECIPE)

dataset-validate: .venv/.installed  ## Validate DATASET_DIR checksums, tensors, and split isolation
	scripts/dataset-validate.sh

refiner-smoke: .venv/.installed  ## Run the short end-to-end local ANSI refiner experiment
	.venv/bin/ansi-scaler refiner-train --training-config configs/training/refiner-smoke.yaml

refiner-train: .venv/.installed  ## Train or resume the full local ANSI refiner experiment
	.venv/bin/ansi-scaler refiner-train --training-config $(TRAINING_CONFIG)

refiner-demo: deps  ## Generate OBJECT and compare Chuda WIDTH with learned 1.5x ANSI
	.venv/bin/ansi-scaler refiner-demo "$(OBJECT)" --width $(WIDTH) --run-config $(RUN_CONFIG) --training-config $(TRAINING_CONFIG) $(if $(CHECKPOINT),--checkpoint $(CHECKPOINT),)

deps: .venv/.installed  ## Check host build tools, Python headers, CUDA, and GPU runtimes
	scripts/check-deps.sh

content: .venv/.installed  ## Validate authored theme/location/object specifications
	.venv/bin/ansi-scaler content validate --run-config $(RUN_CONFIG)

prompts: .venv/.installed  ## Build a deterministic prompt manifest
	scripts/prompts.sh $(RUN_CONFIG)

generate: deps prompts  ## Refresh prompts, then generate Sana rasters (downloads model; CUDA required)
	RETRY_ERRORS=$(RETRY_ERRORS) scripts/generate.sh $(RUN_CONFIG)

background: generate  ## Resume through RGBA extraction with the configured provider
	RETRY_ERRORS=$(RETRY_ERRORS) scripts/background.sh $(RUN_CONFIG)

lod: background  ## Resume through SVG LODs and PNG previews with VTracer
	RETRY_ERRORS=$(RETRY_ERRORS) scripts/lod.sh $(RUN_CONFIG)

pyramid: lod  ## Resume through compressed ANSI pyramids with the pinned Chuda Python backend
	RETRY_ERRORS=$(RETRY_ERRORS) scripts/pyramid.sh $(RUN_CONFIG)

classify: pyramid  ## Resume through cutout classification with the configured Ollama VLM
	RETRY_ERRORS=$(RETRY_ERRORS) scripts/classify.sh $(RUN_CONFIG)

verify: classify  ## Resume through VLM verification with the configured Ollama LLM
	RETRY_ERRORS=$(RETRY_ERRORS) scripts/verify.sh $(RUN_CONFIG)

review: .venv/.installed  ## Open the local corpus review and annotation UI
	scripts/review.sh $(RUN_CONFIG)

corpus: deps  ## Resume the configured corpus through VLM classification and LLM verification
	RETRY_ERRORS=$(RETRY_ERRORS) scripts/corpus.sh $(RUN_CONFIG)

smoke: deps  ## Run ten real samples through all stages (downloads models; CUDA required)
	RETRY_ERRORS=$(RETRY_ERRORS) scripts/corpus.sh configs/runs/smoke.yaml

.venv/.installed: pyproject.toml uv.lock .venv/.created scripts/install.sh
	scripts/install.sh

.venv/.installed-dev: pyproject.toml uv.lock .venv/.created scripts/install-dev.sh
	scripts/install-dev.sh

.venv/.created: .python-version scripts/venv.sh
	scripts/venv.sh

help:  ## Show this help
	@awk 'BEGIN {FS = ":.*?## "}; /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
