# Makefile for Footy Scan
# Use `make help` to see available targets.

UV ?= uv

.PHONY: help
help: ## Show this help
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z0-9_-]+:.*##/ {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

.PHONY: setup
setup: deps
	git lfs install

.PHONY: deps
deps: ## Install python deps (editable)
	$(UV) sync

.PHONY: deps-update
deps-update:
	$(UV) lock --upgrade
	$(UV) sync

.PHONY: ml-deps
ml-deps: ## Install optional ML/video extras
	$(UV) pip install "torch" "opencv-python" "tqdm"

.PHONY: run
run: ## Run the main entrypoint
	$(UV) run footy-track

# test target intentionally commented out

.PHONY: docs
docs: docs-serve ## Serve docs locally (alias)

.PHONY: docs-serve
docs-serve: ## Serve MkDocs (live reload)
	$(UV) run mkdocs serve

.PHONY: docs-build
docs-build: ## Build MkDocs site
	$(UV) run mkdocs build

.PHONY: docs-deploy
docs-deploy: ## Deploy MkDocs to GitHub Pages
	$(UV) run mkdocs gh-deploy

# -----------------------------
# Linting / pre-commit
# -----------------------------
.PHONY: pre-commit-staged
pcr: pre-commit-staged
pre-commit-staged: ## Run pre-commit on staged files only
	$(UV) run pre-commit run

.PHONY: pre-commit-all
pcra: pre-commit-all
pre-commit-all: ## Run pre-commit on all files
	$(UV) run pre-commit run --all-files
