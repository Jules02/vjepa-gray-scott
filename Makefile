.ONESHELL:
.PHONY: help install train eval baselines
.DEFAULT_GOAL := help

# Override on the command line, e.g.:  make train CFG=gray_scott/cfgs/train_large.yaml
CFG ?= gray_scott/cfgs/train.yaml
CKPT ?=
H ?= 30
SPLIT ?= test

help: ## Show this help message
	@grep -hE '^[A-Za-z0-9_ \-]*?:.*##.*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

install: ## Install the package + dependencies (eb_jepa core; add `the-well` for baselines/data)
	uv sync

train: ## Train the Gray-Scott temporal V-JEPA (CFG=gray_scott/cfgs/train.yaml)
	uv run python -m gray_scott.main --fname $(CFG)

eval: ## Evaluate a checkpoint with VRMSE (make eval CKPT=<run>/latest.pth.tar [H=30] [SPLIT=test])
	@test -n "$(CKPT)" || { echo "set CKPT=<path/to/ckpt.pth.tar>"; exit 1; }
	uv run python -m gray_scott.eval --ckpt $(CKPT) --H $(H) --split $(SPLIT)

baselines: ## Train + score the The Well neural baselines (needs `the-well`)
	uv run python -m gray_scott.baselines --split $(SPLIT) --H $(H)
