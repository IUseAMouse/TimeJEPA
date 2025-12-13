# Makefile
.PHONY: help install activate format lint typecheck check-all clean
.PHONY: download-data list-datasets analyze-data setup-all train evaluate
.PHONY: finetune-linear finetune-full scaling-analysis

# Default target
.DEFAULT_GOAL := help

# Default values
EPOCHS ?= 30
CONTEXT_LENGTH ?= 384
TOTAL_POINTS ?= 59000000

##@ General

help: ## Display this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Setup

install: ## Install dependencies in editable mode
	uv venv
	source .venv/bin/activate
	uv pip install -e ".[dev]"
	@echo "✓ Installation complete. Activate venv with: source .venv/bin/activate"

##@ Data

list-datasets: ## List available datasets
	python scripts/download_data.py --list

download-data: ## Download dataset (usage: make download-data DATASET=electricity)
	python scripts/download_data.py --dataset $(or $(DATASET),electricity) -v

download-all: ## Download all datasets
	@echo "📥 Downloading all datasets..."
	python scripts/download_data.py --dataset all -v
	@echo "✓ Download complete!"

count-data: ## Count total datapoints in processed datasets
	@echo "📊 Counting datapoints..."
	python scripts/compute_dataset_stats.py --data-dir data

##@ Scaling Laws

scaling-analysis: ## Compute optimal model config (usage: make scaling-analysis TOTAL_POINTS=60000000 EPOCHS=50)
	@echo "\n🔬 TimeJEPA Scaling Law Analysis"
	@echo "================================"
	python scripts/compute_model_config.py \
		--total-points $(TOTAL_POINTS) \
		--epochs $(EPOCHS) \
		--context-length $(CONTEXT_LENGTH)

scaling-quick: ## Quick scaling analysis with defaults (60M points, 50 epochs)
	@$(MAKE) scaling-analysis TOTAL_POINTS=59000000 EPOCHS=30 CONTEXT_LENGTH=384

setup-all: ## Download all datasets and analyze (complete setup)
	@echo "🚀 Complete setup: download + analysis"
	@echo "========================================"
	@$(MAKE) download-all
	@$(MAKE) scaling-analysis
	@echo "\n✅ Setup complete! Next steps:"
	@echo "  1. Review the recommendations above"
	@echo "  2. Update configs/model/ with recommended config"
	@echo "  3. Run 'make train CONFIG=<config_name>' to start training\n"

##@ Training

train: ## Pretrain model (usage: make train CONFIG=base)
	@if [ -z "$(CONFIG)" ]; then \
		echo "❌ Error: CONFIG is not set."; \
		echo "Usage: make train CONFIG=tiny|small|base|large"; \
		echo ""; \
		echo "Available configs:"; \
		ls -1 configs/model/*.yaml 2>/dev/null | xargs -I {} basename {} .yaml | sed 's/^/  - /'; \
		exit 1; \
	fi
	@echo "🚀 Starting pretraining with config: $(CONFIG)"
	python scripts/train.py --config-name $(CONFIG) $(ARGS)

train-tiny: ## Quick pretrain with tiny config (for debugging)
	@$(MAKE) train CONFIG=tiny ARGS="training.max_epochs=5 $(ARGS)"

train-base: ## Pretrain with base config (recommended for 60M points)
	@$(MAKE) train CONFIG=base $(ARGS)

train-resume: ## Resume training from checkpoint (usage: make train-resume CONFIG=base CHECKPOINT=path/to/ckpt)
	@if [ -z "$(CHECKPOINT)" ]; then \
		echo "❌ Error: CHECKPOINT is not set."; \
		echo "Usage: make train-resume CONFIG=base CHECKPOINT=checkpoints/epoch=10.ckpt"; \
		exit 1; \
	fi
	@echo "🔄 Resuming training from: $(CHECKPOINT)"
	python scripts/train.py --config-name $(or $(CONFIG),base) \
		+trainer.resume_from_checkpoint="$(CHECKPOINT)" \
		$(ARGS)

##@ Finetuning

finetune: ## Linear Probe: Freeze encoder, train decoder only
	@if [ -z "$(CHECKPOINT)" ]; then \
		echo "❌ Error: CHECKPOINT is not set."; \
		echo "Usage: make finetune-linear CHECKPOINT=checkpoints/model.ckpt [CONFIG=tiny] [STRIDE=48] [EPOCHS=40] [LR=1e-4]"; \
		exit 1; \
	fi
	@echo "🧊 Starting Linear Probe (encoder frozen)"
	@echo "   Checkpoint: $(CHECKPOINT)"
	python scripts/train.py --config-name $(or $(CONFIG),tiny) \
        training.mode=finetune \
        training.finetune_mode=$(or $(MODE),full_finetune) \
        +training.pretrained_encoder_path="$(CHECKPOINT)" \
        model.decoder.type=$(or $(DECODER),mlp) \
        training.max_epochs=$(or $(EPOCHS),40) \
        training.optimizer.learning_rate=$(or $(LR),1e-4) \
        wandb.run_name=linear-probe-$(shell date +%Y%m%d-%H%M) \
        data.stride=$(or $(STRIDE),48) \
        $(ARGS)


##@ Evaluation

evaluate: ## Evaluate trained model (usage: make evaluate CHECKPOINT=path/to/ckpt)
	@if [ -z "$(CHECKPOINT)" ]; then \
		echo "❌ Error: CHECKPOINT is not set."; \
		echo "Usage: make evaluate CHECKPOINT=checkpoints/model.ckpt [DATASET=electricity]"; \
		exit 1; \
	fi
	python scripts/evaluate.py \
		--checkpoint $(CHECKPOINT) \
		$(if $(DATASET),--dataset $(DATASET),)

evaluate-all: ## Evaluate on all test datasets
	@if [ -z "$(CHECKPOINT)" ]; then \
		echo "❌ Error: CHECKPOINT is not set."; \
		exit 1; \
	fi
	python scripts/evaluate.py --checkpoint $(CHECKPOINT) --all-datasets

##@ Code Quality

format: ## Format code with ruff
	ruff format src/ scripts/
	ruff check --fix src/ scripts/

lint: ## Lint code with ruff
	ruff check src/ scripts/

typecheck: ## Type check with mypy
	mypy src/

check-all: format lint typecheck ## Run all code quality checks
	@echo "✓ All checks passed!"

##@ Cleanup

clean: ## Remove build artifacts and caches
	rm -rf build/ dist/ *.egg-info .pytest_cache/ .mypy_cache/ .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

clean-data: ## Remove downloaded and processed data (WARNING: destructive)
	rm -rf data/raw/ data/processed/
	@echo "⚠️  Data removed. Run 'make download-all' to re-download."

clean-checkpoints: ## Remove all checkpoints (WARNING: destructive)
	@echo "⚠️  This will delete ALL checkpoints!"
	@read -p "Are you sure? [y/N] " confirm && [ "$$confirm" = "y" ] || exit 1
	rm -rf checkpoints/ lightning_logs/
	@echo "✓ Checkpoints removed."

clean-all: clean clean-data ## Remove all generated files (keeps checkpoints)

##@ Utilities

tensorboard: ## Launch TensorBoard
	tensorboard --logdir lightning_logs/

wandb-sync: ## Sync offline W&B runs
	wandb sync --sync-all

gpu-status: ## Show GPU status
	@nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv

watch-gpu: ## Watch GPU status (updates every 1s)
	watch -n 1 nvidia-smi