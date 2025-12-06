# Makefile
.PHONY: help install activate format lint typecheck check-all clean
.PHONY: download-data list-datasets analyze-data setup-all train evaluate
.PHONY: finetune-linear finetune-full

# Default target
.DEFAULT_GOAL := help

##@ General

help: ## Display this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

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

analyze-data: ## Analyze datasets and compute optimal model config
	@echo "\n📊 Step 1/2: Computing dataset statistics..."
	@echo "================================================"
	python scripts/compute_dataset_stats.py --data-dir data
	@echo "\n📐 Step 2/2: Computing optimal model configuration..."
	@echo "================================================"
	python scripts/compute_model_config.py --data-dir data
	@echo "\n✓ Analysis complete! Update configs/model/jepa.yaml with recommended values.\n"

setup-all: ## Download all datasets and analyze (complete setup)
	@echo "🚀 Complete setup: download + analysis"
	@echo "========================================"
	@$(MAKE) download-all
	@$(MAKE) analyze-data
	@echo "\n✅ Setup complete! Next steps:"
	@echo "  1. Review the recommendations above"
	@echo "  2. Update configs/model/jepa.yaml"
	@echo "  3. Run 'make train' to start training\n"

##@ Training

train: ## Train model from scratch (Pretraining)
	python scripts/train.py --config-name tiny

finetune-linear: ## Linear Probe: Freeze encoder, train decoder. Usage: make finetune-linear CHECKPOINT=path/to/ckpt
	@if [ -z "$(CHECKPOINT)" ]; then \
		echo "❌ Error: CHECKPOINT is not set."; \
		echo "Usage: make finetune-linear CHECKPOINT=checkpoints/my_model/epoch=03.ckpt [EPOCHS=50] [LR=1e-3]"; \
		exit 1; \
	fi
	@echo "🚀 Starting Linear Probe with checkpoint: $(CHECKPOINT)"
	python scripts/train.py --config-name $(or $(CONFIG),tiny) \
		training.mode="finetune" \
		+training.finetune_mode="linear_probe" \
		+'training.pretrained_encoder_path="$(CHECKPOINT)"' \
		model.decoder.type="linear" \
		training.max_epochs=$(or $(EPOCHS),20) \
		training.optimizer.learning_rate=$(or $(LR),1e-4) \
		wandb.run_name="linear-probe-$(shell date +%Y%m%d-%H%M)" \
		$(ARGS)

finetune-full: ## Full Finetune: Train encoder + decoder. Usage: make finetune-full CHECKPOINT=path/to/ckpt
	@if [ -z "$(CHECKPOINT)" ]; then \
		echo "❌ Error: CHECKPOINT is not set."; \
		echo "Usage: make finetune-full CHECKPOINT=checkpoints/my_model/epoch=03.ckpt"; \
		exit 1; \
	fi
	@echo "🚀 Starting Full Finetune with checkpoint: $(CHECKPOINT)"
	python scripts/train.py --config-name $(or $(CONFIG),tiny) \
		training.mode="finetune" \
		training.finetune_mode="full_finetune" \
		'training.pretrained_encoder_path="$(CHECKPOINT)"' \
		model.decoder.type="mlp" \
		training.max_epochs=$(or $(EPOCHS),50) \
		training.optimizer.learning_rate=$(or $(LR),1e-4) \
		training.optimizer.encoder_lr_multiplier=0.1 \
		wandb.run_name="full-finetune-$(shell date +%Y%m%d-%H%M)" \
		$(ARGS)

##@ Evaluation

evaluate: ## Evaluate trained model
	python scripts/evaluate.py --checkpoint data/checkpoints/best.ckpt

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
	@echo "⚠ Data removed. Run 'make download-all' to re-download."

clean-all: clean clean-data ## Remove all generated files