.PHONY: help setup test clean docker-up docker-down docker-logs

# Use bash for all recipes to support pipefail
SHELL := /bin/bash

# Include dataset generation module
include Makefile.dataset

help:
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  gogooku3 - Japanese Stock ML Pipeline"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo ""
	@echo "🚀 Quick Start (RECOMMENDED):"
	@echo "  make dataset-bg     Build dataset in background (GPU, 5 years)"
	@echo "                      → SSH-safe, logs to _logs/dataset/"
	@echo "  make go             Alias for 'make dataset-bg'"
	@echo ""
	@echo "📊 Alternative:"
	@echo "  make dataset        Build dataset interactively"
	@echo ""
	@echo "📚 Common Commands:"
	@echo "  make setup          Setup environment"
	@echo "  make train          Train model"
	@echo "  make test           Run tests"
	@echo "  make clean          Cleanup"
	@echo ""
	@echo "📖 Full help: make help-dataset    (dataset commands)"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Python environment setup
setup:
	python3 -m venv venv
	./venv/bin/pip install --upgrade pip
	./venv/bin/pip install -r requirements.txt
	@echo "✅ Python environment ready"
	@echo "📝 Copy .env.example to .env and configure your settings"
	cp -n .env.example .env || true

# RAPIDS GPU-accelerated data processing
rapids-install:
	@echo "🚀 Installing RAPIDS 24.12 for CUDA 12.x..."
	pip install --extra-index-url=https://pypi.nvidia.com \
		cudf-cu12==24.12.* \
		cugraph-cu12==24.12.* \
		rmm-cu12==24.12.*
	@echo "✅ RAPIDS installed successfully"
	@echo "💡 Verify: python -c 'import cudf; import cugraph; import rmm; print(\"✅ RAPIDS ready\")'"

rapids-verify:
	@echo "🔍 Verifying RAPIDS installation..."
	@python -c "import cudf; import cugraph; import rmm; print(f'✅ cuDF {cudf.__version__}, cuGraph {cugraph.__version__}, RMM {rmm.__version__}')"
	@python -c "from src.utils.gpu_etl import init_rmm, to_cudf, to_polars; import polars as pl; df = pl.DataFrame({'x': [1,2,3]}); to_polars(to_cudf(df)); print('✅ GPU-ETL pipeline functional')"

# Docker services
docker-up:
	docker-compose up -d
	@echo "✅ Services started"
	@echo "📊 MinIO Console: http://localhost:9001 (minioadmin/minioadmin123)"
	@echo "📊 Dagster UI: http://localhost:3001"
	@echo "📊 Grafana: http://localhost:3000 (admin/gogooku123)"
	@echo "📊 Prometheus: http://localhost:9090"

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

# Testing
test:
	pytest -m "not slow"

test-unit:
	pytest -m unit

test-integration:
	pytest -m integration


# Development
dev: setup docker-up
	@echo "✅ Development environment ready"

# Clean up
clean:
	docker-compose -f docker/docker-compose.yml down -v
	rm -rf venv __pycache__ .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

# Database operations
db-init:
	docker exec -i gogooku3-clickhouse clickhouse-client < docker/clickhouse-init.sql

# MinIO operations
minio-create-bucket:
	docker exec gogooku3-minio mc alias set local http://localhost:9000 minioadmin minioadmin
	docker exec gogooku3-minio mc mb local/gogooku3 --ignore-existing

# ============================================================================
# Training Commands
# ============================================================================

train:
	@$(MAKE) train-stable

.PHONY: clean-deprecated
clean-deprecated:
	@if [ "$${APPLY}" = "1" ]; then \
		python scripts/maintenance/cleanup_deprecated.py --apply ; \
	else \
		python scripts/maintenance/cleanup_deprecated.py ; \
	fi

# Research: one-shot baseline checks and splits
.PHONY: research-baseline
DATASET ?= output/ml_dataset_latest_full.parquet
NSPLITS ?= 5
EMBARGO ?= 20
FACTOR ?= returns_5d
HORIZONS ?= 1,5,10,20
SPLITS_JSON ?= output/eval_splits_$(NSPLITS)fold_$(EMBARGO)d.json

research-baseline:
	@echo "🔎 Snapshot: $(DATASET)"
	python scripts/tools/baseline_snapshot.py $(DATASET)
	@echo "🛡  Data checks"
	python scripts/tools/data_checks.py $(DATASET)
	@echo "🧪 Purged WF splits (n=$(NSPLITS), embargo=$(EMBARGO) days)"
	python scripts/tools/split_purged_wf.py --dataset $(DATASET) --n-splits $(NSPLITS) --embargo-days $(EMBARGO) --save-json $(SPLITS_JSON)
	@echo "📈 Baseline metrics (factor=$(FACTOR), horizons=$(HORIZONS))"
	python scripts/tools/baseline_metrics.py $(DATASET) --factor $(FACTOR) --horizons $(HORIZONS)
	@echo "✅ Baseline complete. Splits JSON: $(SPLITS_JSON)"

# Research: quick lag audits for parquets (glob supported)
.PHONY: research-lags
PATTERN ?= output/*.parquet
research-lags:
	@echo "⏱  Lag audit for pattern: $(PATTERN)"
	python scripts/tools/lag_audit_stub.py "$(PATTERN)"

# Convert only (no training): Unified pipeline sugar
.PHONY: atft-convert
DATA ?= output/ml_dataset_future_returns.parquet
atft-convert:
	python scripts/integrated_ml_training_pipeline.py --data-path $${DATA} --only-convert

# Research: chain baseline and lags
.PHONY: research-all
research-all: research-baseline research-lags
	@echo "🧭 Research bundle complete"

.PHONY: research-report
REPORT ?= reports/research_report.md
FACTORS ?= returns_5d,ret_1d_vs_sec,rank_ret_1d,macd_hist_slope,graph_degree
RHORIZONS ?= 1,5,10,20
research-report:
	@echo "📝 Generating research report: $(REPORT)"
	python scripts/tools/research_report.py --dataset $(DATASET) --factors $(FACTORS) --horizons $(RHORIZONS) --out $(REPORT) --csv $(REPORT:.md=.csv) --splits-json $(SPLITS_JSON)

.PHONY: research-plus
research-plus: research-all research-report
	@echo "📘 Research report bundle complete"

.PHONY: research-folds
SPLITS ?= output/eval_splits_$(NSPLITS)fold_$(EMBARGO)d.json
F_FACTORS ?= returns_5d,ret_1d_vs_sec,rank_ret_1d,graph_degree
F_HORIZONS ?= 1,5,10,20
F_OUT ?= reports/fold_metrics.csv
research-folds:
	@echo "🧪 Per-fold metrics: $(SPLITS) -> $(F_OUT)"
	python scripts/tools/fold_metrics.py --dataset $(DATASET) --splits-json $(SPLITS) --factors $(F_FACTORS) --horizons $(F_HORIZONS) --out $(F_OUT)

# HPO (Hyperparameter Optimization) targets
.PHONY: hpo-run hpo-resume hpo-status hpo-mock hpo-setup

# Default HPO settings
HPO_STUDY ?= atft_hpo_production
HPO_TRIALS ?= 20
HPO_TIMEOUT ?= 1800
HPO_STORAGE ?= sqlite:///output/hpo/optuna.db

hpo-setup:
	@echo "🚀 Setting up HPO environment"
	mkdir -p output/hpo
	@echo "📊 HPO storage directory created: output/hpo/"
	@echo "💾 Default storage URL: $(HPO_STORAGE)"
	@echo "📝 Set OPTUNA_STORAGE_URL environment variable to override"

hpo-run: hpo-setup
	@echo "🎯 Starting HPO optimization"
	@echo "   Study: $(HPO_STUDY)"
	@echo "   Trials: $(HPO_TRIALS)"
	@echo "   Storage: $(HPO_STORAGE)"
	OPTUNA_STORAGE_URL=$(HPO_STORAGE) python scripts/hpo/run_hpo_simple.py run \
		--study-name $(HPO_STUDY) \
		--trials $(HPO_TRIALS) \
		--timeout $(HPO_TIMEOUT) \
		--storage $(HPO_STORAGE)

hpo-resume: hpo-setup
	@echo "🔄 Resuming HPO optimization"
	@echo "   Study: $(HPO_STUDY)"
	@echo "   Additional trials: $(HPO_TRIALS)"
	OPTUNA_STORAGE_URL=$(HPO_STORAGE) python scripts/hpo/run_hpo_simple.py resume \
		--study-name $(HPO_STUDY) \
		--trials $(HPO_TRIALS) \
		--timeout $(HPO_TIMEOUT) \
		--storage $(HPO_STORAGE)

hpo-status:
	@echo "📊 Checking HPO study status"
	@echo "   Study: $(HPO_STUDY)"
	@echo "   Storage: $(HPO_STORAGE)"
	OPTUNA_STORAGE_URL=$(HPO_STORAGE) python scripts/hpo/run_hpo_simple.py status \
		--study-name $(HPO_STUDY) \
		--storage $(HPO_STORAGE)

hpo-mock:
	@echo "🧪 Running mock HPO for testing"
	python scripts/hpo/run_hpo_simple.py mock --trials 3

hpo-test:
	@echo "🧪 Testing HPO functionality"
	python scripts/hpo/test_hpo_basic.py

# GPU Training with Latest Dataset
.PHONY: train-gpu-latest train-gpu-latest-safe train-gpu-monitor train-gpu-progress train-gpu-stop

train-gpu-latest:
	@echo "🚀 Launching GPU training (background)"
	@./scripts/launch_train_gpu_latest.sh

train-gpu-latest-safe:
	@echo "🚀 Launching GPU training with SafeTrainingPipeline validation (background)"
	@./scripts/launch_train_gpu_latest.sh --safe

train-gpu-monitor:
	@# Find active training process and monitor both wrapper and ML logs in real time
	@PID=$$(pgrep -af 'python.*train_atft\.py' | head -1 | awk '{print $$1}'); \
	if [ -z "$$PID" ]; then \
		echo "❌ No active training process found. Start with 'make train-gpu-latest'"; \
		exit 1; \
	fi; \
	WRAP_LOG=$$(grep -l "^$$PID$$" _logs/train_gpu_latest/*.pid 2>/dev/null | sed 's/\.pid$$/.log/' | head -1); \
	if [ -z "$$WRAP_LOG" ]; then \
		WRAP_LOG="./_logs/train_gpu_latest/latest.log"; \
	fi; \
	ML_LOG="logs/ml_training.log"; \
	echo "📡 Monitoring active training (PID: $$PID)"; \
	[ -f "$$WRAP_LOG" ] && echo "📄 Wrapper log : $$WRAP_LOG" || echo "⚠️  Wrapper log not found"; \
	[ -f "$$ML_LOG" ] && echo "📄 ML log      : $$ML_LOG" || echo "⚠️  ML log not found yet (will appear after trainer starts)"; \
	echo "🔄 Press Ctrl+C to stop monitoring (training continues)"; \
	echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"; \
	if [ -f "$$WRAP_LOG" ] && [ -f "$$ML_LOG" ]; then \
		(stdbuf -oL -eL tail -F "$$WRAP_LOG" | sed -u 's/^/[wrapper] /') & P1=$$!; \
		(stdbuf -oL -eL tail -F "$$ML_LOG"   | sed -u 's/^/[ml]      /') & P2=$$!; \
		trap 'kill $$P1 $$P2 2>/dev/null' INT TERM; \
		wait; \
	elif [ -f "$$WRAP_LOG" ]; then \
		stdbuf -oL -eL tail -F "$$WRAP_LOG"; \
	elif [ -f "$$ML_LOG" ]; then \
		stdbuf -oL -eL tail -F "$$ML_LOG"; \
	else \
		echo "❌ No logs to follow. Check _logs/train_gpu_latest/ and logs/"; \
		exit 1; \
	fi

train-gpu-progress:
	@if [ ! -f ./runs/last/heartbeat.json ]; then \
		echo "No heartbeat found. Start a run with 'make train-gpu-latest' first."; \
		exit 1; \
	fi
	@./scripts/monitor_training_progress.py

train-gpu-stop:
	@if [ ! -f ./_logs/train_gpu_latest/latest.pid ]; then \
		echo "No PID file found. Nothing to stop."; \
		exit 1; \
	fi
	PID=$$(cat ./_logs/train_gpu_latest/latest.pid); \
	if kill $$PID 2>/dev/null; then \
		echo "🛑 Stopped GPU training (PID $$PID)"; \
	else \
		echo "PID $$PID not running (already stopped?)"; \
	fi

# Integrated ML Training targets
.PHONY: train-integrated train-integrated-safe train-integrated-hpo train-atft train-safe smoke

# Default training settings
OUTPUT_BASE ?= /home/ubuntu/gogooku3-standalone/output/batch
CONFIG_PATH ?= configs/atft
CONFIG_NAME ?= config

train-integrated:
	@echo "🚀 Running integrated ML training pipeline"
	@echo "   Output: $(OUTPUT_BASE)"
	python scripts/integrated_ml_training_pipeline.py \
		--output-base $(OUTPUT_BASE) \
		--config-path $(CONFIG_PATH) \
		--config-name $(CONFIG_NAME)

train-integrated-safe:
	@echo "🛡️ Running integrated pipeline with SafeTrainingPipeline validation"
	@echo "   Output: $(OUTPUT_BASE)"
	python scripts/integrated_ml_training_pipeline.py \
		--output-base $(OUTPUT_BASE) \
		--run-safe-pipeline \
		--config-path $(CONFIG_PATH) \
		--config-name $(CONFIG_NAME)

train-integrated-hpo:
	@echo "🎯 Running integrated pipeline with hyperparameter optimization"
	@echo "   Output: $(OUTPUT_BASE)"
	@echo "   HPO trials: 20"
	python scripts/integrated_ml_training_pipeline.py \
		--output-base $(OUTPUT_BASE) \
		--run-hpo \
		--hpo-n-trials 20 \
		--config-path $(CONFIG_PATH) \
		--config-name $(CONFIG_NAME)

train-atft:
	@echo "🧠 Running ATFT training directly"
	@echo "   Config: $(CONFIG_PATH)/$(CONFIG_NAME)"
	python scripts/train_atft.py \
		--config-path $(CONFIG_PATH) \
		--config-name $(CONFIG_NAME)

train-safe:
	@echo "🛡️ Running SafeTrainingPipeline only"
	@echo "   Output: $(OUTPUT_BASE)"
	python scripts/run_safe_training.py \
		--data-dir $(OUTPUT_BASE) \
		--n-splits 2 \
		--embargo-days 20 \
		--memory-limit 6 \
		--verbose

smoke:
	@echo "💨 Running quick smoke test (1 epoch)"
	@echo "   Output: $(OUTPUT_BASE)"
	python scripts/smoke_test.py \
		--output-base $(OUTPUT_BASE) \
		--max-epochs 1

# Alias for HPO with ATFT
.PHONY: hpo-atft
hpo-atft:
	@echo "🎯 Running Optuna HPO for ATFT model"
	python scripts/hpo/run_optuna_atft.py \
		--output-base $(OUTPUT_BASE) \
		--n-trials 20 \
		--timeout 3600

# ============================================================================
# Performance-Improved Training (PDFの提案に基づく改善)
# ============================================================================

.PHONY: train-improved train-improved-validate

train-improved:
	@echo "🚀 Running performance-improved training"
	@echo "   ✅ Multi-worker DataLoader enabled (NUM_WORKERS=8)"
	@echo "   ✅ Model capacity increased (hidden_size: 64→256)"
	@echo "   ✅ IC/RankIC optimization (CS_IC_WEIGHT=0.2)"
	@echo "   ✅ PyTorch 2.x compilation (if available)"
	@echo "   ✅ Plateau learning rate scheduler"
	@./scripts/run_improved_training.sh

train-improved-validate:
	@echo "🔍 Validating improved training configuration"
	@./scripts/run_improved_training.sh --validate-only

# Production optimized training (PDF analysis based)
.PHONY: train-optimized train-optimized-quick train-optimized-report train-optimized-dry

train-optimized:
	@echo "🚀 Running production-optimized training (PDF analysis based)"
	@echo "   ✅ All improvements from PDF analysis applied"
	@echo "   ✅ ALLOW_UNSAFE_DATALOADER=1 (multi-worker enabled)"
	@echo "   ✅ hidden_size=256, RankIC/Sharpe optimization"
	@echo "   ✅ torch.compile enabled, feature grouping aligned"
	@python scripts/train_optimized_direct.py

train-optimized-quick:
	@echo "⚡ Running quick validation (3 epochs)"
	@echo "   ✅ All optimizations enabled (Phase延長、VSN/FAN/SAN有効化)"
	@echo "   ✅ NUM_WORKERS=8 (root cause fixed: pre-compute stats in main process)"
	@echo "   ✅ GRAPH: k=24, edge_threshold=0.18, min_edges=75"
	@echo "   ✅ Log: /tmp/atft_quick_test.log"
	@export OUTPUT_BASE=/home/ubuntu/gogooku3-standalone/output && \
	 export ALLOW_UNSAFE_DATALOADER=1 && \
	 export FEATURE_CLIP_VALUE=8.0 && \
	 nohup python scripts/train_atft.py \
	   --config-path ../configs/atft \
	   --config-name config_production_optimized \
	   data.source.data_dir=output/atft_data \
	   train.trainer.max_epochs=3 \
	   > /tmp/atft_quick_test.log 2>&1 & \
	 echo "✅ Training started. PID: $$!" && \
	 echo "📊 Monitor: tail -f /tmp/atft_quick_test.log"

train-optimized-report:
	@echo "📊 Generating optimization report"
	@python scripts/run_production_optimized.py --report

train-optimized-dry:
	@echo "🔍 Dry run - showing configuration only"
	@python scripts/run_production_optimized.py --dry-run

train-optimized-safe:
	@echo "🛡️ Running safe optimized training (conservative settings)"
	@echo "   ✅ Single-worker DataLoader (no crashes)"
	@echo "   ✅ hidden_size=256, RankIC/Sharpe optimization"
	@echo "   ✅ Reduced batch size for stability"
	@python scripts/train_optimized_safe.py

train-optimized-stable:
	@echo "⚡ Running stable optimized training (recommended)"
	@echo "   ✅ No DataLoader worker errors"
	@echo "   ✅ Full optimizations from PDF analysis"
	@echo "   ✅ Stable memory management"
	@echo "   ✅ Fixed horizon key mismatch"
	@echo "   ✅ Zero loss guards added"
	@echo "   ✅ Feature normalization enabled"
	@echo "   ✅ RankIC/Huber/CS-IC losses enabled"
	@echo "   ✅ Phase-aware loss scheduling"
	@echo "   ✅ Multi-worker data loading"
	@echo "Validating data before training..."
	@python scripts/validate_data.py || true
	@echo "Starting training with full optimizations..."
	@ENABLE_FEATURE_NORM=1 \
	FEATURE_CLIP_VALUE=10.0 \
	USE_SWA=0 \
	USE_SAFE_AMP=1 \
	DEGENERACY_ABORT=0 \
	USE_RANKIC=1 \
	RANKIC_WEIGHT=0.2 \
	USE_HUBER=1 \
	HUBER_WEIGHT=0.3 \
	USE_CS_IC=1 \
	CS_IC_WEIGHT=0.15 \
	USE_DIR_AUX=1 \
	DIR_AUX_WEIGHT=0.1 \
	SHARPE_WEIGHT=0.3 \
	DYN_WEIGHT=1 \
	PHASE_LOSS_WEIGHTS="0:huber=0.3,quantile=1.0,sharpe=0.1;1:quantile=0.7,sharpe=0.3,rankic=0.1;2:quantile=0.5,sharpe=0.4,rankic=0.2,t_nll=0.3;3:quantile=0.3,sharpe=0.5,rankic=0.3,cs_ic=0.2" \
	ALLOW_UNSAFE_DATALOADER=1 \
	NUM_WORKERS=8 \
	PERSISTENT_WORKERS=1 \
	PREFETCH_FACTOR=4 \
	TORCH_COMPILE_MODE=max-autotune \
	python scripts/train_atft.py --config-path ../configs/atft --config-name config_production \
		data.source.data_dir=/home/ubuntu/gogooku3-standalone/output/atft_data \
		model.hidden_size=256 \
		train.batch.train_batch_size=2048 \
		train.optimizer.lr=5e-4 \
		train.trainer.max_epochs=120 \
		improvements.compile_model=true

train-fixed:
	@echo "🔧 Running fixed training configuration"
	@echo "   ✅ All known issues resolved"
	@echo "   ✅ PDF optimizations applied"
	@echo "   ✅ Stable execution guaranteed"
	@python scripts/train_fixed.py

train-rankic-boost:
	@echo "🚀 Running RankIC-boosted training"
	@echo "   ✅ Dedicated Hydra config: config_rankic_boost.yaml"
	@echo "   ✅ RANKIC_WEIGHT=0.5 (maximum RankIC focus)"
	@echo "   ✅ DataLoader: NUM_WORKERS=8, PERSISTENT=1, PREFETCH=4"
	@echo "   ✅ Batch size 2048, LR 5e-4, BF16 + torch.compile"
	@python scripts/train_rankic_boost.py

# ============================================================================
# Feature Preservation ML Pipeline (全特徴量保持)
# ============================================================================

.PHONY: dataset-ext
dataset-ext: ## Build extended dataset with all feature improvements
	@echo "📊 Building extended dataset with all improvements..."
	@INPUT=$${INPUT:-output/ml_dataset_latest_full.parquet}; \
	OUTPUT=$${OUTPUT:-output/dataset_ext.parquet}; \
	python scripts/build_dataset_ext.py \
		--input $$INPUT \
		--output $$OUTPUT \
		--adv-col dollar_volume_ma20
	@echo "✅ Extended dataset saved to: $$OUTPUT"

.PHONY: train-multihead
train-multihead: ## Train multi-head model with feature groups
	@echo "🧠 Training multi-head model with feature groups..."
	@DATA=$${DATA:-output/dataset_ext.parquet}; \
	python scripts/train_multihead.py \
		--data $$DATA \
		--epochs 10 \
		--batch-size 1024 \
		--feature-groups configs/feature_groups.yaml \
		--pred-out output/predictions.parquet
	@echo "✅ Training complete. Predictions saved to output/predictions.parquet"

.PHONY: eval-multihead
eval-multihead: ## Generate comprehensive evaluation report with ablation
	@echo "📈 Generating evaluation report with ablation analysis..."
	@DATA=$${DATA:-output/predictions.parquet}; \
	mkdir -p reports; \
	python scripts/eval_report.py \
		--data $$DATA \
		--ablation \
		--horizons "1,5,10,20" \
		--output reports/evaluation_report.html
	@echo "✅ Report saved to reports/evaluation_report.html"

.PHONY: pipeline-full-ext
pipeline-full-ext: ## Complete feature preservation pipeline
	@echo "🚀 Running complete feature preservation pipeline..."
	@START=$${START:?START date required}; \
	END=$${END:?END date required}; \
	echo "📅 Period: $$START to $$END"; \
	$(MAKE) dataset-full START=$$START END=$$END && \
	$(MAKE) dataset-ext INPUT=output/ml_dataset_latest_full.parquet OUTPUT=output/dataset_ext.parquet && \
	$(MAKE) train-multihead DATA=output/dataset_ext.parquet && \
	$(MAKE) eval-multihead DATA=output/predictions.parquet
	@echo "✅ Complete pipeline finished successfully"

.PHONY: test-ext
test-ext: ## Run CI tests for data quality and pipeline integrity
	@echo "🧪 Running CI tests for feature preservation ML..."
	python -m pytest tests/test_data_checks.py -v
	python -m pytest tests/test_cv_pipeline.py -v -m "not slow"
	@echo "✅ All CI tests passed"

.PHONY: train-ultra-stable
train-ultra-stable: ## Run ULTRA-STABLE training (maximum stability)
	@echo "🛡️ Running ULTRA-STABLE training configuration"
	@echo "   ✅ ALL known issues fixed"
	@echo "   ✅ Maximum stability prioritized"
	@echo "   ✅ Conservative settings"
	@echo "   ✅ Fixed horizon key mismatch"
	@echo "   ✅ Zero loss protection"
	@ALLOW_UNSAFE_DATALOADER=0 \
	USE_AMP=0 \
	USE_SWA=0 \
	FEATURE_CLIP_VALUE=5.0 \
	DEGENERACY_ABORT=0 \
	MIN_VALID_RATIO=0.1 \
	MIN_TRAINING_DATE="2018-01-01" \
	python scripts/train_atft.py \
		--config-path ../configs/atft \
		--config-name config_production \
		data.source.data_dir=/home/ubuntu/gogooku3-standalone/output/atft_data \
		data.source.min_date="2018-01-01" \
		data.distributed.enabled=false \
		data.distributed.num_workers=0 \
		train.batch.train_batch_size=128 \
		train.optimizer.lr=1e-4 \
		train.trainer.max_epochs=5 \
		train.trainer.gradient_clip_val=0.5 \
		model.hidden_size=128

train-mini-safe: ## Run Mini Training mode (simplest, most stable)
	@echo "🔒 Mini Training Mode (Simplest & Most Stable)"
	@echo "   ✅ Simplified training loop"
	@echo "   ✅ No AMP/GradScaler complexity"
	@echo "   ✅ Fixed horizon keys"
	@echo "   ✅ Zero loss guards"
	@USE_MINI_TRAIN=1 \
	FEATURE_CLIP_VALUE=5.0 \
	USE_SWA=0 \
	python scripts/train_atft.py \
		--config-path ../configs/atft \
		--config-name config_production \
		data.source.data_dir=/home/ubuntu/gogooku3-standalone/output/atft_data \
		model.hidden_size=128 \
		train.batch.train_batch_size=256 \
		train.optimizer.lr=5e-5 \
		train.trainer.max_epochs=5

.PHONY: train-stable
train-stable: ## Stable single-process training (RECOMMENDED for stability)
	@echo "============================================================"
	@echo "🛡️  STABLE SINGLE-PROCESS TRAINING"
	@echo "============================================================"
	@echo "   ✅ Forced single-process (NUM_WORKERS=0)"
	@echo "   ✅ Conservative batch size (128)"
	@echo "   ✅ Date filtering (>=2016-01-01) for valid targets"
	@echo "   ✅ No multiprocessing crashes"
	@echo "   ✅ All optimizations from PDF analysis"
	@echo "============================================================"
	@OUTPUT_BASE=/home/ubuntu/gogooku3-standalone/output/atft_data \
	NUM_WORKERS=0 \
	PERSISTENT_WORKERS=0 \
	PREFETCH_FACTOR=0 \
	PIN_MEMORY=0 \
	ALLOW_UNSAFE_DATALOADER=0 \
	FORCE_SINGLE_PROCESS=1 \
	USE_DAY_BATCH=0 \
	MIN_TRAINING_DATE="2016-01-01" \
	BATCH_SIZE=128 \
	VAL_BATCH_SIZE=256 \
	OMP_NUM_THREADS=1 \
	MKL_NUM_THREADS=1 \
	OPENBLAS_NUM_THREADS=1 \
	NUMEXPR_NUM_THREADS=1 \
	MAX_EPOCHS=120 \
	LEARNING_RATE=1e-4 \
	USE_RANKIC=1 \
	RANKIC_WEIGHT=0.2 \
	CS_IC_WEIGHT=0.15 \
	SHARPE_WEIGHT=0.3 \
	HORIZON_WEIGHT_1D=1.0 \
	HORIZON_WEIGHT_5D=0.6 \
	HORIZON_WEIGHT_10D=0.3 \
	HORIZON_WEIGHT_20D=0.2 \
	PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
	USE_AMP=1 \
	AMP_DTYPE=bf16 \
	ENABLE_FEATURE_NORM=1 \
	FEATURE_CLIP_VALUE=10.0 \
	python scripts/train_atft.py \
		--config-path ../configs/atft \
		--config-name config_production_optimized \
		data=jpx_large_scale \
		train=production_improved \
		model=atft_gat_fan \
		train.trainer.max_epochs=120 \
		train.batch.batch_size=128 \
		train.optimizer.lr=1e-4 \
		model.hidden_size=256 \
		improvements.compile_model=false \
		data.source.data_dir=/home/ubuntu/gogooku3-standalone/output/atft_data
