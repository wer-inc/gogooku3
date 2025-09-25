.PHONY: help setup test run clean docker-up docker-down

help:
	@echo "gogooku3 batch processing"
	@echo "========================"
	@echo "make setup        - Setup Python environment and dependencies"
	@echo "make docker-up    - Start all services (MinIO, ClickHouse, etc.)"
	@echo "make docker-down  - Stop all services"
	@echo "make test         - Run tests"
	@echo "make run          - Start Dagster UI"
	@echo "make clean        - Clean up environment"
	@echo "make dataset-full START=YYYY-MM-DD END=YYYY-MM-DD - Build full enriched dataset"
	@echo "make dataset-full-gpu START=YYYY-MM-DD END=YYYY-MM-DD - Build dataset with GPU-ETL enabled"
	@echo "make dataset-full-prod START=YYYY-MM-DD END=YYYY-MM-DD - Build using configs/pipeline/full_dataset.yaml"
	@echo "make dataset-full-research START=YYYY-MM-DD END=YYYY-MM-DD - Build using configs/pipeline/research_full_indices.yaml"
	@echo "make check-indices  DATASET=output/ml_dataset_latest_full.parquet - Validate indices features"
	@echo "make fetch-all    START=YYYY-MM-DD END=YYYY-MM-DD - Fetch prices/topix/trades_spec/statements"
	@echo "make clean-deprecated                 - Remove deprecated shim scripts (use --apply via VAR APPLY=1)"
	@echo "make research-baseline                - Run snapshot, checks, splits, baseline metrics"
	@echo "make research-lags PATTERN=glob      - Audit Date→effective and publish→effective lags"
	@echo "make research-all                     - Run baseline + lag audit together"
	@echo "make research-report                  - Generate Markdown research report"
	@echo "make research-plus                    - research-all + research-report"
	@echo "make research-folds                  - Per-fold RankIC/HitRate using splits JSON"
	@echo ""
	@echo "HPO (Hyperparameter Optimization):"
	@echo "make hpo-setup                       - Setup HPO environment"
	@echo "make hpo-run                         - Start new HPO optimization"
	@echo "make hpo-resume                      - Resume existing HPO study"
	@echo "make hpo-status                      - Check HPO study status"
	@echo "make hpo-mock                        - Run mock HPO for testing"
	@echo "make hpo-test                        - Test HPO functionality"
	@echo ""
	@echo "Integrated ML Training:"
	@echo "make train-integrated                 - Run full integrated ML training pipeline"
	@echo "make train-integrated-safe            - Run with SafeTrainingPipeline validation first"
	@echo "make train-integrated-hpo             - Run with hyperparameter optimization"
	@echo "make train-atft                       - Run ATFT training directly"
	@echo "make train-safe                       - Run SafeTrainingPipeline only"
	@echo "make smoke                            - Quick 1-epoch smoke test"
	@echo ""
	@echo "Performance-Improved Training:"
	@echo "make train-improved                   - Run with all performance optimizations"
	@echo "make train-improved-validate          - Validate improved configuration only"
	@echo ""
	@echo "Production Optimized (PDF Analysis Based):"
	@echo "make train-optimized                  - Run with all PDF-recommended optimizations"
	@echo "make train-optimized-report           - Show optimization report"
	@echo "make train-optimized-dry              - Dry run to check configuration"
	@echo ""
	@echo "GPU Training with Latest Dataset:"
	@echo "make train-gpu-latest                 - GPU training with auto-detected latest dataset"
	@echo "make train-gpu-latest-safe            - GPU training with SafeTrainingPipeline validation"
	@echo "make train-gpu-monitor                - Tail latest GPU training log"
	@echo "make train-gpu-progress               - Show summarized training heartbeat"
	@echo "make train-gpu-stop                   - Stop latest GPU training run"
	@echo ""
	@echo "Feature Preservation ML Pipeline (全特徴量保持):"
	@echo "make dataset-ext INPUT=output/ml_dataset_latest_full.parquet - Build extended dataset with all improvements"
	@echo "make train-multihead DATA=output/dataset_ext.parquet        - Train multi-head model with feature groups"
	@echo "make eval-multihead DATA=output/predictions.parquet          - Generate evaluation report with ablation"
	@echo "make pipeline-full-ext START=YYYY-MM-DD END=YYYY-MM-DD      - Complete feature preservation pipeline"
	@echo "make test-ext                                                - Run CI tests for data quality checks"

# Python environment setup
setup:
	python3 -m venv venv
	./venv/bin/pip install --upgrade pip
	./venv/bin/pip install -r requirements.txt
	@echo "✅ Python environment ready"
	@echo "📝 Copy .env.example to .env and configure your settings"
	cp -n .env.example .env || true

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
	./venv/bin/pytest batch/tests/ -v --cov=batch

test-unit:
	./venv/bin/pytest batch/tests/unit/ -v

test-integration:
	./venv/bin/pytest batch/tests/integration/ -v

# Run Dagster
run:
	./venv/bin/dagster dev -f batch/dagster/repository.py

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

# Unified dataset build (full pipeline)
.PHONY: dataset-full
dataset-full:
	python scripts/pipelines/run_full_dataset.py --jquants --start-date $${START} --end-date $${END}

# GPU-ETL acceleration enabled dataset generation
.PHONY: dataset-full-gpu
dataset-full-gpu:
	@echo "🚀 Running dataset generation with GPU-ETL enabled"
	@export REQUIRE_GPU=1 USE_GPU_ETL=1 RMM_POOL_SIZE=70GB CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src && \
	python scripts/pipelines/run_full_dataset.py --jquants --start-date $${START} --end-date $${END} --gpu-etl

.PHONY: dataset-full-gpu-bg
dataset-full-gpu-bg:
	@if [ -z "$$START" ] || [ -z "$$END" ]; then \
	  echo "Usage: make dataset-full-gpu-bg START=YYYY-MM-DD END=YYYY-MM-DD"; \
	  exit 1; \
	fi
	@mkdir -p _logs/background
	@ts=$$(date +%Y%m%d_%H%M%S); \
	log=_logs/background/dataset_full_gpu_$$ts.log; \
	echo "🚀 Launching dataset-full-gpu in background (log: $$log)"; \
	nohup bash -lc "START=$$START END=$$END $(MAKE) dataset-full-gpu" > $$log 2>&1 &

.PHONY: dataset-full-prod
dataset-full-prod:
	python scripts/pipelines/run_full_dataset.py --jquants --start-date $${START} --end-date $${END} --config configs/pipeline/full_dataset.yaml

.PHONY: dataset-full-research
dataset-full-research:
	python scripts/pipelines/run_full_dataset.py --jquants --start-date $${START} --end-date $${END} --config configs/pipeline/research_full_indices.yaml

.PHONY: check-indices
check-indices:
	python scripts/tools/check_indices_features.py --dataset $(DATASET)

# Fetch all raw components in one shot
.PHONY: fetch-all
fetch-all:
	python scripts/data/fetch_jquants_history.py --jquants --all --start-date $${START} --end-date $${END}

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

# Research: chain baseline and lags
.PHONY: research-all
research-all: research-baseline research-lags
	@echo "🧭 Research bundle complete"

.PHONY: research-report
REPORT ?= reports/research_report.md
FACTORS ?= returns_5d,ret_1d_vs_sec,rank_ret_1d,macd_hist_slope,graph_degree
RHORIZONS ?= 1,5,10,20
# Defaults for splits reference (reuse baseline defaults)
NSPLITS ?= 5
EMBARGO ?= 20
SPLITS_JSON ?= output/eval_splits_$(NSPLITS)fold_$(EMBARGO)d.json
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
.PHONY: train-optimized train-optimized-report train-optimized-dry

train-optimized:
	@echo "🚀 Running production-optimized training (PDF analysis based)"
	@echo "   ✅ All improvements from PDF analysis applied"
	@echo "   ✅ ALLOW_UNSAFE_DATALOADER=1 (multi-worker enabled)"
	@echo "   ✅ hidden_size=256, RankIC/Sharpe optimization"
	@echo "   ✅ torch.compile enabled, feature grouping aligned"
	@python scripts/train_optimized_direct.py

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
	@python scripts/train_atft.py --config-path $(CONFIG_PATH) --config-name config_production_stable

train-fixed:
	@echo "🔧 Running fixed training configuration"
	@echo "   ✅ All known issues resolved"
	@echo "   ✅ PDF optimizations applied"
	@echo "   ✅ Stable execution guaranteed"
	@python scripts/train_fixed.py

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
	@echo "   ✅ 5-epoch test run"
	@python scripts/train_ultra_stable.py
