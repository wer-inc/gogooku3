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
	@echo "GPU Training with Latest Dataset:"
	@echo "make train-gpu-latest                 - GPU training with auto-detected latest dataset"
	@echo "make train-gpu-latest-safe            - GPU training with SafeTrainingPipeline validation"

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
.PHONY: train-gpu-latest train-gpu-latest-safe

train-gpu-latest:
	@echo "🚀 Running GPU training with latest dataset"
	@./scripts/train_gpu_latest.sh

train-gpu-latest-safe:
	@echo "🚀 Running GPU training with SafeTrainingPipeline validation"
	@./scripts/train_gpu_latest.sh --safe

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
