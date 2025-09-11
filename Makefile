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
	@echo "make train                            - Train ATFT-GAT-FAN model"
	@echo "make fetch-all    START=YYYY-MM-DD END=YYYY-MM-DD - Fetch prices/topix/trades_spec/statements"
	@echo "make clean-deprecated                 - Remove deprecated shim scripts (use --apply via VAR APPLY=1)"

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

# Model training
.PHONY: train
train:
	python scripts/integrated_ml_training_pipeline.py

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
