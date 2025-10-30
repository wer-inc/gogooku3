.PHONY: help build build-optimized test lint format clean

START ?= $(shell date -u -d '30 days ago' +%F)
END ?= $(shell date -u +%F)
REFRESH_LISTED ?=
CACHE_ONLY ?=

REFRESH_FLAG :=
ifeq ($(REFRESH_LISTED),1)
REFRESH_FLAG := --refresh-listed
endif

CACHE_ONLY_FLAG :=
ifeq ($(CACHE_ONLY),1)
CACHE_ONLY_FLAG := --cache-only
endif

help:
	@echo "gogooku5 data package"
	@echo "  make build             -> default dataset pipeline (uses START/END vars)"
	@echo "  make build-optimized   -> optimized dataset pipeline"
	@echo "  make test              -> run unit + integration tests"
	@echo "  make lint              -> run ruff + mypy"

build:
	@python scripts/build.py --start $(START) --end $(END) $(REFRESH_FLAG)

build-optimized:
	@python scripts/build_optimized.py --start $(START) --end $(END) $(CACHE_ONLY_FLAG)

lint:
	@echo "[data] running ruff + mypy"
	@python -m ruff check src || true
	@python -m mypy src || true

format:
	@python -m ruff format src

clean:
	@rm -rf output/cache/* output/raw/*

clean-logs:
	@rm -rf logs/*

clean-tests:
	@rm -rf .pytest_cache htmlcov

clean-all: clean clean-logs clean-tests

clean-cache:
	@echo "Deprecated alias. Use make clean instead."

check-env:
	@test -f .env || (echo ".env missing. Copy from .env.example" && exit 1)
