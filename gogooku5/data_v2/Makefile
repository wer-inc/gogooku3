ifneq (,$(wildcard .env))
include .env
# Export simple KEY=VAL pairs (ignore comments/blank lines) for CLI tools
export $(shell awk -F= '/^[A-Za-z_][A-Za-z0-9_]*=/{print $$1}' .env)
endif

PYTHON ?= python
# When running with `make -C data_v2`, keep paths relative to this directory.
DUCKDB ?= output/jquants.duckdb
FROM_DATE ?= 2015-01-01
TO_DATE ?= 2025-12-31

.PHONY: help dagster-dev dagster-job-duckdb-calendar dagster-job-duckdb-listed \
	dagster-job-duckdb-daily-quotes dagster-job-duckdb-breakdown \
	dagster-job-duckdb-features \
	duckdb-fetch-listed duckdb-fetch-yf duckdb-fetch-daily-quotes duckdb-fetch-breakdown \
	duckdb-compute-features duckdb-compute-listed-features \
	duckdb-init duckdb-load-calendar get-refresh-token

help:
	@echo "data_v2 utilities"
	@echo "  make get-refresh-token           -> fetch J-Quants refreshToken (email/password)"
	@echo "  make duckdb-fetch-listed         -> fetch /listed/info and upsert into DuckDB (API直)"
	@echo "  make duckdb-fetch-yf             -> fetch yfinance macro set and upsert into DuckDB"
	@echo "  make duckdb-fetch-daily-quotes   -> fetch /prices/daily_quotes and upsert into DuckDB"
	@echo "  make dagster-dev                 -> run dagster dev (requires dagster installed)"
	@echo "  make dagster-job-duckdb-calendar -> execute dagster job: duckdb_calendar_job"
	@echo "  make dagster-job-duckdb-listed   -> execute dagster job: duckdb_listed_job"
	@echo "  make dagster-job-duckdb-daily-quotes -> execute dagster job: duckdb_daily_quotes_job"
	@echo "  make dagster-job-duckdb-breakdown -> execute dagster job: duckdb_breakdown_job"
	@echo "  make dagster-job-duckdb-features -> execute dagster job: duckdb_features_job (uses .env START/END)"
	@echo "  make duckdb-init                 -> create DuckDB file and tables (default: $(DUCKDB))"
	@echo "  make duckdb-load-calendar        -> fetch calendar via API and upsert into DuckDB (no Parquet)"
	@echo "  make duckdb-compute-features        -> compute price_flow_features from DuckDB (uses .env START/END)"
	@echo "  make duckdb-compute-listed-features -> add sector/scale features to listed_meta_features (after compute-features)"
	@echo "  make duckdb-compute-financial-features -> add financial statement features to financial_features (after compute-features)"

get-refresh-token:
	@PYTHONPATH=.. $(PYTHON) scripts/get_refresh_token.py

dagster-dev:
	@PYTHONPATH=.. dagster dev -m data_v2.dagster_project

dagster-job-duckdb-calendar:
	@tmpfile=$$(mktemp); \
	echo '{"ops":{"duckdb_upsert_calendar_op":{"inputs":{"from_date":"$(or $(FROM_DATE),2015-01-01)","to_date":"$(or $(TO_DATE),2025-12-31)","force_refresh":false,"db_path":"$(DUCKDB)","threads":0}}}}' > $$tmpfile; \
	PYTHONPATH=.. dagster job execute -m data_v2.dagster_project -j duckdb_calendar_job -c $$tmpfile; \
	rm -f $$tmpfile

dagster-job-duckdb-listed:
	@tmpfile=$$(mktemp); \
	echo '{"ops":{"duckdb_fetch_listed_op":{"inputs":{"start":"$(or $(START),2015-01-01)","end":"$(or $(END),$(shell date +%F))","db_path":"$(DUCKDB)","holiday_division":"$(or $(HOLIDAY_DIVISION),1,2)","sleep_sec":$(or $(SLEEP_SEC),0.0),"threads":0,"auto_fetch_calendar":true,"workers":$(or $(WORKERS),10)}}}}' > $$tmpfile; \
	PYTHONPATH=.. dagster job execute -m data_v2.dagster_project -j duckdb_listed_job -c $$tmpfile; \
	rm -f $$tmpfile

dagster-job-duckdb-daily-quotes:
	@tmpfile=$$(mktemp); \
	echo '{"ops":{"duckdb_fetch_daily_quotes_op":{"inputs":{"start":"$(or $(START),2015-01-01)","end":"$(or $(END),$(shell date +%F))","db_path":"$(DUCKDB)","holiday_division":"$(or $(HOLIDAY_DIVISION),1,2)","sleep_sec":$(or $(SLEEP_SEC),0.0),"threads":0,"auto_fetch_calendar":true,"workers":$(or $(WORKERS),10)}}}}' > $$tmpfile; \
	PYTHONPATH=.. dagster job execute -m data_v2.dagster_project -j duckdb_daily_quotes_job -c $$tmpfile; \
	rm -f $$tmpfile
dagster-job-duckdb-breakdown:
	@tmpfile=$$(mktemp); \
	echo '{"ops":{"duckdb_fetch_breakdown_op":{"inputs":{"start":"$(or $(START),2015-01-01)","end":"$(or $(END),$(shell date +%F))","db_path":"$(DUCKDB)","holiday_division":"$(or $(HOLIDAY_DIVISION),1,2)","sleep_sec":$(or $(SLEEP_SEC),0.0),"threads":0,"auto_fetch_calendar":true,"workers":$(or $(WORKERS),10)}}}}' > $$tmpfile; \
	PYTHONPATH=.. dagster job execute -m data_v2.dagster_project -j duckdb_breakdown_job -c $$tmpfile; \
	rm -f $$tmpfile
dagster-job-duckdb-features:
	@tmpfile=$$(mktemp); \
	echo '{"ops":{"duckdb_compute_features_op":{"inputs":{"db_path":"$(DUCKDB)","threads":0}}}}' > $$tmpfile; \
	PYTHONPATH=.. dagster job execute -m data_v2.dagster_project -j duckdb_features_job -c $$tmpfile; \
	rm -f $$tmpfile

duckdb-fetch-listed:
	@PYTHONPATH=.. $(PYTHON) scripts/duckdb_loader.py \
		--db $(DUCKDB) \
		fetch-listed-direct \
		--start $(or $(START),2015-01-01) \
		--end $(or $(END),$(shell date +%F)) \
		$(if $(HOLIDAY_DIVISION),--holiday-division $(HOLIDAY_DIVISION),) \
		$(if $(SLEEP_SEC),--sleep-sec $(SLEEP_SEC),) \
		$(if $(WORKERS),--workers $(WORKERS),) \
		--auto-fetch-calendar

duckdb-fetch-yf:
	@PYTHONPATH=.. $(PYTHON) scripts/duckdb_loader.py \
		--db $(DUCKDB) \
		fetch-yf-history \
		--start $(or $(START),2015-01-01) \
		--end $(or $(END),$(shell date +%F)) \
		$(if $(SLEEP_SEC),--sleep-sec $(SLEEP_SEC),)

duckdb-fetch-daily-quotes:
	@PYTHONPATH=.. $(PYTHON) scripts/duckdb_loader.py \
		--db $(DUCKDB) \
		fetch-daily-quotes \
		--start $(or $(START),2015-01-01) \
		--end $(or $(END),$(shell date +%F)) \
		$(if $(HOLIDAY_DIVISION),--holiday-division $(HOLIDAY_DIVISION),) \
		$(if $(SLEEP_SEC),--sleep-sec $(SLEEP_SEC),) \
		$(if $(WORKERS),--workers $(WORKERS),) \
		--auto-fetch-calendar

duckdb-fetch-breakdown:
	@PYTHONPATH=.. $(PYTHON) scripts/duckdb_loader.py \
		--db $(DUCKDB) \
		fetch-breakdown \
		--start $(or $(START),2015-01-01) \
		--end $(or $(END),$(shell date +%F)) \
		$(if $(HOLIDAY_DIVISION),--holiday-division $(HOLIDAY_DIVISION),) \
		$(if $(SLEEP_SEC),--sleep-sec $(SLEEP_SEC),) \
		$(if $(WORKERS),--workers $(WORKERS),) \
		--auto-fetch-calendar

duckdb-compute-features:
	@PYTHONPATH=.. $(PYTHON) scripts/duckdb_loader.py \
		--db $(DUCKDB) \
		compute-features

duckdb-compute-listed-features:
	@PYTHONPATH=.. $(PYTHON) scripts/duckdb_loader.py \
		--db $(DUCKDB) \
		compute-listed-features

duckdb-init:
	@PYTHONPATH=.. $(PYTHON) scripts/duckdb_loader.py --db $(DUCKDB) init

duckdb-load-calendar:
	@PYTHONPATH=.. $(PYTHON) scripts/duckdb_loader.py \
		--db $(DUCKDB) \
		fetch-calendar-direct \
		--from $(FROM_DATE) \
		--to $(TO_DATE) \
		$(if $(HOLIDAY_DIVISION),--holiday-division $(HOLIDAY_DIVISION),)
