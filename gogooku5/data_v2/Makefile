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
	dagster-job-duckdb-features dagster-job-duckdb-wmi dagster-job-duckdb-ss dagster-job-duckdb-ssp \
	duckdb-fetch-listed duckdb-fetch-yf duckdb-fetch-daily-quotes duckdb-fetch-breakdown \
	duckdb-fetch-weekly-margin-interest duckdb-fetch-short-selling duckdb-fetch-short-selling-positions \
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
	@echo "  make dagster-job-duckdb-wmi -> execute dagster job: duckdb_weekly_margin_interest_job"
	@echo "  make dagster-job-duckdb-ss -> execute dagster job: duckdb_short_selling_job"
	@echo "  make dagster-job-duckdb-ssp -> execute dagster job: duckdb_short_selling_positions_job"
	@echo "  make dagster-job-duckdb-features -> execute dagster job: duckdb_features_job (uses .env START/END)"
	@echo "  make duckdb-init                 -> create DuckDB file and tables (default: $(DUCKDB))"
	@echo "  make duckdb-load-calendar        -> fetch calendar via API and upsert into DuckDB (no Parquet)"
	@echo "  make duckdb-compute-features        -> compute price_flow_features from DuckDB (uses .env START/END)"
	@echo "  make duckdb-compute-listed-features -> add sector/scale features to listed_meta_features (after compute-features)"
	@echo "  make duckdb-fetch-weekly-margin-interest -> fetch /markets/weekly_margin_interest into DuckDB"
	@echo "  make duckdb-fetch-short-selling -> fetch /markets/short_selling into DuckDB"
	@echo "  make duckdb-fetch-short-sellin-positions -> fetch /markets/short_selling_positions into DuckDB"
	@echo "  make duckdb-compute-financial-features -> add financial statement features to financial_features (after compute-features)"

get-refresh-token:
	@PYTHONPATH=.. $(PYTHON) scripts/get_refresh_token.py

dagster-dev:
	@PYTHONPATH=.. dagster dev -m data_v2.dagster_project

dagster-job-duckdb-calendar:
	@tmpfile=$$(mktemp); \
	echo '{"ops":{"duckdb_upsert_calendar_op":{"inputs":{"from_date":"","to_date":"","force_refresh":false,"db_path":"$(DUCKDB)","threads":0}}}}' > $$tmpfile; \
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

dagster-job-duckdb-price-flow-features:
	@tmpfile=$$(mktemp); \
	echo '{"ops":{"duckdb_build_price_flow_features_op":{"inputs":{"start":"$(or $(START),2015-01-01)","end":"$(or $(END),$(shell date +%F))","db_path":"$(DUCKDB)","threads":0,"warmup_days":$(or $(WARMUP_DAYS),260)}}}}' > $$tmpfile; \
	PYTHONPATH=.. dagster job execute -m data_v2.dagster_project -j duckdb_price_flow_features_job -c $$tmpfile; \
	rm -f $$tmpfile

dagster-job-duckdb-listed-meta-features:
	@tmpfile=$$(mktemp); \
	echo '{"ops":{"duckdb_build_listed_meta_features_op":{"inputs":{"start":"$(or $(START),2015-01-01)","end":"$(or $(END),$(shell date +%F))","db_path":"$(DUCKDB)","threads":0}}}}' > $$tmpfile; \
	PYTHONPATH=.. dagster job execute -m data_v2.dagster_project -j duckdb_listed_meta_features_job -c $$tmpfile; \
	rm -f $$tmpfile

dagster-job-duckdb-statements:
	@tmpfile=$$(mktemp); \
	echo '{"ops":{"duckdb_fetch_statements_op":{"inputs":{"start":"$(or $(START),2015-01-01)","end":"$(or $(END),$(shell date +%F))","db_path":"$(DUCKDB)","holiday_division":"$(or $(HOLIDAY_DIVISION),1,2)","sleep_sec":$(or $(SLEEP_SEC),0.0),"threads":0,"auto_fetch_calendar":true,"workers":$(or $(WORKERS),10)}}}}' > $$tmpfile; \
	PYTHONPATH=.. dagster job execute -m data_v2.dagster_project -j duckdb_statements_job -c $$tmpfile; \
	rm -f $$tmpfile

dagster-job-duckdb-fs-details:
	@tmpfile=$$(mktemp); \
	echo '{"ops":{"duckdb_fetch_fs_details_op":{"inputs":{"start":"$(or $(START),2015-01-01)","end":"$(or $(END),$(shell date +%F))","db_path":"$(DUCKDB)","holiday_division":"$(or $(HOLIDAY_DIVISION),1,2)","sleep_sec":$(or $(SLEEP_SEC),0.0),"threads":0,"auto_fetch_calendar":true,"workers":$(or $(WORKERS),10)}}}}' > $$tmpfile; \
	PYTHONPATH=.. dagster job execute -m data_v2.dagster_project -j duckdb_fs_details_job -c $$tmpfile; \
	rm -f $$tmpfile

dagster-job-duckdb-wmi:
	@tmpfile=$$(mktemp); \
	echo '{"ops":{"duckdb_fetch_weekly_margin_interest_op":{"inputs":{"start":"$(or $(START),2015-01-01)","end":"$(or $(END),$(shell date +%F))","db_path":"$(DUCKDB)","threads":0,"code":"$(or $(CODE),)"}}}}' > $$tmpfile; \
	PYTHONPATH=.. dagster job execute -m data_v2.dagster_project -j duckdb_weekly_margin_interest_job -c $$tmpfile; \
	rm -f $$tmpfile

dagster-job-duckdb-ss:
	@tmpfile=$$(mktemp); \
	echo '{"ops":{"duckdb_fetch_short_selling_op":{"inputs":{"start":"$(or $(START),2015-01-01)","end":"$(or $(END),$(shell date +%F))","db_path":"$(DUCKDB)","threads":0,"sector33code":"$(or $(SECTOR33CODE),)"}}}}' > $$tmpfile; \
	PYTHONPATH=.. dagster job execute -m data_v2.dagster_project -j duckdb_short_selling_job -c $$tmpfile; \
	rm -f $$tmpfile

dagster-job-duckdb-ssp:
	@tmpfile=$$(mktemp); \
	echo '{"ops":{"duckdb_fetch_short_selling_positions_op":{"inputs":{"start":"$(or $(START),2015-01-01)","end":"$(or $(END),$(shell date +%F))","db_path":"$(DUCKDB)","threads":0,"code":"$(or $(CODE),)"}}}}' > $$tmpfile; \
	PYTHONPATH=.. dagster job execute -m data_v2.dagster_project -j duckdb_short_selling_positions_job -c $$tmpfile; \
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

duckdb-fetch-weekly-margin-interest:
	@PYTHONPATH=.. $(PYTHON) scripts/duckdb_loader.py \
		--db $(DUCKDB) \
		fetch-weekly-margin-interest \
		--start $(or $(START),2015-01-01) \
		--end $(or $(END),$(shell date +%F))

duckdb-fetch-short-selling:
	@PYTHONPATH=.. $(PYTHON) scripts/duckdb_loader.py \
		--db $(DUCKDB) \
		fetch-short-selling \
		--start $(or $(START),2015-01-01) \
		--end $(or $(END),$(shell date +%F)) \
		$(if $(SECTOR33CODE),--sector33code $(SECTOR33CODE),)

duckdb-fetch-short-selling-positions:
	@PYTHONPATH=.. $(PYTHON) scripts/duckdb_loader.py \
		--db $(DUCKDB) \
		fetch-short-selling-positions \
		--start $(or $(START),2015-01-01) \
		--end $(or $(END),$(shell date +%F)) \
		$(if $(CODE),--code $(CODE),)

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
