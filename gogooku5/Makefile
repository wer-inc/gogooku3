.PHONY: help dataset dataset-quick train-atft train-apex health-check status \
        build-2025 build-2024 build-2020-2024 build-all \
        build-2025-year-with-graph build-2023-2025-dataset \
        build-raw fetch-raw data-health \
        validate validate-meta validate-quality validate-phase1 \
        clean-chunks list-chunks

help:
	@echo "gogooku5 - Japanese Stock Market ML Pipeline"
	@echo "============================================="
	@echo ""
	@echo "🚀 Quick Start (data_v2 DuckDB pipeline):"
	@echo "  make dataset            -> Full dataset build (DuckDB-based)"
	@echo "  make dataset-quick      -> Quick test (last 30 days)"
	@echo "  make status             -> Show project status"
	@echo "  make health-check       -> Run diagnostics"
	@echo ""
	@echo "📊 Data V2 Commands (Current Infrastructure):"
	@echo "  make -C data_v2 duckdb-init              -> Initialize DuckDB"
	@echo "  make -C data_v2 duckdb-fetch-listed      -> Fetch listed info"
	@echo "  make -C data_v2 duckdb-fetch-daily-quotes-> Fetch price data"
	@echo "  make -C data_v2 duckdb-fetch-breakdown   -> Fetch demand/supply"
	@echo "  make -C data_v2 duckdb-compute-features  -> Compute 150+ features"
	@echo "  make -C data_v2 help                     -> Show all data_v2 targets"
	@echo ""
	@echo "🤖 Model Training:"
	@echo "  make train-atft         -> Train ATFT-GAT-FAN model"
	@echo "  make train-apex         -> Train APEX-Ranker model"
	@echo ""
	@echo "📦 Legacy Pipeline (data_deprecated/, chunk-based):"
	@echo "  make build-2025         -> 2025年全四半期ビルド (Q1-Q4)"
	@echo "  make build-2020-2024    -> 2020-2024年全期間ビルド"
	@echo "  make build-all          -> 全期間ビルド (2020-2025)"
	@echo "  make build-raw          -> Rawスナップショットを四半期チャンク化"
	@echo "  make data-health DATASET=... [BASELINE=...]"
	@echo "      -> 簡易ヘルスチェック (同日相関/PSI/KS)"
	@echo ""
	@echo "🔧 Utilities:"
	@echo "  make list-chunks        -> チャンク一覧表示"
	@echo "  make clean-chunks YEAR=2025  -> 指定年のチャンク削除"
	@echo ""
	@echo "📚 Documentation:"
	@echo "  data_v2/FEATURES.md     -> 150+ feature definitions"
	@echo "  data_v2/DUCKDB_QUICKSTART.md -> Quick reference"
	@echo "  DATA_V2_EXPLORATION_REPORT.md -> Architecture overview"

# Data pipeline paths (data_v2 is the current infrastructure)
_dataset_make := $(CURDIR)/data_v2/Makefile
_models_atft := $(CURDIR)/models/atft_gat_fan/Makefile.train
_models_apex := $(CURDIR)/models/apex_ranker/Makefile.train
_tools_health := $(CURDIR)/tools/health-check.sh

# Legacy data directory (data_deprecated/) for backward compatibility
_legacy_data_dir := $(CURDIR)/data_deprecated

ifeq (,$(wildcard $(_dataset_make)))
$(warning data_v2 package Makefile not found yet; add gogooku5/data_v2/Makefile)
endif
ifeq (,$(wildcard $(_models_atft)))
$(warning ATFT-GAT-FAN Makefile.train not found yet; add gogooku5/models/atft_gat_fan/Makefile.train)
endif
ifeq (,$(wildcard $(_models_apex)))
$(warning APEX-Ranker Makefile.train not found yet; add gogooku5/models/apex_ranker/Makefile.train)
endif
ifeq (,$(wildcard $(_tools_health)))
$(warning health-check script not found yet; add gogooku5/tools/health-check.sh)
endif

# Common variables ---------------------------------------------------------
# NOTE: data_v2 uses DuckDB as system of record, output goes to data_v2/output/
_output_dir := $(CURDIR)/data_v2/output
_data_dir := $(CURDIR)/data_v2
_tests_dir := $(CURDIR)/data_deprecated/tests
_scripts_dir := $(CURDIR)/data_deprecated/scripts
_dim_security := $(_output_dir)/dim_security.parquet
_raw_manifest := $(_output_dir)/raw_manifest.json
RAW_START ?= 2020-01-01
RAW_END ?= 2025-11-20
RAW_CHUNK_MONTHS ?= 3

# Default year range for merged training dataset (can be overridden on CLI)
START_YEAR ?= 2023
END_YEAR ?= 2025

# Build configuration
export DATA_OUTPUT_DIR := $(_output_dir)
export DIM_SECURITY_PATH := $(_dim_security)
export RAW_MANIFEST_PATH := $(_raw_manifest)
export CATEGORICAL_COLUMNS := Code
export MAX_CONCURRENT_FETCH := 200
export MAX_PARALLEL_WORKERS := 128
export QUOTES_PARALLEL_WORKERS := 200

# Dataset Building targets -------------------------------------------------

build-2025:
	@echo "🔨 Building 2025年全四半期 (Q1-Q4)..."
	@CURRENT_YEAR=$$(date +%Y); \
	CURRENT_MONTH=$$(date +%m); \
	if [ "$$CURRENT_YEAR" = "2025" ]; then \
		END_DATE=$$(date -d "$$(date +%Y-%m-01) +1 month -1 day" +%Y-%m-%d); \
	else \
		END_DATE="2025-12-31"; \
	fi; \
	echo "📅 Using end date: $$END_DATE (current: $$(date +%Y-%m-%d))"; \
	python3 $(_scripts_dir)/build_chunks.py \
		--start 2025-01-01 \
		--end $$END_DATE \
		--resume
	@echo "✅ 2025年ビルド完了"

build-2024:
	@echo "🔨 Building 2024年全四半期 (Q1-Q4)..."
	@python3 $(_scripts_dir)/build_chunks.py \
		--start 2024-01-01 \
		--end 2024-12-31 \
		--resume
	@echo "✅ 2024年ビルド完了"

build-2020-2024:
	@echo "🔨 Building 2020-2024年全期間..."
	@python3 $(_scripts_dir)/build_chunks.py \
		--start 2020-01-01 \
		--end 2024-12-31 \
		--resume
	@echo "✅ 2020-2024年ビルド完了"

build-all:
	@echo "🔨 Building 全期間 (2020-2025)..."
	@python3 $(_scripts_dir)/build_chunks.py \
		--start 2020-01-01 \
		--end 2025-11-15 \
		--resume
	@echo "✅ 全期間ビルド完了"

# Year-level dataset with graph features (2025)
#  1) Disable graph_* at chunk build (ENABLE_GRAPH_FEATURES=0)
#  2) Merge 2025Q1–Q4 into a single dataset (using merge_chunks)
#  3) Add graph_* features once over the merged 2025 dataset
build-2025-year-with-graph:
	@echo "🔨 [2025] Step 1/3: Building chunks without graph features (ENABLE_GRAPH_FEATURES=0)..."
	@ENABLE_GRAPH_FEATURES=0 python3 $(_scripts_dir)/build_chunks.py \
		--start 2025-01-01 \
		--end 2025-12-31 \
		--resume
	@echo "✅ [2025] Chunk build without graph features completed"
	@echo "🔗 [2025] Step 2/3: Merging completed chunks into a full dataset..."
	@PYTHONPATH=$(_data_dir)/src ../venv/bin/python $(_data_dir)/tools/merge_chunks.py \
		--chunks-dir $(_output_dir)/chunks \
		--output-dir $(_output_dir)/datasets \
		--allow-partial
	@echo "✅ [2025] Chunk merge completed (see $(_output_dir)/datasets)"
	@echo "📈 [2025] Step 3/3: Adding graph_* features to merged dataset..."
	@../venv/bin/python $(_data_dir)/tools/add_graph_features_full.py \
		--input  $(_output_dir)/datasets/ml_dataset_latest.parquet \
		--output $(_output_dir)/datasets/ml_dataset_latest_with_graph.parquet \
		--date-col Date --code-col Code
	@echo "✅ [2025] Year-level dataset with graph features ready:"
	@echo "   Input:  $(_output_dir)/datasets/ml_dataset_latest.parquet"
	@echo "   Output: $(_output_dir)/datasets/ml_dataset_latest_with_graph.parquet"

# Unified 2023-2025 ML dataset pipeline (chunks -> merge -> beta/alpha -> basis_gate -> graph33 -> drop NULL)
build-2023-2025-dataset:
	@echo "🔨 [2023-2025] Step 1/5: Staging chunks for 2023-2025..."
	@STAGING_DIR="$(_output_dir)/chunks_2023_2025"; \
	rm -rf "$$STAGING_DIR"; \
	mkdir -p "$$STAGING_DIR"; \
	for q in 2023Q1 2023Q2 2023Q3 2023Q4 2024Q1 2024Q2 2024Q3 2024Q4 2025Q1 2025Q2 2025Q3 2025Q4; do \
	  if [ ! -d "$(_output_dir)/chunks/$$q" ]; then \
	    echo "❌ Missing chunk: $$q (expected at $(_output_dir)/chunks/$$q)"; \
	    exit 1; \
	  fi; \
	  ln -s "$(_output_dir)/chunks/$$q" "$$STAGING_DIR/$$q"; \
	done
	@echo "✅ [2023-2025] Staging complete: $(_output_dir)/chunks_2023_2025"
	@echo "🔗 [2023-2025] Step 2/5: Merging staged chunks into a full dataset..."
	@PYTHONPATH=$(_data_dir)/src ../venv/bin/python $(_data_dir)/tools/merge_chunks.py \
		--chunks-dir $(_output_dir)/chunks_2023_2025 \
		--output-dir $(_output_dir)/datasets \
		--allow-partial
	@echo "✅ [2023-2025] Chunk merge completed (see $(_output_dir)/datasets)"
	@echo "📈 [2023-2025] Step 3/5: Adding beta60_topix/alpha60_topix and bd_net_adv60..."
	@BASE="$(_output_dir)/datasets/ml_dataset_latest.parquet"; \
	if [ ! -f "$$BASE" ]; then \
	  echo "❌ Expected merged dataset symlink not found: $$BASE"; \
	  exit 1; \
	fi; \
	OUT_BETA="$(_output_dir)/datasets/ml_dataset_2023_2025_beta_bd.parquet"; \
	if [ -z "$$FORCE_REBUILD" ] && [ -f "$$OUT_BETA" ] && [ "$$OUT_BETA" -nt "$$BASE" ]; then \
	  echo "⏩ [2023-2025] beta/alpha + bd_net_adv60 already up-to-date (use FORCE_REBUILD=1 to recompute)"; \
	else \
	  PYTHONPATH=$(_data_dir)/src ../venv/bin/python $(_data_dir)/tools/add_beta_alpha_bd_features_full.py \
	    --input  "$$BASE" \
	    --output "$$OUT_BETA" \
	    --date-col Date --code-col Code \
	    --window 60 --min-periods-beta 30 \
	    --beta-deriv-window 20 --beta-deriv-min-periods 10 \
	    --beta-deriv-sigma 2.0 --min-periods-adv 10; \
	fi
	@echo "✅ [2023-2025] beta/alpha + bd_net_adv60 augmentation complete"
	@echo "📉 [2023-2025] Step 4/5: Adding basis_gate feature family..."
	@IN_BETA="$(_output_dir)/datasets/ml_dataset_2023_2025_beta_bd.parquet"; \
	OUT_BASIS="$(_output_dir)/datasets/ml_dataset_2023_2025_beta_bd_basis.parquet"; \
	if [ -z "$$FORCE_REBUILD" ] && [ -f "$$OUT_BASIS" ] && [ "$$OUT_BASIS" -nt "$$IN_BETA" ]; then \
	  echo "⏩ [2023-2025] basis_gate already up-to-date (use FORCE_REBUILD=1 to recompute)"; \
	else \
	  PYTHONPATH=$(_data_dir)/src ../venv/bin/python $(_data_dir)/tools/add_basis_gate_full.py \
	    --input  "$$IN_BETA" \
	    --output "$$OUT_BASIS" \
	    --date-col Date --code-col Code; \
	fi
	@echo "✅ [2023-2025] basis_gate augmentation complete"
	@echo "🌐 [2023-2025] Step 5/5: Adding graph_* (base + derived) and dropping all-NULL columns..."
	@IN_BASIS="$(_output_dir)/datasets/ml_dataset_2023_2025_beta_bd_basis.parquet"; \
	OUT_GRAPH="$(_output_dir)/datasets/ml_dataset_2023_2025_with_graph33.parquet"; \
	if [ -z "$$FORCE_REBUILD" ] && [ -f "$$OUT_GRAPH" ] && [ "$$OUT_GRAPH" -nt "$$IN_BASIS" ]; then \
	  echo "⏩ [2023-2025] graph_* already up-to-date (use FORCE_REBUILD=1 to recompute)"; \
	else \
	  PYTHONPATH=$(_data_dir)/src ../venv/bin/python $(_data_dir)/tools/add_graph_features_full.py \
	    --input  "$$IN_BASIS" \
	    --output "$$OUT_GRAPH" \
	    --date-col Date --code-col Code \
	    --window-days 60 --min-observations 20 --correlation-threshold 0.3 \
	    --with-derived; \
	fi
	@IN_GRAPH="$(_output_dir)/datasets/ml_dataset_2023_2025_with_graph33.parquet"; \
	OUT_FINAL="$(_output_dir)/datasets/ml_dataset_2023_2025_final_pruned.parquet"; \
	if [ -z "$$FORCE_REBUILD" ] && [ -f "$$OUT_FINAL" ] && [ "$$OUT_FINAL" -nt "$$IN_GRAPH" ]; then \
	  echo "⏩ [2023-2025] final_pruned already up-to-date (use FORCE_REBUILD=1 to recompute)"; \
	else \
	  PYTHONPATH=$(_data_dir)/src ../venv/bin/python $(_data_dir)/tools/drop_all_null_columns.py \
	    --input  "$$IN_GRAPH" \
	    --output "$$OUT_FINAL" \
	    --keep-col Date --keep-col Code \
	    --keep-col target_1d --keep-col target_5d \
	    --keep-col target_10d --keep-col target_20d \
	    --report $(_output_dir)/datasets/ml_dataset_2023_2025_final_pruned_dropped_cols.json; \
	fi
	@ln -sf ml_dataset_2023_2025_final_pruned.parquet $(_output_dir)/datasets/ml_dataset_2023_2025_final.parquet
	@echo "✅ [2023-2025] Final training dataset ready:"
	@echo "   Path:    $(_output_dir)/datasets/ml_dataset_2023_2025_final_pruned.parquet"
	@echo "   Symlink: $(_output_dir)/datasets/ml_dataset_2023_2025_final.parquet"

# Generic merged ML dataset pipeline for arbitrary year range.
# Usage examples:
#   make build-range-dataset                 # uses START_YEAR=2023 END_YEAR=2025
#   make build-range-dataset START_YEAR=2020 END_YEAR=2025
build-range-dataset:
	@echo "🔨 [$(START_YEAR)-$(END_YEAR)] Step 1/5: Staging chunks..."
	@STAGING_DIR="$(_output_dir)/chunks_$(START_YEAR)_$(END_YEAR)"; \
	rm -rf "$$STAGING_DIR"; \
	mkdir -p "$$STAGING_DIR"; \
	for y in $$(seq $(START_YEAR) $(END_YEAR)); do \
	  for qi in 1 2 3 4; do \
	    q="$${y}Q$${qi}"; \
	    if [ ! -d "$(_output_dir)/chunks/$$q" ]; then \
	      echo "❌ Missing chunk: $$q (expected at $(_output_dir)/chunks/$$q)"; \
	      exit 1; \
	    fi; \
	    ln -s "$(_output_dir)/chunks/$$q" "$$STAGING_DIR/$$q"; \
	  done; \
	done
	@echo "✅ [$(START_YEAR)-$(END_YEAR)] Staging complete: $(_output_dir)/chunks_$(START_YEAR)_$(END_YEAR)"
	@echo "🔗 [$(START_YEAR)-$(END_YEAR)] Step 2/5: Merging staged chunks into a full dataset..."
	@PYTHONPATH=$(_data_dir)/src ../venv/bin/python $(_data_dir)/tools/merge_chunks.py \
		--chunks-dir $(_output_dir)/chunks_$(START_YEAR)_$(END_YEAR) \
		--output-dir $(_output_dir)/datasets \
		--allow-partial
	@echo "✅ [$(START_YEAR)-$(END_YEAR)] Chunk merge completed (see $(_output_dir)/datasets)"
	@echo "📈 [$(START_YEAR)-$(END_YEAR)] Step 3/5: Adding beta60_topix/alpha60_topix and bd_net_adv60..."
	@BASE="$(_output_dir)/datasets/ml_dataset_latest.parquet"; \
	if [ ! -f "$$BASE" ]; then \
	  echo "❌ Expected merged dataset symlink not found: $$BASE"; \
	  exit 1; \
	fi; \
	DATASET_PREFIX="ml_dataset_$(START_YEAR)_$(END_YEAR)"; \
	OUT_BETA="$(_output_dir)/datasets/$${DATASET_PREFIX}_beta_bd.parquet"; \
	if [ -z "$$FORCE_REBUILD" ] && [ -f "$$OUT_BETA" ] && [ "$$OUT_BETA" -nt "$$BASE" ]; then \
	  echo "⏩ [$(START_YEAR)-$(END_YEAR)] beta/alpha + bd_net_adv60 already up-to-date (use FORCE_REBUILD=1 to recompute)"; \
	else \
	  PYTHONPATH=$(_data_dir)/src ../venv/bin/python $(_data_dir)/tools/add_beta_alpha_bd_features_full.py \
	    --input  "$$BASE" \
	    --output "$$OUT_BETA" \
	    --date-col Date --code-col Code \
	    --window 60 --min-periods-beta 30 \
	    --beta-deriv-window 20 --beta-deriv-min-periods 10 \
	    --beta-deriv-sigma 2.0 --min-periods-adv 10; \
	fi
	@echo "✅ [$(START_YEAR)-$(END_YEAR)] beta/alpha + bd_net_adv60 augmentation complete"
	@echo "📉 [$(START_YEAR)-$(END_YEAR)] Step 4/5: Adding basis_gate feature family..."
	@DATASET_PREFIX="ml_dataset_$(START_YEAR)_$(END_YEAR)"; \
	IN_BETA="$(_output_dir)/datasets/$${DATASET_PREFIX}_beta_bd.parquet"; \
	OUT_BASIS="$(_output_dir)/datasets/$${DATASET_PREFIX}_beta_bd_basis.parquet"; \
	if [ -z "$$FORCE_REBUILD" ] && [ -f "$$OUT_BASIS" ] && [ "$$OUT_BASIS" -nt "$$IN_BETA" ]; then \
	  echo "⏩ [$(START_YEAR)-$(END_YEAR)] basis_gate already up-to-date (use FORCE_REBUILD=1 to recompute)"; \
	else \
	  PYTHONPATH=$(_data_dir)/src ../venv/bin/python $(_data_dir)/tools/add_basis_gate_full.py \
	    --input  "$$IN_BETA" \
	    --output "$$OUT_BASIS" \
	    --date-col Date --code-col Code; \
	fi
	@echo "✅ [$(START_YEAR)-$(END_YEAR)] basis_gate augmentation complete"
	@echo "🌐 [$(START_YEAR)-$(END_YEAR)] Step 5/5: Adding graph_* (base + derived) and dropping all-NULL columns..."
	@DATASET_PREFIX="ml_dataset_$(START_YEAR)_$(END_YEAR)"; \
	IN_BASIS="$(_output_dir)/datasets/$${DATASET_PREFIX}_beta_bd_basis.parquet"; \
	OUT_GRAPH="$(_output_dir)/datasets/$${DATASET_PREFIX}_with_graph33.parquet"; \
	if [ -z "$$FORCE_REBUILD" ] && [ -f "$$OUT_GRAPH" ] && [ "$$OUT_GRAPH" -nt "$$IN_BASIS" ]; then \
	  echo "⏩ [$(START_YEAR)-$(END_YEAR)] graph_* already up-to-date (use FORCE_REBUILD=1 to recompute)"; \
	else \
	  PYTHONPATH=$(_data_dir)/src ../venv/bin/python $(_data_dir)/tools/add_graph_features_full.py \
	    --input  "$$IN_BASIS" \
	    --output "$$OUT_GRAPH" \
	    --date-col Date --code-col Code \
	    --window-days 60 --min-observations 20 --correlation-threshold 0.3 \
	    --with-derived; \
	fi; \
	OUT_FINAL="$(_output_dir)/datasets/$${DATASET_PREFIX}_final_pruned.parquet"; \
	if [ -z "$$FORCE_REBUILD" ] && [ -f "$$OUT_FINAL" ] && [ "$$OUT_FINAL" -nt "$$OUT_GRAPH" ]; then \
	  echo "⏩ [$(START_YEAR)-$(END_YEAR)] final_pruned already up-to-date (use FORCE_REBUILD=1 to recompute)"; \
	else \
	  PYTHONPATH=$(_data_dir)/src ../venv/bin/python $(_data_dir)/tools/drop_all_null_columns.py \
	    --input  "$$OUT_GRAPH" \
	    --output "$$OUT_FINAL" \
	    --keep-col Date --keep-col Code \
	    --keep-col target_1d --keep-col target_5d \
	    --keep-col target_10d --keep-col target_20d \
	    --report $(_output_dir)/datasets/$${DATASET_PREFIX}_final_pruned_dropped_cols.json; \
	fi; \
	ln -sf "$${DATASET_PREFIX}_final_pruned.parquet" $(_output_dir)/datasets/"$${DATASET_PREFIX}_final.parquet"; \
	echo "✅ [$(START_YEAR)-$(END_YEAR)] Final training dataset ready:"; \
	echo "   Path:    $(_output_dir)/datasets/$${DATASET_PREFIX}_final_pruned.parquet"; \
	echo "   Symlink: $(_output_dir)/datasets/$${DATASET_PREFIX}_final.parquet"

# Raw snapshot chunking + manifest -----------------------------------------

build-raw:
	@echo "🔨 Rawスナップショットの四半期チャンク化とマニフェスト再生成を実行します..."
	@echo "   RAW_ROOT      = $(_output_dir)/raw"
	@echo "   RAW_MANIFEST  = $(_raw_manifest)"
	@python3 $(_data_dir)/tools/manage_raw_sources.py \
		chunk \
		--raw-root "$(_output_dir)/raw"
	@python3 $(_data_dir)/tools/manage_raw_sources.py \
		manifest \
		--raw-root "$(_output_dir)/raw" \
		--manifest "$(_raw_manifest)"
	@echo "✅ Raw chunk + manifest 更新完了 -> $(_raw_manifest)"

# Fetch RAW snapshots without modifying code (saves under data/output/raw).
# - Uses inline env overrides so .env is untouched.
# - Chunked 3-month build with resume to reduce failures.
fetch-raw:
	@echo "📥 RAW取得 (期間: $(RAW_START)→$(RAW_END), チャンク: $(RAW_CHUNK_MONTHS)ヶ月, 出力: $(_output_dir)/raw)"
	@echo "🪪 listed_info をキャッシュ中 (dim_security用の入力確保)..."
	@cd $(_data_dir) && PYTHONPATH=src RAW_START=$(RAW_START) RAW_END=$(RAW_END) python -c "\
from builder.api.data_sources import DataSourceManager; import os; \
start=os.environ['RAW_START']; end=os.environ['RAW_END']; \
df=DataSourceManager().listed_info(start=start, end=end); \
print(f'listed_info cached rows: {len(df)}')"
	@echo "🔑 dim_security を事前生成中 (cache/dim_security.parquet)..."
	@cd $(_data_dir) && \
		mkdir -p $(_output_dir)/cache && \
		python scripts/build_dim_security.py \
			--output-dir $(_output_dir)/cache \
			--search-dirs $(_output_dir) $(_output_dir)/raw $(CURDIR)/output
	@cd $(_data_dir) && \
	SAVE_RAW_DATA=1 USE_RAW_STORE=1 USE_CACHE=1 \
	MAX_CONCURRENT_FETCH=16 INDEX_OPTION_PARALLEL_CONCURRENCY=4 \
	ENABLE_SCHEMA_VALIDATION=0 \
	PYTHONPATH=src python -m cli.main build \
		--start $(RAW_START) --end $(RAW_END) \
		--chunk-mode chunks --chunk-months $(RAW_CHUNK_MONTHS) \
		--resume --gpu -v
	@echo "✅ RAW取得完了 (途中失敗した場合はそのまま再実行で --resume しながら続行できます)"

# Data health check (correlation leak hints, PSI, KS if SciPy installed)
data-health:
ifndef DATASET
	$(error "DATASET is required, e.g., make data-health DATASET=data/output/chunks/2020Q1/ml_dataset.parquet BASELINE=data/output/chunks/2020Q2/ml_dataset.parquet")
endif
	@echo "🔎 Running data health check..."
	@PYTHONPATH=$(_data_dir)/src python $(_data_dir)/tools/data_health_check.py \
		--dataset $(DATASET) \
		$(if $(BASELINE),--baseline $(BASELINE),) \
		$(if $(TARGET_COLS),--target-cols $(TARGET_COLS),) \
		$(if $(SAMPLE_ROWS),--sample-rows $(SAMPLE_ROWS),)

# Validation targets -------------------------------------------------------

validate: validate-meta validate-quality
	@echo ""
	@echo "✅ 全検証完了 (メタデータ + データ品質)"

validate-meta:
	@echo "📋 2025年メタデータ検証中..."
	@python3 $(_tests_dir)/validate_2025_complete.py

validate-quality:
	@echo "🔍 2025年データ品質検証中..."
	@python3 $(_tests_dir)/validate_2025_data_quality.py

validate-phase1:
	@echo "🔬 Phase 1機能検証中 (2025Q4)..."
	@python3 $(_tests_dir)/validate_phase1_features.py

# Utility targets ----------------------------------------------------------

list-chunks:
	@echo "📂 チャンク一覧:"
	@ls -lh $(_output_dir)/chunks/ 2>/dev/null || echo "チャンクが見つかりません"

clean-chunks:
ifndef YEAR
	@echo "❌ エラー: YEAR変数を指定してください (例: make clean-chunks YEAR=2025)"
	@exit 1
endif
	@echo "🗑️  $(YEAR)年のチャンクを削除中..."
	@rm -rf $(_output_dir)/chunks/$(YEAR)Q*
	@echo "✅ $(YEAR)年のチャンクを削除しました"

# Delegated targets -------------------------------------------------------

# Primary dataset target: use data_v2 DuckDB-based infrastructure
dataset:
	@$(MAKE) -C data_v2 duckdb-init
	@echo "📅 Fetching trading calendar..."
	@$(MAKE) -C data_v2 duckdb-load-calendar
	@echo "🏢 Fetching listed info..."
	@$(MAKE) -C data_v2 duckdb-fetch-listed START=$(or $(START),2015-01-01) END=$(or $(END),$(shell date +%F))
	@echo "📈 Fetching daily quotes..."
	@$(MAKE) -C data_v2 duckdb-fetch-daily-quotes START=$(or $(START),2015-01-01) END=$(or $(END),$(shell date +%F))
	@echo "📊 Fetching breakdown..."
	@$(MAKE) -C data_v2 duckdb-fetch-breakdown START=$(or $(START),2015-01-01) END=$(or $(END),$(shell date +%F))
	@echo "🧮 Computing features..."
	@START=$(or $(START),2015-01-01) END=$(or $(END),$(shell date +%F)) $(MAKE) -C data_v2 duckdb-compute-features
	@echo "✅ Dataset build complete. Data stored in: data_v2/output/jquants.duckdb"

# Quick dataset for testing (last 30 days)
dataset-quick:
	@$(MAKE) dataset START=$(shell date -d '30 days ago' +%Y-%m-%d) END=$(shell date +%F)

train-atft:
	@$(MAKE) -C models/atft_gat_fan train

train-apex:
	@$(MAKE) -C models/apex_ranker train

health-check:
	@echo "🔍 Running health checks..."
	@echo "   DuckDB: $(shell ls -lh data_v2/output/jquants.duckdb 2>/dev/null | awk '{print $$5}' || echo 'not found')"
	@echo "   Models: $(shell ls models/apex_ranker/output/*.pt 2>/dev/null | wc -l) APEX checkpoints"
	@echo "   GPU: $(shell nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo 'not available')"

status:
	@echo "📊 gogooku5 Status"
	@echo "================="
	@echo ""
	@echo "📁 Data Infrastructure:"
	@echo "   data_v2/:    DuckDB-based pipeline (current)"
	@echo "   data_deprecated/: Legacy Polars pipeline (archived)"
	@echo ""
	@echo "📊 DuckDB Database:"
	@if [ -f data_v2/output/jquants.duckdb ]; then \
		echo "   Size: $$(ls -lh data_v2/output/jquants.duckdb | awk '{print $$5}')"; \
		echo "   Tables: $$(PYTHONPATH=data_v2 python -c "import duckdb; con=duckdb.connect('data_v2/output/jquants.duckdb'); print(', '.join([r[0] for r in con.execute('SHOW TABLES').fetchall()]))" 2>/dev/null || echo 'error reading')"; \
	else \
		echo "   Not initialized. Run: make dataset"; \
	fi
	@echo ""
	@echo "🤖 Models:"
	@echo "   APEX-Ranker: $$(ls models/apex_ranker/output/*.pt 2>/dev/null | wc -l) checkpoints"
	@echo "   ATFT-GAT-FAN: $$(ls models/atft_gat_fan/output/*.pt 2>/dev/null | wc -l) checkpoints"
