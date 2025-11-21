.PHONY: help dataset train-atft train-apex health-check status \
        build-2025 build-2024 build-2020-2024 build-all \
        build-2025-year-with-graph build-2023-2025-dataset \
        build-raw \
        validate validate-meta validate-quality validate-phase1 \
        clean-chunks list-chunks

help:
	@echo "gogooku5 migration helper targets"
	@echo ""
	@echo "Dataset Building:"
	@echo "  make build-2025         -> 2025年全四半期ビルド (Q1-Q4)"
	@echo "  make build-2020-2024    -> 2020-2024年全期間ビルド"
	@echo "  make build-all          -> 全期間ビルド (2020-2025)"
	@echo "  make build-raw          -> Rawスナップショットを四半期チャンク化し raw_manifest を再生成"
	@echo ""
	@echo "Validation:"
	@echo "  make validate           -> 全検証 (メタデータ + データ品質)"
	@echo "  make validate-meta      -> メタデータ検証のみ (2025年)"
	@echo "  make validate-quality   -> データ品質検証のみ (2025年)"
	@echo "  make validate-phase1    -> Phase 1機能検証 (単一チャンク)"
	@echo ""
	@echo "Utilities:"
	@echo "  make list-chunks        -> チャンク一覧表示"
	@echo "  make clean-chunks YEAR=2025  -> 指定年のチャンク削除"
	@echo ""
	@echo "Merged Dataset Pipelines:"
	@echo "  make build-2023-2025-dataset"
	@echo "      -> 2023-2025統合データセット (固定範囲)"
	@echo "  make build-range-dataset START_YEAR=2020 END_YEAR=2025"
	@echo "      -> 任意期間の統合データセット"
	@echo ""
	@echo "Legacy (Delegated):"
	@echo "  make dataset            -> delegate to data package"
	@echo "  make train-atft         -> delegate to ATFT-GAT-FAN package"
	@echo "  make train-apex         -> delegate to APEX-Ranker package"
	@echo "  make health-check       -> run shared health diagnostics"
	@echo "  make status             -> summarize dataset + model states"

_dataset_make := $(CURDIR)/data/Makefile
_models_atft := $(CURDIR)/models/atft_gat_fan/Makefile.train
_models_apex := $(CURDIR)/models/apex_ranker/Makefile.train
_tools_health := $(CURDIR)/tools/health-check.sh

ifeq (,$(wildcard $(_dataset_make)))
$(warning data package Makefile not found yet; add gogooku5/data/Makefile)
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

_output_dir := $(CURDIR)/data/output
_data_dir := $(CURDIR)/data
_tests_dir := $(CURDIR)/data/tests
_scripts_dir := $(CURDIR)/data/scripts
_dim_security := $(_output_dir)/dim_security.parquet
_raw_manifest := $(_output_dir)/raw_manifest.json

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

dataset:
	@$(MAKE) -C data build

train-atft:
	@$(MAKE) -C models/atft_gat_fan train

train-apex:
	@$(MAKE) -C models/apex_ranker train

health-check:
	@$(MAKE) -C tools health-check

status:
	@$(MAKE) -C tools status
