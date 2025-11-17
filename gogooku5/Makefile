.PHONY: help dataset train-atft train-apex health-check status \
        build-2025 build-2020-2024 build-all \
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

_output_dir := $(CURDIR)/../output_g5
_data_dir := $(CURDIR)/data
_tests_dir := $(CURDIR)/data/tests
_scripts_dir := $(CURDIR)/data/scripts
_dim_security := $(_output_dir)/dim_security.parquet
_raw_manifest := $(_output_dir)/raw_manifest.json

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
