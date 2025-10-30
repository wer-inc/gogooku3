# gogooku5 モジュラー設計移植計画

## 設計思想

**データセット生成と学習の完全分離** + **マルチモデル対応**

- データセット生成は `data/` ディレクトリに完全隔離（`data/src/builder/` パッケージ）
- 複数モデル（ATFT-GAT-FAN, APEX-Ranker等）が共通データセット（`data/output/ml_dataset_latest.parquet`）を参照
- Claude Code / Codex の積極活用を前提とした構造

---

## プロジェクト全体構造

```
gogooku5/
├── data/                 # データセット生成（完全独立）
│   ├── src/
│   │   └── builder/            # データセット構築パッケージ
│   │       ├── api/            # API クライアント
│   │       ├── features/       # 特徴量生成（30モジュール）
│   │       ├── pipelines/      # データパイプライン
│   │       ├── config/         # 設定管理
│   │       └── utils/          # ユーティリティ
│   ├── scripts/
│   │   ├── build.py            # メインビルドスクリプト
│   │   └── build_optimized.py  # 最適化パイプライン
│   ├── tests/
│   ├── output/                 # 生成データ（共有）
│   │   ├── ml_dataset_v1.0.0.parquet      # バージョン付き実体
│   │   ├── ml_dataset_latest.parquet      # シンボリックリンク（★モデル参照）
│   │   └── ml_dataset.parquet             # レガシー互換（非推奨）
│   ├── pyproject.toml          # 独立した依存管理
│   ├── Makefile
│   └── README.md
│
├── models/               # モデル別ディレクトリ
│   ├── atft_gat_fan/          # ATFT-GAT-FAN
│   │   ├── src/
│   │   │   └── atft_gat_fan/
│   │   │       ├── models/
│   │   │       ├── training/
│   │   │       └── config/
│   │   ├── scripts/
│   │   │   └── train_atft.py
│   │   ├── configs/
│   │   ├── tests/
│   │   ├── pyproject.toml     # モデル固有の依存
│   │   ├── Makefile.train
│   │   └── README.md
│   │
│   └── apex_ranker/           # APEX-Ranker
│       ├── src/
│       │   └── apex_ranker/
│       │       ├── models/
│       │       ├── training/
│       │       └── config/
│       ├── scripts/
│       │   ├── train_v0.py
│       │   └── inference_v0.py
│       ├── tests/
│       ├── pyproject.toml
│       ├── Makefile.train
│       └── README.md
│
├── common/               # 共通ユーティリティ（オプショナル）
│   ├── src/
│   │   └── common/
│   │       ├── data/        # データローダー基底クラス
│   │       ├── metrics/     # 共通評価指標
│   │       └── utils/       # ログ、GCS等
│   └── pyproject.toml
│
├── tools/                # Claude Code / Codex連携
│   ├── claude-code.sh
│   ├── codex.sh
│   └── health-check.sh
│
├── Makefile              # トップレベル統合Makefile
├── MIGRATION_PLAN.md     # このファイル
└── README.md             # プロジェクト全体ドキュメント
```

---

## Phase 1: Dataset生成モジュール（2-3週間）

### 目標
gogooku3と同等の~307特徴量データセット生成を、完全独立したモジュールとして実装

### 1.1 ディレクトリ構造（data/配下）

```
data/
├── src/builder/
│   ├── __init__.py
│   │
│   ├── api/                                # API & データ取得
│   │   ├── __init__.py
│   │   ├── jquants_fetcher.py             # JQuants API クライアント
│   │   ├── margin_fetcher.py              # 信用取引データ
│   │   ├── market_filter.py               # 市場コードフィルタ
│   │   ├── calendar_fetcher.py            # 営業日カレンダー
│   │   ├── axis_decider.py                # 取得軸の自動選択
│   │   ├── quotes_fetcher.py              # 銘柄別価格取得
│   │   ├── listed_manager.py              # 上場情報管理
│   │   └── event_detector.py              # イベント検知
│   │
│   ├── features/                          # 特徴量生成
│   │   ├── __init__.py
│   │   ├── core/                          # 旧src/gogooku3/features/
│   │   │   ├── __init__.py
│   │   │   ├── quality_features_polars.py # ファンダメンタル特徴量
│   │   │   ├── index_features.py          # 指数関連特徴量
│   │   │   ├── index_option.py            # 日経225オプション
│   │   │   ├── margin_daily.py            # 日次信用取引
│   │   │   ├── margin_weekly.py           # 週次信用取引
│   │   │   ├── short_selling.py           # 空売り
│   │   │   ├── short_selling_sector.py    # セクター空売り
│   │   │   ├── sector_aggregation.py      # セクター集計
│   │   │   ├── sector_cross_sectional.py  # セクター横断
│   │   │   ├── advanced_features.py       # 高度な技術指標
│   │   │   ├── advanced_volatility.py     # ボラティリティ
│   │   │   ├── earnings_events.py         # 決算イベント
│   │   │   ├── earnings_features.py       # 決算特徴量
│   │   │   ├── graph_features_gpu.py      # グラフ特徴量（GPU）
│   │   │   ├── graph_features.py          # グラフ特徴量（CPU）
│   │   │   └── ... (他15ファイル)
│   │   │
│   │   ├── legacy/                        # 旧src/features/
│   │   │   ├── __init__.py
│   │   │   ├── calendar_utils.py          # 営業日計算
│   │   │   ├── market_features.py         # 市場特徴量
│   │   │   ├── safe_joiner_v2.py          # 安全なデータ結合
│   │   │   ├── flow_features_v2.py        # フロー特徴量
│   │   │   ├── quality_features.py        # 品質特徴量
│   │   │   └── peer_features.py           # ピア比較特徴量
│   │   │
│   │   └── macro/                         # マクロ経済特徴量
│   │       ├── __init__.py
│   │       ├── vix.py                     # VIX指数
│   │       ├── fx.py                      # 為替レート
│   │       └── btc.py                     # ビットコイン
│   │
│   ├── pipelines/                         # データパイプライン
│   │   ├── __init__.py
│   │   ├── full_pipeline.py               # メインパイプライン
│   │   ├── optimized_pipeline.py          # 最適化パイプライン（キャッシュ含む）
│   │   └── dataset_builder.py             # ML dataset構築
│   │
│   ├── config/                            # 設定管理
│   │   ├── __init__.py
│   │   └── settings.py                    # Pydantic設定
│   │
│   └── utils/                             # ユーティリティ
│       ├── __init__.py
│       ├── env.py                         # 環境変数管理
│       ├── storage.py                     # GCSアップロード
│       ├── cache.py                       # キャッシュ管理
│       └── logger.py                      # ログ設定
│
├── scripts/                               # CLI エントリーポイント
│   ├── build.py                           # メインビルドスクリプト
│   ├── build_optimized.py                 # 最適化パイプライン実行
│   └── update_cache.py                    # キャッシュ更新スクリプト
│
├── tests/                                 # テスト
│   ├── __init__.py
│   ├── conftest.py                        # pytest設定
│   ├── fixtures/                          # テストデータ
│   │   └── sample_data.parquet
│   ├── unit/
│   │   ├── test_jquants_fetcher.py        # API fetcher単体
│   │   ├── test_cache_logic.py            # キャッシュロジック
│   │   └── test_feature_generators.py     # 特徴量生成
│   └── integration/
│       ├── test_pipeline_v4.py            # パイプライン統合
│       └── test_full_dataset.py           # エンドツーエンド
│
├── output/                                # データ出力（共有）
│   ├── raw/                               # 生データキャッシュ
│   │   ├── prices/
│   │   ├── indices/
│   │   └── statements/
│   ├── cache/                             # グラフキャッシュ
│   ├── ml_dataset_v1.0.0.parquet          # バージョン付き実体
│   ├── ml_dataset_latest.parquet          # ★シンボリックリンク（モデル参照用）
│   └── ml_dataset.parquet                 # レガシー互換（オプショナル）
│
├── logs/                                  # ログファイル
│   └── build_YYYYMMDD_HHMMSS.log
│
├── .env.example                           # 環境変数テンプレート
├── pyproject.toml                         # データセット専用依存
├── Makefile                               # データセット生成コマンド
├── README.md                              # データセットモジュール説明
└── CLAUDE.md                              # Claude Code用ガイド
```

### 1.2 pyproject.toml（data専用）

```toml
[project]
name = "builder"
version = "0.1.0"
description = "Japanese stock ML dataset builder"
requires-python = ">=3.10"

dependencies = [
    # コアデータ処理（学習ライブラリは不要）
    "polars>=0.20.0",
    "pyarrow>=14.0.0",
    "numpy>=1.24.0",
    "scipy>=1.11.0",

    # 特徴量エンジニアリング
    "pandas-ta>=0.3.14b0",
    "PyWavelets>=1.4.0",
    "networkx>=3.0",
    "joblib>=1.3.0",

    # API & 非同期
    "aiohttp>=3.9.0",
    "requests>=2.31.0",
    "nest-asyncio>=1.5.8",

    # 設定・ユーティリティ
    "pydantic>=2.5.0",
    "pydantic-settings>=2.0.0",
    "python-dotenv>=1.0.0",
    "PyYAML>=6.0.1",

    # ログ・モニタリング
    "loguru>=0.7.2",
    "tqdm>=4.65.0",

    # データ品質
    "pandera>=0.17.0",

    # その他
    "jpholiday>=0.1.9",
    "yfinance>=0.2.44",
    "psutil>=5.9.0",
]

[project.optional-dependencies]
gpu = [
    "cupy-cuda12x>=13.0.0",
    "rmm-cu12>=24.10.0",
    "cudf-cu12>=24.10.0",  # RAPIDS GPU ETL
]

storage = [
    "s3fs>=2023.12.0",
    "google-cloud-storage>=2.10.0",
]

dev = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.21.0",
    "ruff>=0.1.0",
    "mypy>=1.5.0",
]
```

### 1.3 Makefile

```makefile
# data/Makefile

.PHONY: build build-bg clean cache-verify help

# デフォルト期間（5年分）
START_DATE ?= 2020-09-06
END_DATE ?= 2025-09-06

# ===== Layer 1: ユーザーフレンドリー =====

# データセット生成（フォアグラウンド）
build:
	python scripts/build.py \
		--start-date $(START_DATE) \
		--end-date $(END_DATE) \
		--use-gpu-etl \
		--enable-cache

# バックグラウンド実行（SSH安全）
build-bg:
	nohup python scripts/build.py \
		--start-date $(START_DATE) \
		--end-date $(END_DATE) \
		--use-gpu-etl \
		--enable-cache \
		> logs/build_$(shell date +%Y%m%d_%H%M%S).log 2>&1 &
	@echo "Dataset build started in background"
	@echo "Check logs: tail -f logs/build_*.log"

# クイック検証（1ヶ月）
build-quick:
	python scripts/build.py \
		--start-date 2025-08-01 \
		--end-date 2025-08-31 \
		--use-gpu-etl \
		--enable-cache

# ===== Layer 2: 詳細制御 =====

# GPU ETL有効
build-gpu:
	python scripts/build.py \
		--start-date $(START_DATE) \
		--end-date $(END_DATE) \
		--use-gpu-etl \
		--enable-cache

# CPU専用（GPU不要環境）
build-cpu:
	python scripts/build.py \
		--start-date $(START_DATE) \
		--end-date $(END_DATE) \
		--enable-cache

# ===== Layer 3: ユーティリティ =====

# キャッシュ検証
cache-verify:
	@echo "=== Cache Verification ==="
	@echo "1. Checking USE_CACHE in .env"
	@grep USE_CACHE .env || echo "⚠️  USE_CACHE not set"
	@echo ""
	@echo "2. Checking cache directories"
	@ls -lh output/raw/prices/ 2>/dev/null || echo "⚠️  No price cache"
	@echo ""
	@echo "3. Cache statistics"
	@python scripts/update_cache.py --stats

# キャッシュ統計
cache-stats:
	@python scripts/update_cache.py --stats

# キャッシュクリーン
cache-clean:
	@echo "⚠️  This will delete all cache files"
	@read -p "Continue? (y/N): " confirm && [ "$$confirm" = "y" ]
	rm -rf output/raw/*
	@echo "✅ Cache cleaned"

# クリーンアップ（データセット削除）
clean:
	rm -rf output/ml_dataset_*.parquet
	@echo "✅ Datasets cleaned"

# 完全クリーン（キャッシュ含む）
clean-all: clean cache-clean

# ヘルプ
help:
	@echo "Dataset Generation Commands"
	@echo ""
	@echo "Layer 1 (User-Friendly):"
	@echo "  make build              Generate dataset (foreground)"
	@echo "  make build-bg           Generate dataset (background, SSH-safe)"
	@echo "  make build-quick        Quick validation (1 month)"
	@echo ""
	@echo "Layer 2 (Detailed Control):"
	@echo "  make build-gpu          GPU ETL enabled"
	@echo "  make build-cpu          CPU only"
	@echo ""
	@echo "Layer 3 (Utilities):"
	@echo "  make cache-verify       Verify cache configuration"
	@echo "  make cache-stats        Show cache statistics"
	@echo "  make cache-clean        Clean cache files"
	@echo "  make clean              Clean datasets"
	@echo "  make clean-all          Clean datasets + cache"
	@echo ""
	@echo "Customization:"
	@echo "  make build START_DATE=2024-01-01 END_DATE=2024-12-31"
```

### 1.4 .env.example

```bash
# ===== JQuants API =====
JQUANTS_AUTH_EMAIL=your_email@example.com
JQUANTS_AUTH_PASSWORD=your_password
JQUANTS_PLAN_TIER=standard  # or premium

# ===== API Performance =====
MAX_CONCURRENT_FETCH=75      # 並列リクエスト数
MAX_PARALLEL_WORKERS=20       # CPU並列処理

# ===== GPU Settings =====
USE_GPU_ETL=1                # GPU-accelerated ETL (RAPIDS)
RMM_POOL_SIZE=40GB           # GPU memory pool (A100 80GBの安定設定)
CUDA_VISIBLE_DEVICES=0       # GPU device selection

# ===== Cache Settings (重要！) =====
USE_CACHE=1                  # 価格データキャッシュを有効化
CACHE_MAX_AGE_DAYS=7         # キャッシュ有効期限（日）
MIN_CACHE_COVERAGE=0.3       # 部分一致キャッシュの最小カバレッジ
ENABLE_MULTI_CACHE=1         # 複数キャッシュファイル結合を有効化

# ===== GCS Settings =====
GCS_ENABLED=0                # GCS連携を無効化（デフォルト）
GCS_BUCKET=gogooku-ml-data
GCS_SYNC_AFTER_SAVE=1        # 保存後に自動同期

# ===== PyTorch Thread Settings =====
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # GPU memory fragmentation fix
```

### 1.5 ターゲット/ラベル生成の移植（重要）

**問題**: 特徴量の移植は計画されているが、ターゲット変数（forward returns）の生成ロジックが欠落している。

#### ターゲット列の定義

```python
# data/src/builder/pipelines/target_generator.py

TARGET_COLUMNS = {
    "returns_1d": "1日後リターン",
    "returns_5d": "5日後リターン",
    "returns_10d": "10日後リターン",
    "returns_20d": "20日後リターン",
}

def generate_targets(df: pl.DataFrame) -> pl.DataFrame:
    """
    Forward returns を生成

    Args:
        df: 価格データを含むDataFrame（Code, Date, Close列必須）

    Returns:
        ターゲット列が追加されたDataFrame
    """
    return (
        df
        .sort(["Code", "Date"])
        .with_columns([
            # 1日後リターン
            (pl.col("Close").shift(-1).over("Code") / pl.col("Close") - 1).alias("returns_1d"),
            # 5日後リターン
            (pl.col("Close").shift(-5).over("Code") / pl.col("Close") - 1).alias("returns_5d"),
            # 10日後リターン
            (pl.col("Close").shift(-10).over("Code") / pl.col("Close") - 1).alias("returns_10d"),
            # 20日後リターン
            (pl.col("Close").shift(-20).over("Code") / pl.col("Close") - 1).alias("returns_20d"),
        ])
    )
```

#### ターゲット検証テスト

```python
# data/tests/unit/test_target_generator.py

import pytest
import polars as pl
from builder.pipelines.target_generator import generate_targets

def test_target_generation():
    """ターゲット生成のロジックテスト"""
    # サンプルデータ作成
    df = pl.DataFrame({
        "Code": ["1001"] * 30,
        "Date": pl.date_range(
            pl.date(2024, 1, 1),
            pl.date(2024, 1, 30),
            "1d",
            eager=True
        ),
        "Close": [100.0 + i for i in range(30)],  # 単調増加
    })

    result = generate_targets(df)

    # ターゲット列が存在する
    assert "returns_1d" in result.columns
    assert "returns_5d" in result.columns
    assert "returns_10d" in result.columns
    assert "returns_20d" in result.columns

    # 最終20日はNaN（未来データなし）
    assert result.tail(20).select("returns_20d").null_count().item() == 20

    # リターンが正しく計算されている
    # Close[1] = 101, Close[0] = 100 → returns_1d[0] = 0.01
    assert abs(result[0, "returns_1d"] - 0.01) < 1e-6

def test_target_distribution():
    """ターゲット分布の妥当性チェック"""
    df = generate_targets(create_realistic_data())

    for horizon in ["1d", "5d", "10d", "20d"]:
        col = f"returns_{horizon}"

        # 分布が存在する
        assert df.select(col).drop_nulls().height > 0

        # 外れ値チェック（±50%を超えるリターンは稀）
        outliers = df.filter(pl.col(col).abs() > 0.5).height
        assert outliers / df.height < 0.01  # 1%未満
```

#### gogooku3とのパリティテスト

```python
# data/tests/integration/test_target_parity.py

import os
import pytest
import polars as pl
from pathlib import Path

def _resolve_gogooku3_dataset() -> Path:
    """gogooku3データセットパスを環境変数から解決"""
    if path_str := os.getenv("GOGOOKU3_DATASET_PATH"):
        return Path(path_str)

    # デフォルトフォールバック
    default_path = Path("/workspace/gogooku3/output/ml_dataset_latest.parquet")
    if default_path.exists():
        return default_path

    pytest.skip("GOGOOKU3_DATASET_PATH not set and default path not found")

def _resolve_gogooku5_dataset() -> Path:
    """gogooku5データセットパスを環境変数から解決"""
    if path_str := os.getenv("GOGOOKU5_DATASET_PATH"):
        return Path(path_str)

    # プロジェクトルートから相対パス探索
    project_root = Path(__file__).resolve().parents[3]  # data/tests/integration/ から3階層上
    data_output = project_root / "data" / "output"

    # バージョン付きファイルを優先（最新版）
    versioned_files = sorted(
        data_output.glob("ml_dataset_v*.parquet"),
        reverse=True
    )
    if versioned_files:
        return versioned_files[0]

    # フォールバック: シンボリックリンク
    symlink = data_output / "ml_dataset_latest.parquet"
    if symlink.exists():
        return symlink

    pytest.skip(f"gogooku5 dataset not found in {data_output}")

@pytest.mark.integration
def test_target_parity_with_gogooku3():
    """gogooku3との分布パリティチェック"""
    # データセット読み込み（環境変数ベース）
    gogooku3_path = _resolve_gogooku3_dataset()
    gogooku5_path = _resolve_gogooku5_dataset()

    gogooku3_df = pl.read_parquet(gogooku3_path)
    gogooku5_df = pl.read_parquet(gogooku5_path)

    # 同じ期間・銘柄のサブセットを抽出
    common_codes = set(gogooku3_df["Code"].unique()).intersection(
        set(gogooku5_df["Code"].unique())
    )
    common_dates = set(gogooku3_df["Date"].unique()).intersection(
        set(gogooku5_df["Date"].unique())
    )

    g3_subset = gogooku3_df.filter(
        pl.col("Code").is_in(common_codes) & pl.col("Date").is_in(common_dates)
    )
    g5_subset = gogooku5_df.filter(
        pl.col("Code").is_in(common_codes) & pl.col("Date").is_in(common_dates)
    )

    # 分布統計を比較
    for horizon in ["1d", "5d", "10d", "20d"]:
        col = f"returns_{horizon}"

        g3_stats = g3_subset.select(col).drop_nulls().describe()
        g5_stats = g5_subset.select(col).drop_nulls().describe()

        # 平均が±5%以内で一致
        g3_mean = g3_stats.filter(pl.col("statistic") == "mean")[col].item()
        g5_mean = g5_stats.filter(pl.col("statistic") == "mean")[col].item()
        assert abs(g3_mean - g5_mean) / abs(g3_mean) < 0.05

        # 標準偏差が±5%以内で一致
        g3_std = g3_stats.filter(pl.col("statistic") == "std")[col].item()
        g5_std = g5_stats.filter(pl.col("statistic") == "std")[col].item()
        assert abs(g3_std - g5_std) / abs(g3_std) < 0.05
```

#### 移植タスク

**優先度S**:
1. `scripts/data/ml_dataset_builder.py` の `generate_targets()` 関数 → `data/src/builder/pipelines/target_generator.py`
2. ターゲット検証ロジック → `data/src/builder/pipelines/target_validator.py`
3. ユニットテスト作成: `test_target_generator.py`
4. パリティテスト作成: `test_target_parity.py`

**受け入れ基準**:
- ✅ 4つのターゲット列が正しく生成される
- ✅ gogooku3との分布差異が±5%以内
- ✅ ユニットテストカバレッジ > 90%
- ✅ 統合テストでパリティチェックが成功

---

### 1.6 移植ファイルリスト（優先度順）

#### 優先度S（必須・即時移植）- 27ファイル（+2ターゲット関連）

**設定・ユーティリティ（6ファイル）**:
1. `src/gogooku3/utils/settings.py` → `data/src/builder/utils/env.py`
2. `src/gogooku3/utils/gcs_storage.py` → `data/src/builder/utils/storage.py`
3. `src/gogooku3/config/dataset_config.py` → `data/src/builder/config/settings.py`
4. `.env.example` → `data/.env.example`
5. 新規作成: `data/src/builder/utils/cache.py`
6. 新規作成: `data/src/builder/utils/logger.py`

**API & データアクセス（2ファイル）**:
7. `src/gogooku3/components/jquants_async_fetcher.py` → `data/src/builder/api/jquants_fetcher.py`
8. `src/gogooku3/components/margin_fetcher.py` → `data/src/builder/api/margin_fetcher.py`

**データ取得コンポーネント（8ファイル）**:
9. `scripts/components/trading_calendar_fetcher.py` → `data/src/builder/api/calendar_fetcher.py`
10. `scripts/components/market_code_filter.py` → `data/src/builder/api/market_filter.py`
11. `scripts/components/axis_decider.py` → `data/src/builder/api/axis_decider.py`
12. `scripts/components/daily_quotes_by_code.py` → `data/src/builder/api/quotes_fetcher.py`
13. `scripts/components/listed_info_manager.py` → `data/src/builder/api/listed_manager.py`
14. `scripts/components/event_detector.py` → `data/src/builder/api/event_detector.py`
15. `scripts/components/daily_stock_fetcher.py` → `data/src/builder/api/stock_fetcher.py`
16. `scripts/components/modular_updater.py` → `data/src/builder/api/modular_updater.py`

**パイプライン（3ファイル）**:
17. `scripts/pipelines/run_full_dataset.py` → `data/scripts/build.py`
18. `scripts/pipelines/run_pipeline_v4_optimized.py` → `data/src/builder/pipelines/optimized_pipeline.py`
19. `src/pipeline/full_dataset.py` → `data/src/builder/pipelines/full_pipeline.py`

**データ構築（1ファイル）**:
20. `scripts/data/ml_dataset_builder.py` → `data/src/builder/pipelines/dataset_builder.py`

**ターゲット生成（2ファイル）**:
21. 新規作成: `data/src/builder/pipelines/target_generator.py`
22. 新規作成: `data/src/builder/pipelines/target_validator.py`

**レガシー特徴量（4ファイル + マクロディレクトリ）**:
23. `src/features/calendar_utils.py` → `data/src/builder/features/legacy/calendar_utils.py`
24. `src/features/market_features.py` → `data/src/builder/features/legacy/market_features.py`
25. `src/features/safe_joiner_v2.py` → `data/src/builder/features/legacy/safe_joiner_v2.py`
26. `src/features/flow_features_v2.py` → `data/src/builder/features/legacy/flow_features_v2.py`
27. `src/features/macro/` (ディレクトリ全体) → `data/src/builder/features/macro/`

**Makefile（2ファイル）**:
28. `Makefile.dataset` (318行) → `data/Makefile`
29. `Makefile` (dataset関連部分のみ抽出) → `gogooku5/Makefile`

**テスト基盤（3ファイル）**:
30. 新規作成: `data/tests/conftest.py`
31. 新規作成: `data/tests/unit/test_target_generator.py`
32. 新規作成: `data/tests/integration/test_target_parity.py`

#### 優先度A（重要・早期移植）- 15ファイル

**コア特徴量（10ファイル）**:
26. `src/gogooku3/features/quality_features_polars.py`
27. `src/gogooku3/features/index_features.py`
28. `src/gogooku3/features/index_option.py`
29. `src/gogooku3/features/margin_daily.py`
30. `src/gogooku3/features/margin_weekly.py`
31. `src/gogooku3/features/short_selling.py`
32. `src/gogooku3/features/short_selling_sector.py`
33. `src/gogooku3/features/sector_aggregation.py`
34. `src/gogooku3/features/sector_cross_sectional.py`
35. `src/gogooku3/features/advanced_features.py`

**データ処理補助（5ファイル）**:
36. `src/features/quality_features.py`
37. `src/features/peer_features.py`
38. `src/features/feature_validator.py`
39. `src/features/code_normalizer.py`
40. `src/features/validity_flags.py`

#### 優先度B（補完・段階的移植）- 10ファイル

**高度な特徴量（10ファイル）**:
41. `src/gogooku3/features/earnings_events.py`
42. `src/gogooku3/features/earnings_features.py`
43. `src/gogooku3/features/graph_features_gpu.py`
44. `src/gogooku3/features/graph_features.py`
45. `src/gogooku3/features/advanced_volatility.py`
46. `src/gogooku3/features/option_sentiment_features.py`
47. `src/gogooku3/features/tech_indicators.py`
48. `src/gogooku3/features/listed_features.py`
49. `src/gogooku3/features/enhanced_flow_features.py`
50. `src/gogooku3/features/futures_features.py` (Premium契約後)

---

### 1.7 データコントラクト/スキーマ定義（重要）

**問題**: ~307特徴量の具体的な名前リスト、型定義、バージョニングが欠落している。

#### スキーマ管理戦略

```
data/
├── schema/
│   ├── v1/
│   │   ├── dataset_schema.py          # Pandera DataFrameModel
│   │   ├── FEATURE_NAMES.yaml         # 全特徴量リスト
│   │   └── DATA_DICTIONARY.md         # 説明書
│   └── v2/                            # 将来のスキーマ変更用
└── output/
    └── ml_dataset_v1.0.0.parquet      # バージョン付きファイル名
```

#### Panderaスキーマ定義

```python
# data/schema/v1/dataset_schema.py

import pandera as pa
import pandera.polars as pap
from pandera import Column, DataFrameSchema
import polars as pl

class MLDatasetSchemaV1(pap.DataFrameModel):
    """
    MLデータセット v1.0.0 のスキーマ定義

    バージョン: 1.0.0
    リリース日: 2025-XX-XX
    特徴量数: 307 (ターゲット4列を除く)
    """

    # ===== 基本列 =====
    Date: pl.Date = pap.Field(nullable=False, description="取引日")
    Code: str = pap.Field(nullable=False, str_length={"min_value": 4, "max_value": 4}, description="銘柄コード")

    # ===== ターゲット列 =====
    returns_1d: float = pap.Field(nullable=True, ge=-1.0, le=10.0, description="1日後リターン")
    returns_5d: float = pap.Field(nullable=True, ge=-1.0, le=10.0, description="5日後リターン")
    returns_10d: float = pap.Field(nullable=True, ge=-1.0, le=10.0, description="10日後リターン")
    returns_20d: float = pap.Field(nullable=True, ge=-1.0, le=10.0, description="20日後リターン")

    # ===== 価格特徴量 (5列) =====
    Close: float = pap.Field(nullable=False, gt=0, description="終値")
    Open: float = pap.Field(nullable=True, gt=0, description="始値")
    High: float = pap.Field(nullable=True, gt=0, description="高値")
    Low: float = pap.Field(nullable=True, gt=0, description="安値")
    Volume: float = pap.Field(nullable=True, ge=0, description="出来高")

    # ===== ファンダメンタル特徴量 (50列程度) =====
    # 例: PER, PBR, ROE, 売上高成長率 etc.
    # （実際の特徴量名は FEATURE_NAMES.yaml を参照）

    class Config:
        """スキーマ設定"""
        strict = True  # 未定義列を許可しない
        coerce = True  # 型変換を試みる

    @pa.dataframe_check
    def check_date_range(cls, df: pl.DataFrame) -> bool:
        """日付範囲の妥当性チェック"""
        min_date = df["Date"].min()
        max_date = df["Date"].max()
        # 2000年以降、未来日付でないこと
        return (
            min_date >= pl.date(2000, 1, 1) and
            max_date <= pl.date.today().add(days=1)
        )

    @pa.dataframe_check
    def check_no_duplicate_rows(cls, df: pl.DataFrame) -> bool:
        """(Date, Code) の重複チェック"""
        return df.select(["Date", "Code"]).is_duplicated().sum() == 0
```

#### 特徴量名リスト (FEATURE_NAMES.yaml)

```yaml
# data/schema/v1/FEATURE_NAMES.yaml

version: "1.0.0"
created: "2025-XX-XX"
description: "gogooku5 MLデータセット v1の全特徴量リスト"

core_columns:
  - Date
  - Code

target_columns:
  - returns_1d
  - returns_5d
  - returns_10d
  - returns_20d

feature_groups:
  price_features:
    count: 5
    columns:
      - Close
      - Open
      - High
      - Low
      - Volume

  fundamental_features:
    count: 50
    columns:
      - per  # PER
      - pbr  # PBR
      - roe  # ROE
      - roa  # ROA
      # ... (実際の特徴量名を列挙)

  technical_indicators:
    count: 40
    columns:
      - sma_5
      - sma_20
      - ema_12
      - rsi_14
      # ...

  index_features:
    count: 15
    columns:
      - topix_close
      - topix_volume
      - nk225_close
      # ...

  margin_features:
    count: 20
    columns:
      - margin_buy_volume
      - margin_sell_volume
      - short_sell_ratio
      # ...

  sector_features:
    count: 30
    columns:
      - sector_rank
      - sector_momentum
      # ...

  # ... (他のグループ)

total_features: 307  # ターゲット列を除く
```

#### データディクショナリ (DATA_DICTIONARY.md)

```markdown
# MLデータセット v1.0.0 データディクショナリ

## 基本情報

- **バージョン**: 1.0.0
- **リリース日**: 2025-XX-XX
- **特徴量数**: 307列（ターゲット4列を除く）
- **対象市場**: 日本株式市場（東証プライム・スタンダード・グロース）
- **期間**: 2015年〜現在

## 列定義

### コア列

| 列名 | 型 | Nullable | 説明 | 範囲 |
|------|-----|----------|------|------|
| Date | Date | False | 取引日 | 2000-01-01 〜 今日 |
| Code | String | False | 銘柄コード（4桁） | - |

### ターゲット列

| 列名 | 型 | Nullable | 説明 | 計算式 |
|------|-----|----------|------|--------|
| returns_1d | Float64 | True | 1日後リターン | (Close[t+1] / Close[t]) - 1 |
| returns_5d | Float64 | True | 5日後リターン | (Close[t+5] / Close[t]) - 1 |
| returns_10d | Float64 | True | 10日後リターン | (Close[t+10] / Close[t]) - 1 |
| returns_20d | Float64 | True | 20日後リターン | (Close[t+20] / Close[t]) - 1 |

### 価格特徴量

| 列名 | 型 | Nullable | 説明 | 単位 |
|------|-----|----------|------|------|
| Close | Float64 | False | 終値 | 円 |
| Open | Float64 | True | 始値 | 円 |
| High | Float64 | True | 高値 | 円 |
| Low | Float64 | True | 安値 | 円 |
| Volume | Float64 | True | 出来高 | 株 |

### ファンダメンタル特徴量

| 列名 | 型 | Nullable | 説明 | 計算式 |
|------|-----|----------|------|--------|
| per | Float64 | True | 株価収益率 | 株価 / EPS |
| pbr | Float64 | True | 株価純資産倍率 | 株価 / BPS |
| roe | Float64 | True | 自己資本利益率 | 純利益 / 自己資本 |
| ... | ... | ... | ... | ... |

(以下、全307特徴量の詳細定義)

## 欠損値処理

| グループ | 欠損値の意味 | 処理方法 |
|----------|--------------|----------|
| 価格データ | 市場休業日、上場廃止 | NULL維持（モデル側でマスク） |
| ファンダメンタル | 決算未発表、データ未提供 | NULL維持または forward fill |
| テクニカル指標 | 計算期間不足 | NULL維持 |
| ターゲット | 未来データなし（最終N日） | NULL維持（学習時に除外） |

## バージョン履歴

### v1.0.0 (2025-XX-XX)
- 初版リリース
- 307特徴量、4ターゲット列
- J-Quants Standard プラン対応

### v1.1.0 (予定)
- 予定: 先物特徴量追加（88-92列）※Premium契約後
```

#### スキーマバリデーションテスト

```python
# data/tests/unit/test_schema_validation.py

import pytest
import polars as pl
import pandera as pa
from builder.schema.v1.dataset_schema import MLDatasetSchemaV1

def test_schema_validation_success():
    """正常なデータセットのバリデーション"""
    df = create_valid_dataset()  # 正常データ作成

    # スキーマ検証（例外が発生しないこと）
    validated_df = MLDatasetSchemaV1.validate(df)
    assert validated_df.shape == df.shape

def test_schema_validation_missing_required_column():
    """必須列欠落時のエラー"""
    df = create_valid_dataset().drop("Date")

    with pytest.raises(pa.errors.SchemaError):
        MLDatasetSchemaV1.validate(df)

def test_schema_validation_invalid_code_format():
    """銘柄コード形式エラー"""
    df = create_valid_dataset()
    df = df.with_columns(pl.lit("12345").alias("Code"))  # 5桁（不正）

    with pytest.raises(pa.errors.SchemaError):
        MLDatasetSchemaV1.validate(df)

def test_schema_validation_duplicate_rows():
    """重複行エラー"""
    df = create_valid_dataset()
    df = pl.concat([df, df.head(1)])  # 最初の行を重複

    with pytest.raises(pa.errors.SchemaError):
        MLDatasetSchemaV1.validate(df)
```

#### セマンティックバージョニング

```python
# data/src/builder/config/version.py

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import os

@dataclass
class DatasetVersion:
    """データセットバージョン管理"""
    major: int  # 互換性のない変更
    minor: int  # 後方互換性のある機能追加
    patch: int  # 後方互換性のあるバグ修正

    def __str__(self) -> str:
        return f"v{self.major}.{self.minor}.{self.patch}"

    @classmethod
    def from_string(cls, version_str: str) -> "DatasetVersion":
        """'v1.2.3' から DatasetVersion を生成"""
        parts = version_str.lstrip("v").split(".")
        return cls(int(parts[0]), int(parts[1]), int(parts[2]))

    def to_filename(self, base_name: str = "ml_dataset") -> str:
        """バージョン付きファイル名生成"""
        return f"{base_name}_{self}.parquet"

    def create_symlink(self, output_dir: Path, base_name: str = "ml_dataset"):
        """
        バージョン付きファイルへのシンボリックリンクを作成

        ml_dataset_v1.0.0.parquet → ml_dataset_latest.parquet (symlink)

        これにより、バージョン管理と後方互換性の両立が可能
        """
        versioned_file = output_dir / self.to_filename(base_name)
        symlink = output_dir / f"{base_name}_latest.parquet"

        # 既存のシンボリックリンクを削除
        if symlink.is_symlink() or symlink.exists():
            symlink.unlink()

        # 新しいシンボリックリンクを作成
        symlink.symlink_to(versioned_file.name)  # 相対パス

# 現在のバージョン
CURRENT_VERSION = DatasetVersion(1, 0, 0)
```

#### ファイル配置戦略

```
data/output/
├── ml_dataset_v1.0.0.parquet      # バージョン付き（実体）
├── ml_dataset_latest.parquet      # シンボリックリンク → v1.0.0
└── ml_dataset.parquet             # レガシー互換（オプショナル）

利点:
- バージョン付きファイル: 履歴管理、再現性確保
- シンボリックリンク: 後方互換性、最新版への自動参照
- DatasetLoader: 自動的に最新バージョンを検出
```

#### 移植タスク

**優先度S（追加）**:
33. 新規作成: `data/schema/v1/dataset_schema.py`
34. 新規作成: `data/schema/v1/FEATURE_NAMES.yaml`
35. 新規作成: `data/schema/v1/DATA_DICTIONARY.md`
36. 新規作成: `data/src/builder/config/version.py`
37. 新規作成: `data/tests/unit/test_schema_validation.py`

**受け入れ基準**:
- ✅ Panderaスキーマがコンパイルエラーなし
- ✅ FEATURE_NAMES.yamlに全307特徴量がリストされている
- ✅ DATA_DICTIONARY.mdが完成
- ✅ スキーマバリデーションテストが成功
- ✅ 出力ファイル名にバージョンが含まれる（`ml_dataset_v1.0.0.parquet`）

---

### 1.8 依存関係の修正戦略

#### A. 循環参照の解消

**問題（gogooku3）**:
```python
# src/gogooku3/components/jquants_async_fetcher.py
from scripts.components.market_code_filter import MarketCodeFilter  # NG

# src/gogooku3/pipeline/builder.py
from scripts.data.ml_dataset_builder import MLDatasetBuilder  # NG
```

**解決策（gogooku5）**:
```python
# data/src/builder/api/jquants_fetcher.py
from builder.api.market_filter import MarketFilter  # OK

# data/src/builder/pipelines/full_pipeline.py
from builder.pipelines.dataset_builder import DatasetBuilder  # OK
```

#### B. インポートパスの統一

**移植時の置換ルール**:
```python
# Before (gogooku3)
from src.gogooku3.* → from builder.*
from scripts.components.* → from builder.api.*
from scripts.pipelines.* → from builder.pipelines.*
from src.features.* → from builder.features.legacy.*
from src.pipeline.* → from builder.pipelines.*
```

### 1.7 テスト戦略（TDD）

#### ユニットテスト（高速・独立）
```python
# tests/unit/test_jquants_fetcher.py

import pytest
from builder.api.jquants_fetcher import JQuantsFetcher

@pytest.mark.asyncio
async def test_authentication():
    """JQuants API認証テスト"""
    fetcher = JQuantsFetcher()
    token = await fetcher.authenticate()
    assert token is not None
    assert len(token) > 0

@pytest.mark.asyncio
async def test_daily_quotes_fetch():
    """日次株価取得テスト"""
    fetcher = JQuantsAsyncFetcher()
    df = await fetcher.get_daily_quotes(
        start_date="2024-10-01",
        end_date="2024-10-31"
    )
    assert df.height > 0
    assert "Date" in df.columns
    assert "Code" in df.columns
    assert "Close" in df.columns
```

#### 統合テスト（エンドツーエンド）
```python
# tests/integration/test_full_dataset.py

import pytest
import polars as pl
from pathlib import Path

@pytest.mark.integration
@pytest.mark.slow
def test_1month_dataset_generation():
    """1ヶ月分データセット生成テスト"""
    from builder.pipelines.full_pipeline import build_dataset

    output_path = build_dataset(
        start_date="2024-10-01",
        end_date="2024-10-31",
        use_gpu_etl=True,
        enable_cache=True
    )

    # 基本検証
    assert Path(output_path).exists()
    df = pl.read_parquet(output_path)

    assert df.height > 0, "Empty dataset"
    assert "Date" in df.columns
    assert "Code" in df.columns
    assert "Close" in df.columns

    # 日付範囲検証
    assert df["Date"].min() >= pl.lit("2024-10-01").cast(pl.Date)
    assert df["Date"].max() <= pl.lit("2024-10-31").cast(pl.Date)

    # 特徴量数検証（~100-150を期待）
    assert len(df.columns) >= 100
```

#### モック/フィクスチャ戦略（ライブAPI回避）

**問題**: 現在のテストコードはライブAPIを直接呼び出しており、CI環境でフレイキーになる。

**解決策**: pytest-mockを使用してAPIレスポンスをモック化し、事前定義されたフィクスチャで高速・安定したテストを実現。

```python
# data/tests/conftest.py

import pytest
import polars as pl
from pathlib import Path
from unittest.mock import AsyncMock

@pytest.fixture
def mock_jquants_api_response():
    """JQuants API のモックレスポンス（日次株価）"""
    return {
        "daily_quotes": [
            {
                "Date": "2024-10-01",
                "Code": "86970",
                "Open": 1500.0,
                "High": 1550.0,
                "Low": 1480.0,
                "Close": 1520.0,
                "Volume": 1000000.0,
            },
            {
                "Date": "2024-10-02",
                "Code": "86970",
                "Open": 1520.0,
                "High": 1580.0,
                "Low": 1510.0,
                "Close": 1560.0,
                "Volume": 1200000.0,
            },
        ]
    }

@pytest.fixture
def mock_daily_quotes_df():
    """日次株価のPolars DataFrameフィクスチャ"""
    return pl.DataFrame({
        "Date": ["2024-10-01", "2024-10-02"] * 100,
        "Code": ["86970"] * 100 + ["13010"] * 100,
        "Open": [1500.0, 1520.0] * 100,
        "High": [1550.0, 1580.0] * 100,
        "Low": [1480.0, 1510.0] * 100,
        "Close": [1520.0, 1560.0] * 100,
        "Volume": [1000000.0, 1200000.0] * 100,
    }).with_columns(pl.col("Date").str.to_date())

@pytest.fixture
async def mock_jquants_fetcher(monkeypatch, mock_daily_quotes_df):
    """JQuantsFetcher全体をモック化"""
    from builder.api.jquants_fetcher import JQuantsAsyncFetcher

    async def mock_authenticate():
        return "mock_id_token_12345"

    async def mock_get_daily_quotes(start_date, end_date, codes=None):
        return mock_daily_quotes_df

    monkeypatch.setattr(JQuantsAsyncFetcher, "authenticate", mock_authenticate)
    monkeypatch.setattr(JQuantsAsyncFetcher, "get_daily_quotes", mock_get_daily_quotes)

    return JQuantsAsyncFetcher()

# data/tests/unit/test_jquants_fetcher_mocked.py

import pytest
import polars as pl
from builder.api.jquants_fetcher import JQuantsAsyncFetcher

@pytest.mark.asyncio
async def test_authentication_mocked(mock_jquants_fetcher):
    """認証テスト（モック化）"""
    token = await mock_jquants_fetcher.authenticate()
    assert token == "mock_id_token_12345"

@pytest.mark.asyncio
async def test_daily_quotes_fetch_mocked(mock_jquants_fetcher, mock_daily_quotes_df):
    """日次株価取得テスト（モック化）"""
    df = await mock_jquants_fetcher.get_daily_quotes(
        start_date="2024-10-01",
        end_date="2024-10-02"
    )

    assert df.height == mock_daily_quotes_df.height
    assert "Date" in df.columns
    assert "Code" in df.columns
    assert "Close" in df.columns
    assert df["Code"].unique().to_list() == ["86970", "13010"]

# data/tests/unit/test_feature_generator_mocked.py

import pytest
import polars as pl
from builder.features.price_features import PriceFeatureGenerator

def test_price_features_with_fixture(mock_daily_quotes_df):
    """価格特徴量生成テスト（フィクスチャ使用）"""
    generator = PriceFeatureGenerator()
    result = generator.generate(mock_daily_quotes_df)

    # 基本列の存在確認
    assert "returns_1d" in result.columns
    assert "volatility_5d" in result.columns

    # 値の妥当性検証
    assert result["returns_1d"].drop_nulls().abs().max() < 1.0  # リターン < 100%
```

#### ライブAPI統合テスト（オプトイン）

ライブAPIテストは環境変数で明示的にオプトインする設計：

```python
# data/tests/integration/test_live_api.py

import os
import pytest
import polars as pl
from builder.api.jquants_fetcher import JQuantsAsyncFetcher

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LIVE_API_TESTS") != "1",
    reason="Live API tests disabled (set RUN_LIVE_API_TESTS=1 to enable)"
)

@pytest.mark.live
@pytest.mark.asyncio
async def test_live_authentication():
    """【ライブAPI】認証テスト"""
    fetcher = JQuantsAsyncFetcher()
    token = await fetcher.authenticate()
    assert token is not None
    assert len(token) > 20  # JWTトークンは通常長い

@pytest.mark.live
@pytest.mark.asyncio
async def test_live_daily_quotes_fetch():
    """【ライブAPI】日次株価取得テスト"""
    fetcher = JQuantsAsyncFetcher()
    df = await fetcher.get_daily_quotes(
        start_date="2024-10-01",
        end_date="2024-10-03"
    )
    assert df.height > 0
    assert "Date" in df.columns
    assert "Code" in df.columns
    assert df["Date"].min() >= pl.lit("2024-10-01").cast(pl.Date)
```

**CI設定例（GitHub Actions）**:
```yaml
# .github/workflows/test.yml

- name: Run unit tests (mocked, fast)
  run: pytest tests/unit/ -v

- name: Run integration tests (no live API)
  run: pytest tests/integration/ -v -m "not live"

- name: Run live API tests (manual trigger only)
  if: github.event_name == 'workflow_dispatch'
  env:
    RUN_LIVE_API_TESTS: 1
    JQUANTS_AUTH_EMAIL: ${{ secrets.JQUANTS_EMAIL }}
    JQUANTS_AUTH_PASSWORD: ${{ secrets.JQUANTS_PASSWORD }}
  run: pytest tests/integration/ -v -m live
```

**メリット**:
- ✅ 通常のCI実行では高速・安定（モックのみ）
- ✅ ライブAPIテストは手動トリガーでのみ実行
- ✅ 本番環境デプロイ前に実APIを検証可能
- ✅ API制限・レート制限の影響を受けない

---

## Phase 2: ATFT-GAT-FANモデル（1-2週間）

### 目標
ATFT-GAT-FANモデルを独立したモジュールとして実装。共通データセット（`data/output/ml_dataset_latest.parquet`）を参照。

### 2.1 ディレクトリ構造（models/atft_gat_fan/）

```
models/atft_gat_fan/
├── src/atft_gat_fan/
│   ├── __init__.py
│   │
│   ├── models/                            # モデル実装
│   │   ├── __init__.py
│   │   ├── atft_gat_fan.py               # ATFT-GAT-FANメインモデル
│   │   ├── gat_layer.py                  # Graph Attention Layer
│   │   ├── fan_layer.py                  # Feature Attention Network
│   │   ├── vsn.py                        # Value-Semantic Network
│   │   └── san.py                        # Semantic-Attention Network
│   │
│   ├── training/                          # 学習パイプライン
│   │   ├── __init__.py
│   │   ├── safe_training_pipeline.py     # SafeTrainingPipeline
│   │   ├── data_module.py                # LightningDataModule
│   │   ├── loss_functions.py             # カスタム損失関数
│   │   ├── callbacks.py                  # 学習コールバック
│   │   └── phase_trainer.py              # フェーズベース学習
│   │
│   ├── data/                              # データローダー
│   │   ├── __init__.py
│   │   ├── dataset_loader.py             # 共通データセット参照
│   │   ├── production_dataset_v3.py      # ProductionDatasetV3
│   │   ├── normalizer_v2.py              # CrossSectionalNormalizerV2
│   │   ├── splitter_v2.py                # WalkForwardSplitterV2
│   │   └── sampler.py                    # WeightedTimeSampler
│   │
│   └── config/                            # 設定管理
│       ├── __init__.py
│       └── train_config.py               # Pydantic学習設定
│
├── scripts/                               # CLI エントリーポイント
│   ├── train_atft.py                     # 学習メインスクリプト
│   ├── evaluate_model.py                 # モデル評価
│   └── predict.py                        # 予測スクリプト
│
├── configs/                               # Hydra設定
│   └── atft/
│       ├── config_production.yaml        # 本番設定
│       ├── feature_categories.yaml       # 特徴量カテゴリ定義
│       └── train/
│           └── production_improved.yaml
│
├── tests/                                 # テスト
│   ├── unit/
│   │   ├── test_model.py
│   │   ├── test_loss_functions.py
│   │   └── test_normalizer.py
│   └── integration/
│       └── test_training_pipeline.py
│
├── output/                                # 学習結果
│   ├── models/                           # 保存モデル
│   └── logs/                             # 学習ログ
│
├── pyproject.toml                         # ATFT-GAT-FAN専用依存
├── Makefile.train                         # 学習コマンド
├── README.md                              # モデル説明
└── CLAUDE.md                              # Claude Code用ガイド
```

### 2.2 共通データセット参照パターン

```python
# models/atft_gat_fan/src/atft_gat_fan/data/dataset_loader.py

import os
import re
import polars as pl
from pathlib import Path
from typing import Optional

class DatasetLoader:
    """共有データセットへのアクセスレイヤー"""

    @classmethod
    def _resolve_dataset_path(cls) -> Path:
        """
        データセットパスを解決（環境変数 > 設定ファイル > デフォルト）

        バージョン付きファイル名とシンボリックリンクの両方をサポート
        """
        # 1. 環境変数から取得
        if dataset_path := os.getenv("GOGOOKU5_DATASET_PATH"):
            return Path(dataset_path)

        # 2. プロジェクトルート検出（pyproject.toml探索）
        current = Path(__file__).resolve()
        for parent in [current] + list(current.parents):
            if (parent / "pyproject.toml").exists():
                data_output = parent / "data" / "output"

                # バージョン付きファイルを優先（最新版）
                versioned_files = sorted(
                    data_output.glob("ml_dataset_v*.parquet"),
                    reverse=True
                )
                if versioned_files:
                    return versioned_files[0]  # 最新バージョン

                # フォールバック: シンボリックリンク
                symlink = data_output / "ml_dataset_latest.parquet"
                if symlink.exists():
                    return symlink

                # フォールバック: バージョンなし（開発環境）
                legacy = data_output / "ml_dataset.parquet"
                if legacy.exists():
                    return legacy

                break

        # 3. 最終フォールバック（エラーメッセージ用）
        raise FileNotFoundError(
            "Dataset not found. Please:\n"
            "1. Build dataset: cd data && make build\n"
            "2. Or set GOGOOKU5_DATASET_PATH environment variable"
        )

    @classmethod
    def load(cls, dataset_path: Optional[Path] = None) -> pl.DataFrame:
        """
        データセット読み込み

        Args:
            dataset_path: カスタムデータセットパス（指定しない場合は自動解決）

        Returns:
            pl.DataFrame: ロードされたデータセット

        Raises:
            FileNotFoundError: データセットが存在しない場合
        """
        path = dataset_path or cls._resolve_dataset_path()
        return pl.read_parquet(path)

    @classmethod
    def check_dataset_exists(cls) -> bool:
        """データセットの存在確認"""
        try:
            cls._resolve_dataset_path()
            return True
        except FileNotFoundError:
            return False

    @classmethod
    def get_dataset_version(cls) -> Optional[str]:
        """
        データセットバージョンを取得

        Returns:
            バージョン文字列（例: "v1.0.0"）、バージョンなしの場合None
        """
        try:
            path = cls._resolve_dataset_path()
            # ファイル名からバージョンを抽出
            # ml_dataset_v1.0.0.parquet → v1.0.0
            match = re.search(r"_v(\d+\.\d+\.\d+)\.parquet$", path.name)
            return f"v{match.group(1)}" if match else None
        except FileNotFoundError:
            return None
```

### 2.3 pyproject.toml（ATFT-GAT-FAN専用）

```toml
[project]
name = "atft-gat-fan"
version = "0.1.0"
description = "ATFT-GAT-FAN: Advanced Financial ML Model"
requires-python = ">=3.10"

dependencies = [
    # 深層学習
    "torch>=2.0.0",
    "torch-geometric>=2.4.0",
    "lightning>=2.0.0",

    # データ処理（読み込みのみ）
    "polars>=0.20.0",
    "pyarrow>=14.0.0",
    "numpy>=1.24.0",

    # 機械学習
    "scikit-learn>=1.3.0",

    # 設定管理
    "hydra-core>=1.3.0",
    "pydantic>=2.5.0",
    "python-dotenv>=1.0.0",

    # ログ・モニタリング
    "loguru>=0.7.2",
    "tqdm>=4.65.0",
    "tensorboard>=2.14.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "ruff>=0.1.0",
    "mypy>=1.5.0",
]

# 注意: gogooku5-dataset は依存に含めない（ファイル参照のみ）
```

### 2.4 Makefile.train（ATFT-GAT-FAN）

```makefile
# models/atft_gat_fan/Makefile.train

.PHONY: train train-quick train-safe evaluate help

# デフォルト設定
EPOCHS ?= 120
BATCH_SIZE ?= 2048
LR ?= 5e-4
HIDDEN_SIZE ?= 256

# ===== Layer 1: ユーザーフレンドリー =====

# 最適化学習（バックグラウンド）
train:
	@echo "Checking dataset availability..."
	@python -c "from atft_gat_fan.data.dataset_loader import DatasetLoader; DatasetLoader.check_dataset_exists()" || \
		(echo "❌ Dataset not found. Run: cd ../../data && make build" && exit 1)
	@echo "✅ Dataset found. Starting training..."
	nohup python scripts/train_atft.py \
		--max-epochs $(EPOCHS) \
		--batch-size $(BATCH_SIZE) \
		--lr $(LR) \
		> output/logs/train_$(shell date +%Y%m%d_%H%M%S).log 2>&1 &
	@echo "Training started in background"
	@echo "Monitor: tail -f output/logs/train_*.log"

# クイック検証（3エポック、フォアグラウンド）
train-quick:
	python scripts/train_atft.py \
		--max-epochs 3 \
		--batch-size 512 \
		--lr 5e-4

# セーフモード（シングルワーカー）
train-safe:
	FORCE_SINGLE_PROCESS=1 python scripts/train_atft.py \
		--max-epochs $(EPOCHS) \
		--batch-size 256 \
		--lr $(LR)

# ===== Layer 2: 詳細制御 =====

# 最適化フル設定
train-optimized:
	ALLOW_UNSAFE_DATALOADER=1 \
	NUM_WORKERS=8 \
	USE_RANKIC=1 \
	python scripts/train_atft.py \
		--max-epochs $(EPOCHS) \
		--batch-size $(BATCH_SIZE) \
		--lr $(LR)

# 標準設定（保守的）
train-standard:
	python scripts/train_atft.py \
		--max-epochs $(EPOCHS) \
		--batch-size 1024 \
		--lr 3e-4

# ===== Layer 3: ユーティリティ =====

# モデル評価
evaluate:
	python scripts/evaluate_model.py \
		--model-path output/models/best_model.pt

# 学習状態確認
train-status:
	@ps aux | grep train_atft | grep -v grep || echo "No training process"

# ヘルプ
help:
	@echo "ATFT-GAT-FAN Training Commands"
	@echo ""
	@echo "Layer 1 (User-Friendly):"
	@echo "  make train              Optimized training (background)"
	@echo "  make train-quick        Quick validation (3 epochs)"
	@echo "  make train-safe         Safe mode (single-worker)"
	@echo ""
	@echo "Layer 2 (Detailed Control):"
	@echo "  make train-optimized    Full optimization stack"
	@echo "  make train-standard     Conservative settings"
	@echo ""
	@echo "Layer 3 (Utilities):"
	@echo "  make evaluate           Evaluate trained model"
	@echo "  make train-status       Check training status"
	@echo ""
	@echo "Customization:"
	@echo "  make train EPOCHS=75 BATCH_SIZE=4096 LR=1e-4"
```

### 2.5 移植ファイルリスト（優先度順）

#### 優先度S（必須）- 15ファイル

**モデル実装（5ファイル）**:
1. `src/atft_gat_fan/models/architectures/atft_gat_fan.py` → `models/atft_gat_fan/src/atft_gat_fan/models/atft_gat_fan.py`
2. `src/atft_gat_fan/models/layers/gat_layer.py` → `models/atft_gat_fan/src/atft_gat_fan/models/gat_layer.py`
3. `src/atft_gat_fan/models/layers/fan_layer.py` → `models/atft_gat_fan/src/atft_gat_fan/models/fan_layer.py`
4. `src/atft_gat_fan/models/layers/vsn.py` → `models/atft_gat_fan/src/atft_gat_fan/models/vsn.py`
5. `src/atft_gat_fan/models/layers/san.py` → `models/atft_gat_fan/src/atft_gat_fan/models/san.py`

**学習パイプライン（5ファイル）**:
6. `src/gogooku3/training/safe_training_pipeline.py` → `models/atft_gat_fan/src/atft_gat_fan/training/safe_training_pipeline.py`
7. `src/gogooku3/training/data_module.py` → `models/atft_gat_fan/src/atft_gat_fan/training/data_module.py`
8. `src/gogooku3/training/loss_functions.py` → `models/atft_gat_fan/src/atft_gat_fan/training/loss_functions.py`
9. `scripts/train_atft.py` → `models/atft_gat_fan/scripts/train_atft.py`
10. `scripts/integrated_ml_training_pipeline.py` の一部 → 統合

**データローダー（3ファイル）**:
11. `src/gogooku3/data/production_dataset_v3.py` → `models/atft_gat_fan/src/atft_gat_fan/data/production_dataset_v3.py`
12. `src/gogooku3/data/cross_sectional_normalizer_v2.py` → `models/atft_gat_fan/src/atft_gat_fan/data/normalizer_v2.py`
13. `src/gogooku3/data/walk_forward_splitter_v2.py` → `models/atft_gat_fan/src/atft_gat_fan/data/splitter_v2.py`

**新規作成（2ファイル）**:
14. 新規作成: `models/atft_gat_fan/src/atft_gat_fan/data/dataset_loader.py`（共通データセット参照）
15. 新規作成: `models/atft_gat_fan/src/atft_gat_fan/config/train_config.py`

---

## Phase 3: APEX-Rankerモデル（1週間）

### 目標
APEX-Rankerモデルを独立したモジュールとして実装。同じく共通データセットを参照。

### 3.1 ディレクトリ構造（models/apex_ranker/）

```
models/apex_ranker/
├── src/apex_ranker/
│   ├── __init__.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── apex_ranker_v0.py             # APEXRankerV0
│   │   └── patchtst.py                   # PatchTST実装
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py                    # 学習ロジック
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset_loader.py             # 共通データセット参照
│   │   ├── panel_dataset.py              # PanelDataset
│   │   └── feature_selector.py           # 特徴量選択
│   │
│   └── config/
│       ├── __init__.py
│       └── model_config.py
│
├── scripts/
│   ├── train_v0.py                       # 学習
│   ├── inference_v0.py                   # 推論
│   └── backtest_v0.py                    # バックテスト
│
├── configs/
│   ├── v0_base.yaml                      # ベース設定
│   └── v0_pruned.yaml                    # プルーニング版
│
├── tests/
│   ├── unit/
│   │   └── test_model.py
│   └── integration/
│       └── test_training.py
│
├── output/
│   ├── models/
│   └── logs/
│
├── pyproject.toml
├── Makefile.train
├── README.md
└── CLAUDE.md
```

### 3.2 移植ファイルリスト

#### 優先度S（必須）- 8ファイル

1. `apex-ranker/apex_ranker/models/apex_ranker_v0.py` → `models/apex_ranker/src/apex_ranker/models/apex_ranker_v0.py`
2. `apex-ranker/apex_ranker/models/patchtst.py` → `models/apex_ranker/src/apex_ranker/models/patchtst.py`
3. `apex-ranker/apex_ranker/data/panel_dataset.py` → `models/apex_ranker/src/apex_ranker/data/panel_dataset.py`
4. `apex-ranker/apex_ranker/data/feature_selector.py` → `models/apex_ranker/src/apex_ranker/data/feature_selector.py`
5. `apex-ranker/scripts/train_v0.py` → `models/apex_ranker/scripts/train_v0.py`
6. `apex-ranker/scripts/inference_v0.py` → `models/apex_ranker/scripts/inference_v0.py`
7. `apex-ranker/configs/v0_pruned.yaml` → `models/apex_ranker/configs/v0_pruned.yaml`
8. 新規作成: `models/apex_ranker/src/apex_ranker/data/dataset_loader.py`

---

## Phase 4: 共通ユーティリティ（オプショナル）

### 4.1 ディレクトリ構造（common/）

```
common/
├── src/common/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   └── base_loader.py                # データローダー基底クラス
│   │
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── financial.py                  # Sharpe, IC, RankIC
│   │   └── backtest.py                   # バックテスト指標
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logger.py                     # 統一ログ設定
│       └── config.py                     # 共通設定
│
├── tests/
│   └── test_metrics.py
│
├── pyproject.toml
└── README.md
```

### 4.2 使用例

```python
# models/atft_gat_fan/src/atft_gat_fan/training/loss_functions.py

from common.metrics import calculate_sharpe_ratio, calculate_rank_ic

class RankICLoss(nn.Module):
    def forward(self, predictions, targets):
        rank_ic = calculate_rank_ic(predictions, targets)  # 共通関数
        return -rank_ic
```

---

## Phase 5: Claude Code / Codex 連携

### 5.1 tools/claude-code.sh

```bash
#!/bin/bash
# tools/claude-code.sh - Enhanced Claude Code launcher with health check

echo "========================================="
echo "gogooku5 System Health Check"
echo "========================================="

# 1. データセット存在確認
echo "1. Dataset availability:"
DATASET_DIR="data/output"

# バージョン付きファイルまたはシンボリックリンクを検索
if [ -L "$DATASET_DIR/ml_dataset_latest.parquet" ]; then
    # シンボリックリンクが存在
    echo "   ✅ Dataset: Available (symlink)"
    RESOLVED_PATH=$(readlink -f "$DATASET_DIR/ml_dataset_latest.parquet")
    echo "   → $(basename "$RESOLVED_PATH")"
    du -sh "$RESOLVED_PATH"
elif compgen -G "$DATASET_DIR/ml_dataset_v*.parquet" > /dev/null; then
    # バージョン付きファイルが存在
    LATEST_VERSION=$(ls -t "$DATASET_DIR"/ml_dataset_v*.parquet 2>/dev/null | head -1)
    echo "   ✅ Dataset: Available (versioned)"
    echo "   → $(basename "$LATEST_VERSION")"
    du -sh "$LATEST_VERSION"
elif [ -f "$DATASET_DIR/ml_dataset.parquet" ]; then
    # レガシーファイルが存在（開発環境）
    echo "   ✅ Dataset: Available (legacy)"
    du -sh "$DATASET_DIR/ml_dataset.parquet"
else
    echo "   ⚠️  Dataset: Not found"
    echo "   Run: cd data && make build"
fi
echo ""

# 2. GPU確認
echo "2. GPU status:"
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader
echo ""

# 3. モデル学習状態
echo "3. Training processes:"
ps aux | grep -E "train_atft|train_v0|run_full_dataset" | grep -v grep || echo "   No training process"
echo ""

# 4. キャッシュ状態
echo "4. Cache status:"
if [ -d "data/output/raw/prices" ]; then
    echo "   ✅ Price cache exists"
    du -sh data/output/raw/prices
else
    echo "   ⚠️  No price cache"
fi
echo ""

# 5. ディスク容量
echo "5. Disk space:"
df -h . | awk 'NR==2 {print "   Used: "$3" / "$2" ("$5")"}'
echo ""

echo "========================================="
echo "Launching Claude Code..."
echo "========================================="
claude-code "$@"
```

### 5.2 CLAUDE.md（プロジェクトルート）

```markdown
# CLAUDE.md

## gogooku5 Project Overview

gogooku5は日本株式市場向けのモジュラーML system。データセット生成と複数モデル学習を完全分離。

### Architecture

```
gogooku5/
├── data/                # データセット生成（独立）
├── models/
│   ├── atft_gat_fan/   # ATFT-GAT-FANモデル
│   └── apex_ranker/    # APEX-Rankerモデル
├── common/             # 共通ユーティリティ
└── tools/              # Claude Code連携
```

### Quick Commands

**データセット生成**:
```bash
cd data
make build-bg          # バックグラウンド実行（5年分）
make build-quick       # クイック検証（1ヶ月）
make cache-verify      # キャッシュ検証
```

**ATFT-GAT-FAN学習**:
```bash
cd models/atft_gat_fan
make train             # 最適化学習（バックグラウンド）
make train-quick       # クイック検証（3エポック）
make train-safe        # セーフモード（安定）
```

**APEX-Ranker学習**:
```bash
cd models/apex_ranker
make train             # 学習
make inference         # 推論
```

### Development Workflow

1. **データセット更新**: `cd data && make build-bg`
2. **モデル開発**: `cd models/<model_name>` で独立環境
3. **テスト**: 各ディレクトリで `pytest tests/`
4. **並行開発**: データセット生成とモデル学習は並行実行可能

### Important Notes

- **データセット参照**: 全モデルが `data/output/ml_dataset_latest.parquet` (シンボリックリンク) または `data/output/ml_dataset_v*.parquet` (バージョン付き実体) を参照
  - `DatasetLoader._resolve_dataset_path()` が自動検出
  - 環境変数 `GOGOOKU5_DATASET_PATH` で明示的に指定可能
  - レガシー `ml_dataset.parquet` は開発環境のみ（本番非推奨）
- **モジュール独立性**: 各モジュールは独立した `pyproject.toml` と `Makefile` を持つ
- **循環依存なし**: クリーンアーキテクチャで保守性を確保

### Hardware Environment

- **GPU**: NVIDIA A100-SXM4-80GB (81920 MiB)
- **CPU**: 255-core AMD EPYC 7763 64-Core Processor
- **Memory**: 2.0Ti RAM
- **CUDA**: 12.4 (PyTorch 2.8.0+cu128)
- **FlashAttention 2**: Compatible

### Environment Variables

重要な環境変数は各モジュールの `.env.example` を参照:
- `data/.env.example`: JQuants API, キャッシュ設定
- `models/atft_gat_fan/.env.example`: 学習設定（あれば）

### Testing Strategy

- **Unit tests**: 各モジュール単体テスト（`tests/unit/`）
- **Integration tests**: パイプライン統合テスト（`tests/integration/`）
- **TDD**: テスト先行開発を推奨

### Common Issues

1. **Dataset not found**: `cd data && make build`
2. **GPU OOM**: `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
3. **Cache miss**: `cd data && make cache-verify`

### Next Steps

1. データセット生成の動作確認
2. 各モデルの学習テスト
3. エンドツーエンド検証
```

---

## トップレベル統合（Makefile）

```makefile
# gogooku5/Makefile

.PHONY: data-build train-atft train-apex help

# ===== データセット生成 =====

data-build:
	@echo "Building dataset..."
	cd data && $(MAKE) build

data-build-bg:
	@echo "Building dataset in background..."
	cd data && $(MAKE) build-bg

data-quick:
	@echo "Quick dataset validation (1 month)..."
	cd data && $(MAKE) build-quick

# ===== モデル学習 =====

train-atft:
	@echo "Training ATFT-GAT-FAN..."
	cd models/atft_gat_fan && $(MAKE) train

train-atft-quick:
	@echo "Quick ATFT-GAT-FAN validation..."
	cd models/atft_gat_fan && $(MAKE) train-quick

train-apex:
	@echo "Training APEX-Ranker..."
	cd models/apex_ranker && $(MAKE) train

# ===== ユーティリティ =====

health-check:
	@bash tools/health-check.sh

cache-verify:
	cd data && $(MAKE) cache-verify

status:
	@echo "=== System Status ==="
	@echo ""
	@echo "Dataset:"
	@if [ -L data/output/ml_dataset_latest.parquet ]; then \
		echo "  ✅ Symlink: $$(readlink data/output/ml_dataset_latest.parquet)"; \
		ls -lh data/output/ml_dataset_latest.parquet; \
	elif ls data/output/ml_dataset_v*.parquet >/dev/null 2>&1; then \
		echo "  ✅ Versioned:"; \
		ls -lht data/output/ml_dataset_v*.parquet | head -1; \
	elif [ -f data/output/ml_dataset.parquet ]; then \
		echo "  ⚠️  Legacy file:"; \
		ls -lh data/output/ml_dataset.parquet; \
	else \
		echo "  ❌ Not found"; \
	fi
	@echo ""
	@echo "Training processes:"
	@ps aux | grep -E "train_atft|train_v0" | grep -v grep || echo "  None"
	@echo ""
	@echo "GPU:"
	@nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader

# ===== ヘルプ =====

help:
	@echo "========================================="
	@echo "gogooku5 - Modular ML System"
	@echo "========================================="
	@echo ""
	@echo "Dataset Generation:"
	@echo "  make data-build       Generate dataset (foreground)"
	@echo "  make data-build-bg    Generate dataset (background)"
	@echo "  make data-quick       Quick validation (1 month)"
	@echo ""
	@echo "Model Training:"
	@echo "  make train-atft          Train ATFT-GAT-FAN"
	@echo "  make train-atft-quick    Quick ATFT validation"
	@echo "  make train-apex          Train APEX-Ranker"
	@echo ""
	@echo "Utilities:"
	@echo "  make health-check        System health check"
	@echo "  make cache-verify        Verify cache configuration"
	@echo "  make status              Show system status"
	@echo ""
	@echo "For detailed help:"
	@echo "  cd data && make help"
	@echo "  cd models/atft_gat_fan && make help"
	@echo "  cd models/apex_ranker && make help"
```

---

## 実装の進め方

### Week 1-2: Dataset基盤構築

**Day 1-2**: プロジェクト構造作成
```bash
cd /workspace/gogooku3/gogooku5

# ディレクトリ構造作成
mkdir -p data/src/builder/{api,features/{core,legacy,macro},pipelines,config,utils}
mkdir -p data/scripts
mkdir -p data/tests/{unit,integration,fixtures}
mkdir -p data/output/{raw,cache}
mkdir -p data/logs

# 初期ファイル作成
touch data/src/builder/__init__.py
touch data/pyproject.toml
touch data/.env.example
touch data/Makefile
touch data/README.md
touch data/CLAUDE.md
```

**Day 3-5**: コア設定・API fetcher移植
- 設定ファイル（4ファイル）
- JQuants API fetcher（2ファイル）
- ユニットテスト作成

**Day 6-10**: データパイプライン移植
- データ取得コンポーネント（8ファイル）
- パイプライン（3ファイル）
- キャッシュロジック検証

**Day 11-14**: エンドツーエンドテスト
- 1ヶ月分データセット生成テスト
- 統合テスト実装
- ドキュメント整備

### Week 3-4: 特徴量フル実装

**Day 15-21**: コア特徴量移植（10ファイル）
- 各特徴量モジュールの移植
- ユニットテスト作成
- 出力検証（gogooku3と一致確認）

**Day 22-28**: レガシー特徴量統合
- マクロ経済特徴量（3ファイル）
- 残りの補助特徴量（5ファイル）
- 1年分データセット生成成功

### Week 5-6: モデル実装

**Day 29-35**: ATFT-GAT-FAN
- モデル実装移植（5ファイル）
- データローダー実装（3ファイル）
- 学習パイプライン移植（5ファイル）

**Day 36-42**: APEX-Ranker
- モデル実装移植（8ファイル）
- エンドツーエンドテスト

---

## リスク管理

### 高リスク項目

1. **循環依存の解消**（15-20時間）
   - **対策**: インポートパス統一、段階的リファクタリング
   - **検証**: `ruff check` で未解決インポートチェック

2. **キャッシュロジックの移植**（10-15時間）
   - **対策**: Phase 2キャッシュ最適化の詳細テスト
   - **検証**: キャッシュヒット率100%確認

3. **特徴量の動作検証**（20-30時間）
   - **対策**: gogooku3との出力一致テスト
   - **検証**: 同一期間で生成した特徴量の差分チェック

### 中リスク項目

4. **環境変数の整合性**（5-10時間）
   - **対策**: `.env.example` の詳細ドキュメント
   - **検証**: 必須変数チェックスクリプト

5. **Polars vs Pandas互換性**（5-10時間）
   - **対策**: Polars APIの統一使用
   - **検証**: 型チェック（mypy）

---

## 期待される成果物

### Dataset生成モジュール
- ✅ gogooku3と同等の~307特徴量データセット生成
- ✅ GPU ETL（RAPIDS/cuDF）標準装備
- ✅ Phase 2キャッシュ最適化（100%ヒット率）
- ✅ 完全独立したモジュール（他への依存なし）
- ✅ 包括的テストカバレッジ（>80%）
- ✅ `make build-bg` で5年分データ生成可能

### ATFT-GAT-FANモデル
- ✅ 独立したモデル実装
- ✅ 共通データセット参照（data/output/ml_dataset_latest.parquet）
- ✅ SafeTrainingPipeline対応
- ✅ フェーズベース学習
- ✅ `make train` でエンドツーエンド学習可能

### APEX-Rankerモデル
- ✅ 独立したモデル実装
- ✅ 共通データセット参照（data/output/ml_dataset_latest.parquet）
- ✅ 推論・バックテスト対応
- ✅ `make train` で学習可能

### 共通基盤
- ✅ クリーンアーキテクチャ（循環依存なし）
- ✅ モジュラー構造（新モデル追加が容易）
- ✅ Claude Code / Codex フレンドリー
- ✅ 包括的ドキュメント（README, CLAUDE.md）
- ✅ ヘルスチェックスクリプト

---

## 次のアクション

1. **この計画の確認**
   - ディレクトリ構造は適切か？
   - 移植ファイルリストは妥当か？
   - 期間見積もりは現実的か？

2. **実装開始**
   - Week 1から段階的に進める
   - 各マイルストーンでレビュー

3. **継続的な改善**
   - 実装中に発見した問題の記録
   - ドキュメントの随時更新

---

**確認事項**:
1. `data/` ディレクトリの完全独立化（データセット生成専用）はOKか？
2. `models/` 配下に各モデルを配置する構造（ATFT-GAT-FAN, APEX-Ranker等）でOKか？
3. `common/` ユーティリティは必要か？（現在はオプショナル）
4. この計画で実装を開始してよろしいでしょうか？
