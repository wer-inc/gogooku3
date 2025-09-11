# Scripts Directory Structure

## Overview
JQuants APIを利用した金融データパイプラインのスクリプト群

## Directory Structure
```
scripts/
├── core/                 # コアモジュール
│   └── ml_dataset_builder.py   # ML特徴量生成エンジン
├── components/           # 独立データコンポーネント
│   └── modular_updater.py      # モジュラーETLシステム
├── pipelines/            # 実行可能パイプライン
│   ├── run_pipeline.py         # メインETLパイプライン
│   └── update_topix.py         # TOPIX専用更新スクリプト
└── utils/                # ユーティリティ（今後拡張）
```

## Quick Start

### 1. フルパイプライン実行（One‑Shot Full Dataset）
```bash
# サンプルデータで実行（最適化版）
python scripts/pipelines/run_pipeline_v4_optimized.py --stocks 10 --days 100

# JQuants APIで実行
python scripts/pipelines/run_pipeline_v4_optimized.py --jquants --stocks 100 --days 300
```

#### セクター機能付きのフルデータセット生成
```bash
python scripts/pipelines/run_full_dataset.py \
  --jquants \
  --start-date 2020-09-06 --end-date 2025-09-06 \
  # オフライン時の代替入力
  [--topix-parquet output/topix_history_YYYYMMDD_YYYYMMDD.parquet] \
  [--statements-parquet output/event_raw_statements_YYYYMMDD_YYYYMMDD.parquet] \
  [--listed-info-parquet output/listed_info_history_YYYYMMDD.parquet] \
  # セクター機能トグル
  [--sector-onehot33] \
  [--sector-series-mcap auto|always|never] \
  [--sector-series-levels 33|17|33,17] \
  [--sector-te-targets target_5d[,target_1d]] \
  [--sector-te-levels 33|17|33,17]
```

セクター機能の概要:
- 基礎列: `sector17/33_code/name/id`（as‑of結合、リークなし）
- カテゴリ: 17業種 One‑Hot（希少カテゴリはOther）、日別頻度 `sec17/33_daily_freq`
- 集約系列: `sec_ret_1d/5d/20d_eq`、`sec_mom_20`、`sec_ema_5/20`、`sec_gap_5_20`、`sec_vol_20`、`sec_vol_20_z`
- 時価総額加重（オプション）: `sec_ret_1d/5d/20d_mcap`（`shares_outstanding` があれば自動）
- 相対化: `rel_to_sec_5d`、`alpha_vs_sec_1d`、`ret_1d_demeaned`、`z_in_sec_*`
- ターゲットエンコード: `te_sec_<target>`（K‑fold＋Δラグ＋ベイズ平滑、フォールバックあり）

### 2. TOPIX特徴量の追加
```bash
# 既存データセットにTOPIXを追加
python scripts/pipelines/update_topix.py --latest --days 100
```

### 3. モジュラー更新
```bash
# 特定コンポーネントのみ更新
python scripts/components/modular_updater.py \
  --dataset output/ml_dataset_latest.parquet \
  --update topix trades_spec \
  --days 30
```

## Module Descriptions

### core/ml_dataset_builder.py
- **役割**: 技術指標と特徴量の計算エンジン
- **機能**:
  - 62種類の技術指標生成
  - TOPIX相対特徴量計算
  - pandas-ta統合
  - バグ修正（P0-P2）実装済み

### components/modular_updater.py
- **役割**: 独立コンポーネント管理システム
- **コンポーネント**:
  - PriceDataComponent: 価格データ
  - TopixComponent: TOPIX指数
  - TradesSpecComponent: 売買仕様
  - ListedInfoComponent: 企業情報

### pipelines/run_pipeline_v4_optimized.py
- **役割**: 最適化済みメインETLパイプライン
- **機能**:
  - JQuants API認証
  - 非同期データ取得（150並行接続）
  - 特徴量生成（pandas-ta含む）
  - データ保存（Parquet + メタデータ）

### pipelines/update_topix.py
- **役割**: TOPIX専用更新スクリプト
- **機能**:
  - 既存データセットの読み込み
  - TOPIX特徴量の追加/更新
  - メタデータ更新

## Data Flow
```
JQuants API
    ↓
[Async Fetchers]
    ↓
[Data Components]
    ↓
[ML Dataset Builder]
    ↓
[Storage Layer]
    ├── Parquet (Primary)
    ├── CSV (Compatibility)
    └── Metadata (JSON)
```

## Output Files
- `ml_dataset_YYYYMMDD_HHMMSS.parquet` - メインデータセット
- `ml_dataset_YYYYMMDD_HHMMSS.csv` - CSV形式
- `ml_dataset_YYYYMMDD_HHMMSS_metadata.json` - メタデータ
- `ml_dataset_latest.parquet` - 最新データへのシンボリックリンク

## Environment Variables
`.env`ファイルに以下を設定：
```bash
JQUANTS_AUTH_EMAIL=your_email@example.com
JQUANTS_AUTH_PASSWORD=your_password
MAX_CONCURRENT_FETCH=150
DEFAULT_STOCKS=100
DEFAULT_DAYS=300
```

## Testing
```bash
# ユニットテスト（今後実装）
python -m pytest scripts/tests/

# 動作確認
python scripts/pipelines/run_pipeline_v4_optimized.py --stocks 5 --days 10
```

## Archived Scripts（移管済み）
以下は `scripts/_archive/` に移動しました。代替をご利用ください。

- run_pipeline.py → `scripts/pipelines/run_pipeline_v4_optimized.py`
- create_full_historical_dataset.py / create_historical_dataset.py → `scripts/pipelines/run_pipeline_v4_optimized.py`
- generate_full_dataset.py → `scripts/pipelines/run_full_dataset.py`
- その他一覧はリポジトリの `README.md > Archived Scripts` を参照

## Integration with Dagster/Airflow
`components/modular_updater.py`のコンポーネントは独立したタスクとして定義可能。
詳細は`/docs/MODULAR_ETL_DESIGN.md`を参照。
