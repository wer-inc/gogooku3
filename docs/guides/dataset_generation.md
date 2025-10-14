# Dataset Generation Guide - Up to 395 Features (Currently ~307)

⚠️ **Note (2025年10月更新)**: 先物機能（88-92列）が無効化されているため、実際の生成列数は約303-307列となります。詳細は `docs/ml/dataset_new.md` Section 13.1 を参照してください。

## Current Status

現在のデータセット：**198特徴量** / 理論的最大：**395特徴量** / 実際生成数：**約307特徴量**

### ✅ 既に実装されている特徴量（198）

#### 相互作用特徴量（x_*）: 23個
- ✅ `x_trend_intensity`, `x_trend_intensity_g`
- ✅ `x_rel_sec_mom`, `x_z_sec_gap_mom`
- ✅ `x_mom_sh_5`, `x_mom_sh_10`, `x_mom_sh_5_mktneu`
- ✅ `x_rvol5_dir`, `x_rvol5_bb`
- ✅ `x_squeeze_pressure`, `x_credit_rev_bias`
- ✅ `x_pead_effect`, `x_pead_times_mkt`
- ✅ `x_rev_gate`, `x_bo_gate`
- ✅ `x_alpha_meanrev_stable`
- ✅ `x_foreign_relsec`, `x_tri_align`
- ✅ `x_bbpos_rvol5`, `x_bbneg_rvol5`
- ✅ `x_liquidityshock_mom`, `x_dmi_impulse_dir`, `x_breadth_rel`
- ❌ `x_flow_smart_rel` (欠落)

#### 基本特徴量
- 価格・ボリューム: 19個
- 移動平均: 27個
- テクニカル指標: 17個
- 市場特徴量: 19個
- フロー特徴量: 9個
- 週次マージン: 17個
- 財務諸表: 17個

### ❌ 欠落している主要特徴量

#### 1. 先物特徴量（fut_*）: 88-92列 **（API制限により無効化）**
- ON (T+0): 5列 × 4カテゴリ = 20列
- EOD (T+1): 17列 × 4カテゴリ = 68列
- Continuous: 1列 × 4カテゴリ = 4列（`--futures-continuous`）
- **再有効化**: `--futures-parquet` でオフラインデータ指定が必要

#### 2. その他の特徴量（データ依存）
- 日次マージン特徴量（dmi_*）: ~41個（データ利用可能時）
- セクター集約特徴量: ~40個（実装済み）
- オプション特徴量（未実装）
- 高度なボラティリティ特徴量（実装済み）

## MLDatasetBuilderの全機能（24メソッド）

```python
1. add_daily_margin_block        # 日次マージン特徴量（41個）
2. add_earnings_features         # 決算イベント特徴量
3. add_enhanced_flow_features    # 拡張フロー特徴量
4. add_enhanced_listed_features  # 拡張上場情報特徴量
5. add_enhanced_margin_features  # 拡張マージン特徴量
6. add_flow_features             # 基本フロー特徴量
7. add_futures_block             # 先物特徴量
8. add_index_features            # インデックス特徴量
9. add_interaction_features      # 相互作用特徴量（実装済み）
10. add_margin_weekly_block      # 週次マージン（実装済み）
11. add_option_sentiment_features # オプションセンチメント
12. add_pandas_ta_features       # pandas-ta テクニカル指標
13. add_relative_to_sector       # セクター相対特徴量
14. add_sector_encodings         # セクターエンコーディング
15. add_sector_features          # セクター特徴量
16. add_sector_index_features    # セクターインデックス特徴量
17. add_sector_series            # セクター時系列集約
18. add_sector_target_encoding   # セクターターゲットエンコーディング
19. add_short_position_features  # ショートポジション特徴量
20. add_short_selling_block      # 空売り特徴量
21. add_statements_features      # 財務諸表特徴量（実装済み）
22. add_topix_features          # TOPIX特徴量（実装済み）
23. create_metadata             # メタデータ作成
24. create_technical_features   # テクニカル特徴量（実装済み）
```

## 完全なデータセット生成コマンド

### オプション1: バックグラウンド実行（最も推奨）

```bash
# SSH切断にも安全。ログ/ PID / PGID を保存します。
make dataset-bg

# モニタ
tail -f _logs/dataset/*.log

# 停止
kill <PID>
# またはプロセスグループごとに停止
kill -TERM -<PGID>
```

### オプション2: Makefileを使用（インタラクティブ）

```bash
# GPUを使用して全機能を有効化（期間指定）
make dataset-full-gpu START=2020-01-01 END=2024-12-31
```

### オプション3: 直接Pythonスクリプトを実行

```bash
export REQUIRE_GPU=1 USE_GPU_ETL=1 RMM_POOL_SIZE=70GB CUDA_VISIBLE_DEVICES=0

python scripts/pipelines/run_full_dataset.py \
  --jquants \
  --start-date 2020-01-01 \
  --end-date 2024-12-31 \
  --gpu-etl \
  --enable-indices \
  --enable-advanced-features \
  --enable-sector-cs \
  --enable-graph-features \
  --enable-daily-margin \
  --enable-margin-weekly \
  --enable-earnings-events \
  --enable-sector-short-selling \
  --enable-short-selling \
  --enable-advanced-vol \
  --enable-option-market-features \
  --enable-futures
```

### オプション4: 生成スクリプトを使用

```bash
./scripts/generate_full_dataset.sh
```

## 重要なポイント

1. **日次マージン自動検出**: J-Quantsまたは既存のParquetファイルがあれば自動的に有効化されるはず
2. **GPU-ETL**: `USE_GPU_ETL=1`で高速化（cuDF/RAPIDS使用）
3. **メモリ制限**: 大規模データセットの場合、`RMM_POOL_SIZE`を調整

## 検証方法

```bash
# データセット特徴量の検証
python scripts/verify_dataset_features.py

# 期待される出力（先物無効化時）：
# ✅ Total Features: 303-307 (理論最大: 395)
# ⚠️  Futures (fut_*): 0 features (88-92 disabled)
# ✅ Daily Margin (dmi_*): 41 features (データ依存)
# ✅ Interaction Features: 22 features
# ✅ Sector Aggregates: 40+ features
```

## トラブルシューティング

### 日次マージンが有効化されない場合

```python
# run_full_dataset.py:946 の自動検出ロジック確認
enable_daily_margin=(args.jquants or args.enable_daily_margin
                    or bool(daily_margin_parquet is not None and daily_margin_parquet.exists()))
```

### メモリエラーの場合

```bash
# メモリ制限を下げる
export RMM_POOL_SIZE=50GB  # 70GBから50GBに削減
```

### 特定の機能のみテスト

```python
# 単一機能のテスト
from pathlib import Path
from scripts.data.ml_dataset_builder import MLDatasetBuilder

builder = MLDatasetBuilder(output_dir=Path('output/test'))
df = pl.read_parquet('output/ml_dataset_latest_full.parquet')
df_with_dmi = builder.add_daily_margin_block(df)
print(f"Added {len([c for c in df_with_dmi.columns if c.startswith('dmi_')])} DMI features")
```

## 次のステップ

1. ✅ 検証スクリプト作成済み
2. ✅ 日次マージンデータ確認済み
3. ✅ 全有効機能でのデータセット生成（~307特徴量）
4. ⚠️  先物機能再有効化（オフラインデータが必要）
5. ⏳ 学習パイプラインでのテスト
