📊 データセット検証レポート

  ✅ 基本情報

  現在使用中のデータセット:
  ファイル: output/ml_dataset_latest_full.parquet
  実体: output/datasets/ml_dataset_20151013_20250926_20251019_120214_full.parquet
  サイズ: 4.2GB
  作成日時: 2025-10-19 12:04 (本日、28分前)

  📈 データセット統計

  | 項目      | 値                               |
  |---------|---------------------------------|
  | 総行数     | 8,988,034 rows                  |
  | 列数      | 112 columns                     |
  | 期間      | 2015-10-13 → 2025-09-26 (約10年間) |
  | 取引日数    | 2,434 days                      |
  | ユニーク銘柄数 | 4,484 stocks                    |

  ⚠️ 重要な発見: 特徴量不足

  期待値: ~303-307 features (Standard plan, futures除く)実際: 112 columns差分: 約195 columns不足

  🔍 欠損している特徴量カテゴリ

  現在のデータセットには基本的な特徴量のみが含まれています:
  - ✅ OHLCV基本データ
  - ✅ リターン系 (1d, 5d, 10d, 20d, 60d, 120d)
  - ✅ テクニカル指標 (SMA, EMA, RSI, MACD, BB, ATR等)
  - ✅ マーケット特徴量 (TOPIX)
  - ✅ ベータ/アルファ特徴量
  - ✅ ターゲット変数

  ❌ 欠損している高度な特徴量:
  1. 財務諸表特徴量 (QualityFinancialFeaturesGenerator)
    - YoY成長率、営業利益率、ROE/ROA等
  2. 決算イベント特徴量 (earnings_events)
  3. 信用取引特徴量 (daily_margin_interest)
  4. 空売り特徴量 (short_selling, sector_short_selling)
  5. オプション特徴量 (NK225 index options)
  6. 高度ボラティリティ特徴量 (advanced_volatility)
  7. グラフ特徴量 (graph features from correlation networks)
  8. フロー特徴量 (取引データベース)
  9. セクタークロスセクショナル特徴量 (sector cross-sectional)

  📋 データ品質チェック

  NULL値 (先頭100行サンプル):
  - returns_120d: 100% NULL (正常 - 120日履歴必要)
  - returns_60d: 60% NULL (正常 - 60日履歴必要)
  - returns_20d: 20% NULL (正常 - 20日履歴必要)
  - returns_10d: 10% NULL (正常)
  - returns_5d: 5% NULL (正常)
  - returns_1d: 1% NULL (正常)

  → NULL値は正常範囲内

  🔧 原因分析

  ビルドログ(_logs/dataset/dataset_bg_20251019_120052.log)を確認したところ:

  1. エラー検出:
  Failed to fetch short selling data: 'Expr' object has no attribute 'dtype'
    - Polarsバージョン互換性の問題
  2. ビルドプロセス:
    - run_pipeline_v4_optimized.py → 161 columns (中間ファイル)
    - 期待: さらに+100列以上の enrichment
    - 実際: 112 columns (減少!)
  3. 可能性:
    - 簡易版データセットビルダー(ml_dataset_builder.py)が使用された
    - またはrun_full_dataset.pyの特徴量生成が部分的に失敗

  ✅ 現状の訓練への影響

  訓練は正常に進行中:
  PID: 376015
  経過時間: 2時間以上
  CPU: 50.6%
  データセット: output/ml_dataset_latest_full.parquet (112 columns)

  ただし:
  - ✅ 基本的な特徴量で訓練は可能
  - ⚠️ フル性能を発揮できていない可能性
  - ⚠️ ~195 columns分の情報が欠落

  💡 推奨アクション

  1. 現在の訓練は継続 (中断不要)
  2. 訓練完了後、フルデータセット再生成:
  make dataset-bg
  3. 次回訓練でフル特徴量セット使用 (~307 features)

  📊 結論

  データセットの正確性:
  - ✅ データ品質: 正確 (NULL値正常、期間完全)
  - ⚠️ 特徴量数: 不完全 (112/307 = 36%のみ)
  - ✅ 訓練可能: Yes
  - ⚠️ 最適性能: No (特徴量不足)

  データセットは正しく作成されていますが、簡易版です。フル性能には~307特徴量が必要です。
  