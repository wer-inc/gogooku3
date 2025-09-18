# Safe Training Regression – 2025-09-18

最新データセット（`output/ml_dataset_20250918_060642_full.parquet`）と直前の旧データセット（`output/ml_dataset_20250918_054314_full.parquet`）で Safe Training Pipeline を比較した結果をまとめます。

## 実行環境
- コマンド: `python scripts/run_safe_training.py --data-dir <dataset_dir> --n-splits 2 --memory-limit 6`
- 新データ: `output/experiments/safe_training_20250918_060904/`
- 旧データ: `output/experiments/safe_training_20250918_063246/`

## 指標比較（第1スプリット）
| Horizon | New IC | Old IC | ΔIC | New RankIC | Old RankIC | ΔRankIC |
|---------|-------:|-------:|----:|-----------:|-----------:|--------:|
| 1d | 0.286 | 0.288 | -0.002 | 0.192 | 0.188 | +0.003 |
| 5d | 0.319 | 0.320 | -0.001 | 0.254 | 0.249 | +0.005 |
| 10d | 0.358 | 0.357 | +0.001 | 0.275 | 0.270 | +0.004 |
| 20d | 0.402 | 0.402 | -0.001 | 0.293 | 0.295 | -0.002 |

- RankIC は 1d/5d/10d で +0.003〜+0.005 と改善、20d のみ -0.002 で横ばい。
- IC は 10d を除き誤差レベルの微減。進捗比クリップ導入による悪化は確認されていません。

## パイプライン健全性
- 正規化ステップでの平均絶対偏差は ~0（新: 6.99e-08, 旧: 3.28e-07）、警告ゼロ。
- グラフ構築は両ケースともノード 50 / エッジ 5 で一致。
- 実行時間: 新 81.1s、旧 79.9s（±1.2s 差は許容範囲）。

## 結論
- 新データセットは RankIC を中心に改善傾向で、異常値クリップや TE 追加による負の影響は認められません。
- 今後の学習ジョブでも `output/ml_dataset_latest_full.parquet` を使用して問題ないと判断します。
