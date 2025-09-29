# データセット生成
make rebuild-dataset START=2015-09-27 END=2025-09-26
make dataset-full-gpu START=2015-09-28 END=2025-09-29
make dataset-full-gpu-bg START=2015-09-27 END=2025-09-26

# 学習コマンド
make train-rankic-boost


---

2025-09-29 22:36:34,175 - run_full_dataset - INFO - Full enriched dataset saved
2025-09-29 22:36:34,175 - run_full_dataset - INFO -   Dataset : output/datasets/ml_dataset_20150928_20250929_20250929_223552_full.parquet
2025-09-29 22:36:34,175 - run_full_dataset - INFO -   Metadata: output/datasets/ml_dataset_20150928_20250929_20250929_223552_full_metadata.json
2025-09-29 22:36:34,175 - run_full_dataset - INFO -   Symlink : output/ml_dataset_latest_full.parquet
2025-09-29 22:36:35,562 - run_full_dataset - INFO - Saved TOPIX market features: output/datasets/topix_market_features_20150928_20250929.parquet
2025-09-29 22:36:36,577 - run_full_dataset - INFO - Fetching index options 2015-09-28 → 2025-09-29