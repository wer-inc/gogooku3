## Deep Optimisation Phases – Draft Roadmap

| Phase | Status | Owner | Completed | 主な成果物 |
|-------|--------|-------|-----------|-------------|
| **Phase 1 – Streaming DataLoader** | ✅ Complete | Claude | 2025-10-28 | `ParquetStockIterableDataset` + `OnlineRobustScaler`、GPU/CPU ベンチマーク（21-27%高速化）、回帰テスト 3 件、`output/reports/atft_gap_analysis.md` 更新 |
| **Phase 2 – Graph Reconstruction** | ✅ Complete | Claude + Team | 2025-10-29 | `src/graph/graph_builder.py` リファクタ、config 統一 (k=20, threshold=0.3, lookback=60)、`tests/graph/test_graph_builder.py`（12ケース）、`output/reports/graph_*` レポート更新 |
| **Phase 3 – FAN/SAN レイヤ強化** | 🚧 In Progress | TBD | - | `src/atft_gat_fan/models/components/adaptive_normalization.py` 改良、`tests/atft/test_adaptive_normalization.py`、FAN/SAN smoke ログ |
| **Phase 4 – Phase Training Framework** | 🚧 In Progress | TBD | 2025-10-29 (partial) | `configs/atft/train/phase0.yaml`〜`phase4.yaml`、`--phase/--resume-checkpoint` CLI、フェーズ別チェックポイント出力 |
| **Phase 5 – 評価 & レポート自動化** | ⏳ Planned | TBD | - | `scripts/evaluate_trained_model.py` の CI95・可視化拡張、`scripts/backtest_sharpe_model.py` 連携、`output/reports/max_push_evaluation.md` |
| **Phase 6 – パフォーマンス最適化 / HPO** | ⏳ Planned | TBD | - | `make hpo-run` 20 トライアル、自動サマリ `output/reports/hpo_summary_YYYYMMDD.md`、`docs/EVALUATION_REPORT_20251028.md` 更新 |

---

## Phase 1: Streaming DataLoader (✅ Complete)

### Achievements
- **Performance**: GPU 21%高速化、CPU 27%高速化
- **Memory**: GPU メモリ使用量 20%削減 (16GB vs 20-25GB)
- **Scalability**: OOM リスク解消、無制限のデータセットサイズ対応
- **Tests**: 回帰テスト 3 件追加（shard stitching, worker sharding, NaN handling）

### Deliverables
- `src/data/parquet_stock_dataset.py`: ParquetStockIterableDataset + OnlineRobustScaler
- `src/gogooku3/training/atft/data_module.py`: ProductionDataModuleV2 統合
- `tests/unit/test_parquet_stock_iterable.py`: 回帰テスト 3 件
- `output/reports/atft_gap_analysis.md`: ベンチマーク比較表
- Benchmark logs: `output/reports/iterable_gpu_smoke.log`, `iterable_cpu_smoke.log`

### Known Issues (P1)
- Multi-worker DataLoader デッドロック（num_workers=2 でテストがハング）
- Pre-commit hook 失敗（既存の lint 問題、Phase 1 非関連）

### Branch
- `feature/phase1-streaming-dataloader`

---

## Phase 2: Graph Reconstruction (✅ Complete)

### Highlights
- `src/graph/graph_builder.py` を全面リファクタし、`lookback=60`・`k=20`・`threshold=0.3` の絶対相関 kNN グラフへ刷新。
- 旧グラフ（平均次数 3.90, エッジ 7,244）から、新グラフは平均次数 **31.0**・エッジ **59,253**・ノード 3,818 を達成。
- すべての Hydra config を統一し (`configs/atft/config*.yaml`, `max_push.yaml`, `unified_config.yaml`)、GAT 近傍情報を拡充。
- `tests/graph/test_graph_builder.py`（12 ケース）で相関計算・kNN・属性付与・キャッシュ再利用のリグレッションを整備。
- レポート類を更新：`output/reports/graph_baseline_analysis.md`, `output/reports/correlation_prototype.md`, `output/reports/atft_gap_analysis.md`。

### Follow-up
- Phase 3 以降のプランに基づき、FAN/SAN の精度検証とフェーズ制御の実装へ移行。
- グラフキャッシュ再生成時の平均次数・生成時間を継続的にモニタリング。

---

## Phase 3: FAN/SAN レイヤ強化 (🚧 In Progress)

### Objectives
FAN/SAN を多窓・スライス適応に対応させ、NaN/勾配の安定性を高める。

### Tasks
- [x] `tests/atft/test_adaptive_normalization.py` を新設し、FAN/SAN の NaN ガード・勾配伝播を検証（4 ケース）。
- [x] `src/atft_gat_fan/models/components/adaptive_normalization.py` を改良（多窓 Softmax／スライス学習重み／NaN クリーニング）。
- [ ] Phase 0→2 チェックポイントを用いた 1 epoch スモーク（GPU unavailable のため未完了。ログ: `output/reports/fan_san_smoke_*.log`）。
- [ ] Hydra 設定に FAN/SAN の窓・スライス数をパラメタ化し、段階的に ON/OFF 切替可能にする。
- [ ] 5 epoch 程度の追加検証を実施し、RankIC / Sharpe への影響を計測（GPU 環境復旧後）。

### Next
- GPU 環境が利用可能になり次第、`--max-epochs 1` スモークを再実行してログを確定させる。
- FAN/SAN の Phase 学習制御（Phase 0→3 での ON/OFF）と Hydra プロファイル整備に着手。

---

## Phase 4: Phase Training Framework (🚧 In Progress)

### Achievements
- `scripts/integrated_ml_training_pipeline.py` に `--phase` / `--resume-checkpoint` / `--save-phase-checkpoint` フラグを実装し、Phase 0→4 の段階学習が CLI 一発で実行可能になった。
- `prepare_phase_training()` が `PHASE_RESET_EPOCH=1` / `PHASE_RESET_OPTIMIZER=1` / `PHASE_INDEX` を設定し、`train_atft.py` 側でチェックポイント再開時に epoch・Optimizer/GradScaler state をリセット。フェーズ跨ぎの 1 epoch スモークでも確実に学習ループが動作する。
- Phase 向け Hydra 設定 (`configs/atft/train/phase0_baseline.yaml`〜`phase4_finetune.yaml`) を整備し、GAT/FAN/SAN の有効化や LR / 損失ウェイトを段階的に切替可能に。
- Iterable DataLoader のスケーラフィットが `NORMALIZATION_MAX_SAMPLES` / `NORMALIZATION_MAX_FILES` を尊重するようになり、Phase 3 (SAN) で発生していた 200k サンプル走査ハングを解消（デフォルト 8,192 サンプルに短縮）。

### Next
- GPU 復旧後、`python scripts/integrated_ml_training_pipeline.py --phase N --max-epochs 2 --save-phase-checkpoint` を Phase 0→4 で連鎖実行し、`phase{N}_metrics.json` を収集。
- `PHASE_CHECKPOINT_PREFIX` を `train_atft.py` まで反映し、自動で `phase{N}_best.pt` を命名。
- README / docs/architecture に Phase トレーニング手順とトラブルシューティング（env override 含む）を追記。
