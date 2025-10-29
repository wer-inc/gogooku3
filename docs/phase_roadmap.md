## Deep Optimisation Phases – Draft Roadmap

| Phase | Status | Owner | Completed | 主な成果物 |
|-------|--------|-------|-----------|-------------|
| **Phase 1 – Streaming DataLoader** | ✅ Complete | Claude | 2025-10-28 | `ParquetStockIterableDataset` + `OnlineRobustScaler`、GPU/CPU ベンチマーク（21-27%高速化）、回帰テスト 3 件、`output/reports/atft_gap_analysis.md` 更新 |
| **Phase 2 – Graph Reconstruction** | 🚧 In Progress | TBD | - | 新 `FinancialGraphBuilder` 実装（時系列整列 → 相関 → kNN/閾値）、`src/graph/utils.py` の code↔node マップ、`output/graph_cache/graph_snapshot_YYYYMMDD.pkl` |
| **Phase 3 – FAN/SAN レイヤ強化** | ⏳ Planned | TBD | - | `src/atft_gat_fan/layers/` への FAN/SAN 実装、`tests/atft/test_fan.py` / `test_san.py` 、5 epoch の検証ログ |
| **Phase 4 – Phase Training Framework** | ⏳ Planned | TBD | - | `configs/atft/train/phase0.yaml`〜`phase4.yaml`、`make train-phase` CLI、各フェーズのメトリクス JSON |
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

## Phase 2: Graph Reconstruction (🚧 In Progress)

### Objectives
相関グラフ生成の全面刷新：直近リターン系列の相関 → kNN + 閾値 → 市場/セクター属性付与

### Tasks

#### 2.1 Core Implementation
- [ ] `src/graph/graph_builder.py` リファクタリング
  - [ ] 時系列整列（Date でソート、欠損期間の補完）
  - [ ] リターン系列からの相関計算（Pearson/Spearman）
  - [ ] kNN + 閾値によるエッジ選択（平均次数 20 を確保）
  - [ ] 市場/セクター属性の付与（node features）
- [ ] `src/graph/utils.py` の code↔node_id マップ整備
  - [ ] `CodeToNodeMapper` クラス実装
  - [ ] バッチごとの動的マッピング対応
- [ ] `src/graph/cache_manager.py` のスナップショット保存
  - [ ] 月次キャッシュ構造: `output/graph_cache/YYYYMM/graph_snapshot_YYYYMMDD.pkl`
  - [ ] メタデータ付き保存（相関パラメータ、エッジ数、ノード数）

#### 2.2 Testing
- [ ] `tests/graph/test_graph_builder.py` 追加
  - [ ] マルチ shard でのグラフ生成テスト
  - [ ] 閾値カットのエッジ数検証
  - [ ] code↔node_id マップの一貫性テスト
  - [ ] 相関計算の正確性テスト（合成データ）
- [ ] `tests/graph/test_cache_manager.py` 追加
  - [ ] キャッシュ保存/読み込みテスト
  - [ ] 古いキャッシュの自動削除テスト

#### 2.3 Integration
- [ ] `scripts/train_atft.py` へのグラフ生成統合
  - [ ] エポックごとのグラフ再構築フラグ
  - [ ] キャッシュヒット率のロギング
- [ ] `configs/atft/graph_config.yaml` 追加
  - [ ] 相関パラメータ（window, method, threshold, k）
  - [ ] キャッシュ設定（max_age, compression）

#### 2.4 Documentation
- [ ] `docs/architecture/graph_reconstruction.md` 作成
  - [ ] 設計思想（なぜ相関ベースか）
  - [ ] パラメータ調整ガイド
  - [ ] パフォーマンス特性（メモリ、計算時間）
- [ ] `output/reports/atft_gap_analysis.md` Phase 2 セクション更新
  - [ ] グラフ生成前後のメトリクス比較
  - [ ] エッジ数分布の可視化

### Acceptance Criteria
- [ ] 平均次数 20±3 を維持
- [ ] 相関計算が正しい（単体テスト pass）
- [ ] キャッシュヒット率 > 80%（2 回目以降の実行）
- [ ] グラフ生成時間 < 10 秒（バッチサイズ 256）
- [ ] `pytest tests/graph/ -v` が全 pass

### Branch
- `feature/phase2-graph-rebuild`

---

### Notes
- Phase 1 完了により、DataLoader の P0 課題は解決
- Phase 2 と 3 はエンジニアのアサインによっては並行実施を検討
- 各 Phase の成果が揃い次第、`docs/architecture/atft_gat_fan_implementation.md` に現状メモを追記する
