## Deep Optimisation Phases – Draft Roadmap

| Phase | Owner (Draft) | Kick-off目安 | 主な成果物 |
|-------|---------------|--------------|-------------|
| **Phase 2 – Graph Reconstruction** | TBD (Graph team) | Week 2 | 新 `FinancialGraphBuilder` 実装（時系列整列 → 相関 → kNN/閾値）、`src/graph/utils.py` の code↔node マップ、`output/graph_cache/graph_snapshot_YYYYMMDD.pkl` |
| **Phase 3 – FAN/SAN レイヤ強化** | TBD (Model team) | Week 3 | `src/atft_gat_fan/layers/` への FAN/SAN 実装、`tests/atft/test_fan.py` / `test_san.py` 、5 epoch の検証ログ |
| **Phase 4 – Phase Training Framework** | TBD (Training Ops) | Week 4 | `configs/atft/train/phase0.yaml`〜`phase4.yaml`、`make train-phase` CLI、各フェーズのメトリクス JSON |
| **Phase 5 – 評価 & レポート自動化** | TBD (Analytics) | Week 5 | `scripts/evaluate_trained_model.py` の CI95・可視化拡張、`scripts/backtest_sharpe_model.py` 連携、`output/reports/max_push_evaluation.md` |
| **Phase 6 – パフォーマンス最適化 / HPO** | TBD (Research) | Week 6 | `make hpo-run` 20 トライアル、自動サマリ `output/reports/hpo_summary_YYYYMMDD.md`、`docs/EVALUATION_REPORT_20251028.md` 更新 |

### Notes
- Phase 2 と 3 はエンジニアのアサインによっては並行実施を検討。
- 各 Phase の成果が揃い次第、`docs/architecture/atft_gat_fan_implementation.md` に現状メモを追記する。
