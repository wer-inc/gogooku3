# Gogooku3 ドキュメント索引

## 📚 ドキュメント構成

本プロジェクトのドキュメントは、以下の4つのカテゴリに整理されています。

### 📋 [仕様書 (specifications/)](specifications/)
システムの技術仕様とデータ定義を記載したドキュメント群

- **[ML_DATASET_COLUMNS.md](specifications/ML_DATASET_COLUMNS.md)**
  - MLデータセットの全カラム仕様（62+ 特徴量）
  - 各特徴量の計算式と意味
  - データ型とNULL許容性

- **[MODULAR_ETL_DESIGN.md](specifications/MODULAR_ETL_DESIGN.md)**
  - モジュール化ETL設計の詳細
  - 独立更新可能なコンポーネント設計
  - データフロー仕様

- **[TECHNICAL_INDICATORS_COMPARISON.md](specifications/TECHNICAL_INDICATORS_COMPARISON.md)**
  - 技術指標の実装比較（pandas-ta vs 独自実装）
  - 使用推奨指標の一覧
  - パフォーマンス比較

### 📊 [レポート (reports/)](reports/)
プロジェクトの進捗と成果物に関する報告書

- **[PROJECT_SUMMARY.md](reports/PROJECT_SUMMARY.md)**
  - プロジェクト全体の成果サマリー
  - 実装済み機能一覧
  - パフォーマンス統計

- **[BUG_FIXES_REPORT.md](reports/BUG_FIXES_REPORT.md)**
  - 修正済みバグ一覧（P0-P2）
  - データリーク防止策
  - 計算精度改善内容

### 📖 [ガイド (guides/)](guides/)
実装と運用に関する実践的なガイド

- **[PROCESSING_FLOW.md](guides/PROCESSING_FLOW.md)** 📌 **重要**
  - データ処理フロー全体図
  - 各フェーズの詳細説明
  - 部分実行パターン
  - トラブルシューティング

### 🗄️ [アーカイブ (archive/)](archive/)
過去のドキュメントと設計書

- **[gogooku3-spec.md](archive/gogooku3-spec.md)** - 原設計仕様書
- **[BATCH_REDESIGN_PLAN.md](archive/BATCH_REDESIGN_PLAN.md)** - バッチ再設計計画
- **[IMPLEMENTATION_PLAN.md](archive/IMPLEMENTATION_PLAN.md)** - 実装計画書
- **[legacy/](archive/legacy/)** - 過去の開発履歴

## 🚀 クイックリファレンス

### 最初に読むべきドキュメント
1. [PROCESSING_FLOW.md](guides/PROCESSING_FLOW.md) - 処理フローを理解
2. [ML_DATASET_COLUMNS.md](specifications/ML_DATASET_COLUMNS.md) - 生成される特徴量を確認
3. [PROJECT_SUMMARY.md](reports/PROJECT_SUMMARY.md) - プロジェクト全体像を把握

### ユースケース別ガイド

#### 🔧 開発者向け
- モジュール設計を理解したい → [MODULAR_ETL_DESIGN.md](specifications/MODULAR_ETL_DESIGN.md)
- バグ修正内容を確認したい → [BUG_FIXES_REPORT.md](reports/BUG_FIXES_REPORT.md)
- 技術指標の実装を知りたい → [TECHNICAL_INDICATORS_COMPARISON.md](specifications/TECHNICAL_INDICATORS_COMPARISON.md)

#### 📊 データサイエンティスト向け
- 特徴量の詳細を知りたい → [ML_DATASET_COLUMNS.md](specifications/ML_DATASET_COLUMNS.md)
- データ品質を確認したい → [BUG_FIXES_REPORT.md](reports/BUG_FIXES_REPORT.md)

#### 🏃 運用担当者向け
- パイプライン実行方法 → [PROCESSING_FLOW.md](guides/PROCESSING_FLOW.md)
- トラブルシューティング → [PROCESSING_FLOW.md#トラブルシューティング](guides/PROCESSING_FLOW.md#トラブルシューティング)

## 📝 ドキュメントメンテナンス

### 更新頻度
- **仕様書**: APIや特徴量変更時に更新
- **レポート**: マイルストーン完了時に更新
- **ガイド**: 運用手順変更時に更新

### 命名規則
- 仕様書: `{COMPONENT}_SPEC.md` または `{COMPONENT}_COLUMNS.md`
- レポート: `{TOPIC}_REPORT.md` または `{PROJECT}_SUMMARY.md`
- ガイド: `{PROCESS}_FLOW.md` または `{TASK}_GUIDE.md`

---
*最終更新: 2025年1月27日*
*整理実施者: Claude*
