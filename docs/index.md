# 🏛️ Gogooku3 ドキュメントポータル

<!-- TOC -->

Gogooku3は日本株式向けMLOpsシステムです。JQuants API → 特徴量生成 → ATFT-GAT-FAN → Feast/Redis → 推論のパイプラインを提供します。

## 🎯 役割別クイックナビゲーション

### 👨‍💻 **開発者**
- [**📋 開発ガイド**](development/contributing.md) - コード規約・PR・開発フロー
- [**🏗️ アーキテクチャ概要**](architecture/overview.md) - システム全体設計
- [**🧪 テスト実行**](development/testing.md) - 単体・結合・スモークテスト
- [**🐳 ローカル環境構築**](development/local-environment.md) - Docker・Make使用方法

### 🚀 **新規参加者**
- [**🌟 はじめに**](getting-started.md) - セットアップ・最初の実行
- [**📚 用語集**](glossary.md) - JQuants・JPX・金融ML用語
- [**❓ FAQ**](faq.md) - よくある質問と回答
- [**🔄 移行ガイド**](migration.md) - 旧システムからの移行

### 📊 **データサイエンティスト・アナリスト**  
- [**📈 機械学習モデル**](ml/model-training.md) - ATFT-GAT-FAN・学習戦略
- [**📊 メトリクス**](ml/metrics.md) - IC・RankIC・Decile・Sharpe
- [**🛡️ セーフガード**](ml/safety-guardrails.md) - embargo・データリーク防止
- [**🏗️ データパイプライン**](architecture/data-pipeline.md) - 特徴量・正規化・品質管理

### ⚙️ **運用・DevOps担当者**
- [**📋 運用手順**](operations/runbooks.md) - 起動・停止・定期メンテナンス
- [**🔧 トラブルシューティング**](operations/troubleshooting.md) - 典型的障害と対処法  
- [**📈 観測性**](operations/observability.md) - Grafana・Prometheus・アラート
- [**🗄️ ストレージ**](architecture/storage.md) - MinIO・ClickHouse・Redis・PostgreSQL

---

## 📖 全ドキュメント構成

### 🏗️ **アーキテクチャ**
- [**overview.md**](architecture/overview.md) - システム全体設計・Mermaid図
- [**data-pipeline.md**](architecture/data-pipeline.md) - JQuants→特徴量→正規化フロー  
- [**model.md**](architecture/model.md) - ATFT-GAT-FAN・学習アーキテクチャ
- [**orchestration.md**](architecture/orchestration.md) - Dagster（assets/jobs/schedules）
- [**feature-store.md**](architecture/feature-store.md) - Feast・オンライン特徴量配信
- [**storage.md**](architecture/storage.md) - データストレージ戦略

### ⚙️ **運用**
- [**runbooks.md**](operations/runbooks.md) - 標準運用手順・チェックリスト
- [**troubleshooting.md**](operations/troubleshooting.md) - 障害対応・復旧手順
- [**observability.md**](operations/observability.md) - 監視・アラート・ダッシュボード

### 👨‍💻 **開発**  
- [**contributing.md**](development/contributing.md) - 開発フロー・PR規約・コード品質
- [**conventions.md**](development/conventions.md) - 命名・構成・コミット規約  
- [**testing.md**](development/testing.md) - テスト戦略・CI/CD
- [**local-environment.md**](development/local-environment.md) - Docker・Make・デバッグ

### 🧠 **機械学習**
- [**metrics.md**](ml/metrics.md) - 金融ML評価指標（IC/RankIC/Decile/Sharpe）
- [**safety-guardrails.md**](ml/safety-guardrails.md) - データリーク防止・embargo・正規化
- [**model-training.md**](ml/model-training.md) - ATFT-GAT-FAN学習パイプライン

### 🔒 **セキュリティ**
- [**credentials.md**](security/credentials.md) - 認証情報・.env・Secret管理

### 📝 **リリース**  
- [**changelog.md**](releases/changelog.md) - バージョン履歴・変更内容・マイグレーション

### 🏛️ **ガバナンス**
- [**adr/template.md**](governance/adr/template.md) - Architecture Decision Records テンプレート
- [**adr/ADR-0001-modern-package-migration.md**](governance/adr/ADR-0001-modern-package-migration.md) - パッケージ現代化決定記録

### 🌍 **多言語版**
- [**🇯🇵 日本語版**](ja/index.md) - メインドキュメントの日本語版
- [**🇺🇸 English**](index.md) - Main documentation in English

---

## 🔗 **重要なリンク**

### 📁 **リポジトリ構成**
```
gogooku3-standalone/
├── src/gogooku3/          # メインパッケージ（v2.0.0）
├── scripts/               # レガシースクリプト（互換性）
├── configs/               # 設定ファイル（model/data/training）
├── docs/                  # このドキュメント群
├── tests/                 # テストスイート
└── output/                # 実行結果・ログ・モデル
```

### 🚀 **主要コマンド**
```bash
# セットアップ
make setup                 # 初期セットアップ
make docker-up             # 全サービス起動

# 開発
make dev                   # 開発モード起動  
make test                  # テスト実行
make lint                  # コード品質チェック

# ML実行
make train-cv              # クロスバリデーション学習
make infer                 # 推論実行
make smoke                 # スモークテスト

# 運用
make docker-logs           # ログ確認
make docker-down           # 全サービス停止
make clean                 # 環境リセット
```

### 🏆 **成果指標**
- **パフォーマンス**: 1.9s パイプライン実行（606K件処理）
- **メモリ効率**: 7GB使用（目標<8GB達成）
- **データ品質**: Walk-Forward + 20日embargo実装
- **コード品質**: pre-commit hooks・型チェック完備

---

## 🔄 **文書更新履歴**

| 日付 | 更新内容 | 担当 |
|------|----------|------|
| 2025-08-28 | ドキュメント再編・統合ポータル作成 | Claude |
| 2025-08-28 | v2.0.0 パッケージ移行完了・MIGRATION.md作成 | Claude |
| 2025-01-27 | 既存INDEX.md作成・分類整理 | Claude |

---

**🇺🇸 [English](index.md) | 🇯🇵 [日本語](ja/index.md)**

*Gogooku3 - 壊れず・強く・速く 金融MLシステム*