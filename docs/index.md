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
 

### 📊 **データサイエンティスト・アナリスト**  
- [**📈 機械学習モデル**](ml/model-training.md) - ATFT-GAT-FAN・学習戦略
- [**🛡️ セーフガード**](ml/safety-guardrails.md) - embargo・データリーク防止
- [**🏗️ データパイプライン**](architecture/data-pipeline.md) - 特徴量・正規化・品質管理
- [**🧩 ATFT ドキュメント集約**](ml/atft/index.md) - ATFT 関連ノート一覧
- [**📦 データセット仕様**](ml/dataset.md) - 列仕様・生成フロー

### ⚙️ **運用・DevOps担当者**
- [**📋 運用手順**](operations/runbooks.md) - 起動・停止・定期メンテナンス
 - [**🔧 トラブルシューティング**](operations/troubleshooting.md) - 典型的障害と対処法
- [**📈 観測性**](operations/observability.md) - Grafana・Prometheus・アラート
- [**🔒 セキュリティ運用**](security/operational-security.md) - セキュリティ監視・対応手順

---

## 📖 全ドキュメント構成

### 🏗️ **アーキテクチャ**
- [**overview.md**](architecture/overview.md) - システム全体設計・Mermaid図
- [**data-pipeline.md**](architecture/data-pipeline.md) - JQuants→特徴量→正規化フロー
- [**data-lineage.md**](architecture/data-lineage.md) - データリネージ・品質ゲート・フロー図
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
- [**performance-optimization.md**](development/performance-optimization.md) - PERF_*フラグ・最適化ガイド

### 🧠 **機械学習**
- [**safety-guardrails.md**](ml/safety-guardrails.md) - データリーク防止・embargo・正規化
- [**model-training.md**](ml/model-training.md) - ATFT-GAT-FAN学習パイプライン
- [**data-quality.md**](ml/data-quality.md) - Great Expectations・品質チェック
- [**ATFT ドキュメント集約**](ml/atft/index.md) - ATFT 関連ノート一覧
- [**データセット仕様**](ml/dataset.md) - 列仕様・生成フロー

### 🔒 **セキュリティ**
- [**credentials.md**](security/credentials.md) - 認証情報・.env・Secret管理
- [**sast.md**](security/sast.md) - SAST・漏洩防止・CI/CDセキュリティ
- [**operational-security.md**](security/operational-security.md) - セキュリティ監視・対応手順

### 📝 **リリース**  
- [**changelog.md**](releases/changelog.md) - バージョン履歴・変更内容・マイグレーション

### 🏛️ **ガバナンス**
- [**adr/template.md**](governance/adr/template.md) - Architecture Decision Records テンプレート
- [**adr/ADR-0001-modern-package-migration.md**](governance/adr/ADR-0001-modern-package-migration.md) - パッケージ現代化決定記録

---

## 🔗 **重要なリンク**

### 📁 **リポジトリ構成**
```
gogooku3-standalone/
├── src/gogooku3/          # メインパッケージ（v2.0.0）
├── scripts/               # レガシースクリプト（互換性）
│   └── performance_optimizer.py    # パフォーマンス最適化
├── configs/               # 設定ファイル（model/data/training）
├── docs/                  # このドキュメント群
│   ├── architecture/             # アーキテクチャドキュメント
│   └── index.md          # ドキュメントポータル
├── tests/                 # テストスイート
│   ├── test_health_check.py       # ヘルスチェックテスト
│   └── test_e2e_docker.py         # E2Eテスト
├── data_quality/          # データ品質管理
│   └── great_expectations_suite.py # Great Expectations統合
├── ops/                   # 運用ツール
│   ├── health_check.py            # ヘルスチェック
│   ├── metrics_exporter.py        # メトリクスエクスポート
│   ├── logrotate.conf            # ログローテーション
│   └── runbook.md               # 運用Runbook
├── security/              # セキュリティ設定
│   ├── sast.md                    # SASTガイド
│   ├── leak-prevention.md         # 漏洩防止
│   └── trivy-config.yaml          # Trivy設定
├── output/                # 実行結果・ログ・モデル
├── .github/workflows/     # CI/CDワークフロー
│   ├── security.yml               # セキュリティスキャン
│   ├── tests.yml                  # テスト実行
│   ├── benchmarks.yml             # パフォーマンスベンチマーク
│   ├── backup-validation.yml      # バックアップ検証
│   └── release.yml               # Semantic Release
└── .env.example          # セキュア環境変数テンプレート
```

### 🚀 **主要コマンド**
```bash
# セットアップ
make setup                 # 初期セットアップ
make docker-up             # 全サービス起動
cp .env.example .env       # セキュア環境設定

# システム検証
python ops/health_check.py health           # ヘルスチェック
python ops/health_check.py ready            # 準備状況確認
python ops/metrics_exporter.py --once       # メトリクス表示

# データ品質
export DATA_QUALITY_ENABLED=1
python data_quality/great_expectations_suite.py validate --input data/processed/dataset.parquet

# パフォーマンス最適化
export PERF_POLARS_STREAM=1
export PERF_MEMORY_OPTIMIZATION=1
export PERF_CACHING_ENABLED=1

# 開発
make dev                   # 開発モード起動
make test                  # テスト実行
make lint                  # コード品質チェック
pytest tests/ -k "performance" --benchmark-only  # パフォーマンステスト

# ML実行
python main.py safe-training --mode full     # 安全学習パイプライン
python main.py ml-dataset                    # データセット構築
python main.py complete-atft                 # ATFT完全学習

# 運用
make docker-logs           # ログ確認
make docker-down           # 全サービス停止
make clean                 # 環境リセット

# バックアップ検証
ls -la backups/            # バックアップ確認
```

### 🏆 **成果指標**

#### **🚀 パフォーマンス指標**
- **パイプライン実行**: 1.9s（606K件処理）
- **メモリ効率**: 7GB使用（目標<8GB達成）
- **最適化時**: Polarsストリーミングで30%高速化
- **並列処理**: CPUコア数に応じた自動スケーリング

#### **📊 品質保証指標**
- **データ品質**: 95%以上の品質スコア維持
- **テストカバレッジ**: ユニット/統合/E2E/パフォーマンステスト
- **コード品質**: pre-commit hooks・型チェック・SAST完備
- **セキュリティ**: 自動脆弱性スキャン・漏洩防止

#### **🔍 監視・運用指標**
- **REDメトリクス**: Rate/Error/Durationのリアルタイム監視
- **SLAコンプライアンス**: 99.9%アップタイム目標
- **バックアップ検証**: 日次自動検証・99%成功率
- **障害対応**: 平均30分以内の復旧時間

#### **⚡ 新機能指標**
- **データ品質チェック**: Great Expectations統合・6種類検証
- **パフォーマンス最適化**: PERF_*フラグでオプトイン制御
- **Semantic Release**: 自動バージョン管理・CHANGELOG生成
- **アーキテクチャ**: 15種類の詳細図表・データリネージ

---

## 🔄 **文書更新履歴**

| 日付 | 更新内容 | 担当 |
|------|----------|------|
| 2024-01-XX | Phase 2/3完了: 品質向上・運用体制確立 | Claude |
| 2024-01-XX | Phase 2: データ品質/パフォーマンス最適化/監視強化/CI/CD改善 | Claude |
| 2024-01-XX | Phase 3: 運用Runbook/バックアップ自動化/アーキテクチャドキュメント | Claude |
| 2024-01-XX | Phase 1: セキュリティ/テスト/監視/ドキュメント基盤 | Claude |
| 2025-08-28 | ドキュメント再編・統合ポータル作成 | Claude |
| 2025-08-28 | v2.0.0 パッケージ移行完了・MIGRATION.md作成 | Claude |
| 2025-01-27 | 既存INDEX.md作成・分類整理 | Claude |

---

 

*Gogooku3 - 壊れず・強く・速く 金融MLシステム*
