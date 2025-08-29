# ⚠️ このファイルは移行されました

**新しいドキュメントポータル**: [docs/index.md](index.md) をご利用ください。

## 🔄 ドキュメント再構成完了 (2025-08-28)

Gogooku3のドキュメントは全面的に再編成され、以下の新しい構造になりました：

### 📍 新しいメインポータル
- **[📖 docs/index.md](index.md)** - 役割別ナビゲーション・全体索引

### 🗂️ 主要ドキュメント新配置

**ML・評価:**
- **[📊 ML評価メトリクス](ml/metrics.md)** ← 旧 specifications/TECHNICAL_INDICATORS_COMPARISON.md
- **[🛡️ 安全性ガードレール](ml/safety-guardrails.md)** ← 新規作成（データリーク防止）
- **[🧠 モデル学習](ml/model-training.md)** ← 新規作成（ATFT-GAT-FAN詳細）

**アーキテクチャ:**
- **[🏗️ データパイプライン](architecture/data-pipeline.md)** ← 旧 specifications/ 統合
- **[🏛️ システム概要](architecture/overview.md)** ← 新規作成

**開発・運用:**
- **[👥 開発ガイド](development/contributing.md)** ← 旧 CLAUDE.md大幅リファクタリング
- **[📋 変更履歴](releases/changelog.md)** ← 旧 reports/ 統合

**基本情報:**
- **[🚀 はじめに](getting-started.md)** ← 新規作成（包括的セットアップ）
- **[📚 用語集](glossary.md)** ← 新規作成
- **[❓ FAQ](faq.md)** ← 新規作成

### 🗂️ アーカイブ保管
旧ファイルは安全に保管されています：
- **[docs/_archive/](\_archive/)** - 全元ファイル保管

## 📋 旧ファイル対応表

| 旧パス | 新パス | 内容 |
|--------|--------|------|
| specifications/TECHNICAL_INDICATORS_COMPARISON.md | [ml/metrics.md](ml/metrics.md) | 技術指標・評価メトリクス |
| specifications/ML_DATASET_COLUMNS.md | [_archive/specifications_original/](\_archive/specifications_original/) | アーカイブ保管 |
| specifications/MODULAR_ETL_DESIGN.md | [architecture/data-pipeline.md](architecture/data-pipeline.md) | データパイプライン設計 |
| reports/ | [releases/changelog.md](releases/changelog.md) | 変更履歴・リリース情報 |
| guides/PROCESSING_FLOW.md | [getting-started.md](getting-started.md) | セットアップ・使用方法 |
| CLAUDE.md | [development/contributing.md](development/contributing.md) | 開発ガイド |

---

**🔄 完全移行完了日**: 2025年8月28日  
**📖 新ドキュメントポータル**: [index.md](index.md)  

このファイル（INDEX.md）は参照用として残していますが、最新情報は新しいポータルをご利用ください。
