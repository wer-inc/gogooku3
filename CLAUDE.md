# CLAUDE.md

> **📍 MOVED**: この文書の詳細な開発ガイドは **[docs/development/contributing.md](docs/development/contributing.md)** に移動されました。

## 🚀 Gogooku3 クイックスタート

Gogooku3は日本株式向けMLOpsシステム（v2.0.0）です。

### セットアップ
```bash
make setup && make docker-up && make smoke
```

### 主要コマンド  
```bash
make train-cv              # ML学習実行
make infer                 # 推論実行
make test                  # テスト実行
```

## 📚 ドキュメント

**📋 [docs/index.md](docs/index.md) - メインドキュメントポータル**

### 開発者向け
- **[👥 開発貢献ガイド](docs/development/contributing.md)** ← 旧CLAUDE.md内容
- [🏗️ アーキテクチャ概要](docs/architecture/overview.md)  
- [🧪 テスト実行](docs/development/testing.md)

### 新規参加者向け  
- [🌟 はじめに](docs/getting-started.md)
- [❓ FAQ](docs/faq.md)
- [📚 用語集](docs/glossary.md)

### ML・データ担当者向け
- [🛡️ 安全性ガードレール](docs/ml/safety-guardrails.md)
- [📊 評価メトリクス](docs/ml/metrics.md)

---

## Repository Overview (他プロジェクト)

This workspace contains three projects:

1. **Gogooku2** (`/home/ubuntu/gogooku2/`) - Financial AI/ML System
2. **🌟 Gogooku3-standalone** (`/home/ubuntu/gogooku3-standalone/`) - Enhanced financial ML system ← THIS
3. **Ripipi** (`/home/ubuntu/ripipi/`) - LINE LIFF-based reservation system

*詳細情報は [docs/development/contributing.md](docs/development/contributing.md) を参照*